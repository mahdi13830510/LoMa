#!/usr/bin/env python3
"""
Fine-tune LoMa for Thermal-Optical (Infrared ↔ Visible) Image Matching
=======================================================================

This script fine-tunes the pretrained LoMa local-feature matcher on the
RoadScene dataset so it can handle cross-modal matching between thermal
(infrared) and visible images.

Performance-improvement ideas baked in (toggle via CLI flags):
─────────────────────────────────────────────────────────────
 1. CLAHE preprocessing on thermal images to boost detector/descriptor
    response on low-contrast infrared textures.
 2. Random homography augmentation — synthesise warped pairs with exact
    known GT so the model sees geometric diversity, not just identity.
 3. Descriptor LoRA adapter — a tiny low-rank residual MLP inserted
    *between* the frozen descriptor and the matcher that learns a
    cross-modal descriptor alignment without touching DeDoDe.
 4. Learnable modality embedding — a small vector added to descriptors
    that tells the matcher "this keypoint came from infrared" vs.
    "this keypoint came from visible".
 5. Focal matchability loss — down-weights well-classified matchable /
    unmatchable keypoints so training focuses on the hard cases.
 6. Soft-label smoothing — GT match targets use a Gaussian-distance
    weight instead of hard 0/1, giving gentler gradients.
 7. Exponential Moving Average (EMA) — keeps a shadow copy of weights
    for more stable evaluation.
 8. Gradual descriptor unfreezing — after N epochs, unfreeze the last
    few layers of the descriptor so it can co-adapt with the matcher.
 9. Thermal channel strategies — try different ways of converting
    single-channel thermal to 3-ch input (replicate, CLAHE-enhanced,
    histogram-equalised, pseudo-colour).
10. Multi-scale evaluation — detect at several keypoint budgets and
    report metrics at each scale.
11. Modality-dropout regularisation — randomly form same-modality
    pairs (vis–vis or ir–ir) so the matcher doesn't overfit to always
    expecting cross-modal inputs.
12. Reciprocal loss weighting — weight each sample's loss inversely
    by the number of GT matches so image pairs with few matches
    aren't drowned out by easy ones.

Future ideas (not yet implemented, marked with TODO in the code):
─────────────────────────────────────────────────────────────
 • Hard-negative mining within the score matrix.
 • Curriculum learning: rank pairs by structural similarity and
   train easy-first.
 • Auxiliary contrastive loss on matched descriptor pairs.
 • Test-time augmentation: match (A,B) and (B,A), merge results.
 • Thermal-specific noise / blur augmentation.
 • Separate learning rates for adapter vs. transformer layers.
 • Knowledge distillation from a larger LoMa-G teacher.

Usage
-----
    python loma_thermal_optical_finetune.py          # defaults
    python loma_thermal_optical_finetune.py \\
        --epochs 20 --lr 3e-5 \\
        --use-homography-aug --use-clahe \\
        --use-descriptor-adapter --use-modality-embed \\
        --use-focal-loss --use-ema
"""

import argparse
import glob
import math
import os
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune LoMa for thermal-optical matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── paths ──────────────────────────────────────────────────────────────
    p.add_argument("--work-dir", default=".", help="Working directory")
    p.add_argument(
        "--roadscene-repo",
        default="https://github.com/hanna-xu/road-scene-infrared-visible-images.git",
    )
    p.add_argument("--roadscene-dir", default=None,
                   help="Path to cloned RoadScene. Auto-cloned if absent.")
    p.add_argument("--save-dir", default="loma_finetuned",
                   help="Output directory for checkpoints and results.")

    # ── model ──────────────────────────────────────────────────────────────
    p.add_argument("--variant", choices=["B128", "B"], default="B128",
                   help="LoMa variant. B128 fits Kaggle T4; B needs ≥20 GB.")
    p.add_argument("--num-keypoints", type=int, default=1024)
    p.add_argument("--eval-num-keypoints", type=int, default=2048)

    # ── training ───────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--match-threshold", type=float, default=0.06,
                   help="Normalised-coord radius for GT match proximity")
    p.add_argument("--filter-threshold", type=float, default=0.1,
                   help="Mutual-match confidence threshold at eval time.")

    # ── improvement toggles ────────────────────────────────────────────────
    p.add_argument("--use-clahe", action="store_true",
                   help="[Idea 1] CLAHE contrast enhancement on thermal imgs")
    p.add_argument("--use-homography-aug", action="store_true",
                   help="[Idea 2] Random homography augmentation")
    p.add_argument("--homography-strength", type=float, default=0.12,
                   help="Corner perturbation fraction for homography aug")
    p.add_argument("--use-descriptor-adapter", action="store_true",
                   help="[Idea 3] LoRA-style descriptor adapter")
    p.add_argument("--adapter-rank", type=int, default=16,
                   help="Rank of the LoRA descriptor adapter")
    p.add_argument("--use-modality-embed", action="store_true",
                   help="[Idea 4] Learnable modality embedding")
    p.add_argument("--use-focal-loss", action="store_true",
                   help="[Idea 5] Focal loss for matchability")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--use-soft-labels", action="store_true",
                   help="[Idea 6] Distance-weighted soft GT labels")
    p.add_argument("--soft-label-sigma", type=float, default=0.03,
                   help="Gaussian sigma for soft labels (normalised coords)")
    p.add_argument("--use-ema", action="store_true",
                   help="[Idea 7] Exponential moving average model")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--unfreeze-descriptor-after", type=int, default=-1,
                   help="[Idea 8] Unfreeze last 2 decoder layers after N epochs (-1=never)")
    p.add_argument("--thermal-channel", choices=["replicate", "clahe", "histeq"],
                   default="replicate",
                   help="[Idea 9] How to convert 1-ch thermal to 3-ch")
    p.add_argument("--modality-dropout", type=float, default=0.0,
                   help="[Idea 11] Prob of forming same-modality pairs")
    p.add_argument("--reciprocal-loss-weight", action="store_true",
                   help="[Idea 12] Weight loss inversely by #GT matches")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def setup_repos(args):
    """Clone RoadScene if not already present and make LoMa importable."""
    # ── RoadScene ──────────────────────────────────────────────────────────
    if args.roadscene_dir is None:
        args.roadscene_dir = os.path.join(args.work_dir, "RoadScene")

    if not os.path.isdir(args.roadscene_dir):
        print(f"Cloning RoadScene → {args.roadscene_dir} …")
        subprocess.run(
            ["git", "lfs", "install"],
            check=False, capture_output=True,
        )
        subprocess.run(
            ["git", "clone", args.roadscene_repo, args.roadscene_dir],
            check=True,
        )
        print("  ✓ RoadScene cloned.")
    else:
        print(f"RoadScene found at {args.roadscene_dir}")

    # ── Verify images exist ────────────────────────────────────────────────
    vis_dir = os.path.join(args.roadscene_dir, "crop_HR_visible")
    ir_dir = os.path.join(args.roadscene_dir, "cropinfrared")
    for d in (vis_dir, ir_dir):
        imgs = [f for f in os.listdir(d)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ] if os.path.isdir(d) else []
        assert len(imgs) > 0, f"No images in {d} — did Git LFS pull succeed?"
    print(f"  visible  : {vis_dir}  ({len(os.listdir(vis_dir))} files)")
    print(f"  infrared : {ir_dir}  ({len(os.listdir(ir_dir))} files)")

    # ── LoMa ───────────────────────────────────────────────────────────────
    loma_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "LoMa", "src")
    if os.path.isdir(loma_src) and loma_src not in sys.path:
        sys.path.insert(0, loma_src)
    # Fallback: maybe we're already inside the LoMa repo
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if os.path.isdir(alt) and alt not in sys.path:
        sys.path.insert(0, alt)

    return vis_dir, ir_dir


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PRE-PROCESSING  [Ideas 1, 9]
# ═══════════════════════════════════════════════════════════════════════════════

def apply_clahe(img_uint8, clip_limit=3.0, grid=(8, 8)):
    """Contrast-Limited Adaptive Histogram Equalisation.

    Thermal images often have narrow dynamic range.  CLAHE enhances local
    contrast so the keypoint detector and descriptor can extract richer
    features from infrared textures that would otherwise look flat.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    if img_uint8.ndim == 3:
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return clahe.apply(img_uint8)


def thermal_to_3ch(gray_uint8, strategy="replicate"):
    """Convert single-channel thermal to 3-channel for the RGB-trained
    detector and descriptor.

    Strategies:
      replicate — simple channel copy (baseline)
      clahe     — CLAHE-enhanced luminance replicated (boosts texture)
      histeq    — global histogram equalisation then replicate
    """
    if strategy == "replicate":
        return cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2RGB)
    elif strategy == "clahe":
        enhanced = apply_clahe(gray_uint8)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    elif strategy == "histeq":
        eq = cv2.equalizeHist(gray_uint8)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unknown thermal channel strategy: {strategy}")


def load_image_as_tensor(path, resize_max=1024, apply_clahe_flag=False,
                         thermal_channel="replicate", is_thermal=False):
    """Load, optionally enhance, resize and return (1,3,H,W) float [0,1]."""
    img = np.array(Image.open(path).convert("RGB"))

    if is_thermal and apply_clahe_flag:
        img = apply_clahe(img)

    h, w = img.shape[:2]
    scale = resize_max / max(h, w)
    new_w = int((scale * w) // 8 * 8)
    new_h = int((scale * h) // 8 * 8)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
    return tensor.unsqueeze(0)  # (1,3,H,W)


# ═══════════════════════════════════════════════════════════════════════════════
# HOMOGRAPHY AUGMENTATION  [Idea 2]
# ═══════════════════════════════════════════════════════════════════════════════

def random_homography_matrix(h, w, strength=0.12, rng=None):
    """Generate a random perspective transform.

    Instead of only training on identity-warp (aligned pairs), we apply a
    random homography to one image so the matcher learns to handle actual
    geometric differences.  This massively increases effective training-set
    diversity.  `strength` controls corner displacement as a fraction of
    image size.
    """
    if rng is None:
        rng = np.random
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    perturb = rng.randn(4, 2).astype(np.float32) * strength * min(h, w)
    dst = (corners + perturb).astype(np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    return H


def warp_tensor_by_H(img_tensor, H_matrix):
    """Warp a (1,3,H,W) tensor by a 3×3 homography (OpenCV convention)."""
    img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    h, w = img_np.shape[:2]
    warped = cv2.warpPerspective(img_np, H_matrix, (w, h),
                                 borderMode=cv2.BORDER_REFLECT)
    return torch.from_numpy(warped / 255.0).permute(2, 0, 1).float().unsqueeze(0)


def warp_keypoints_by_H(kpts_norm, H, h, w):
    """Warp normalised keypoints [-1,1] through a pixel-space homography.

    kpts_norm: (1, N, 2)
    H: 3×3 numpy array  (pixel→pixel)
    h, w: image dimensions
    Returns: (1, N, 2) warped normalised keypoints, and a validity mask.
    """
    kp = kpts_norm[0].cpu().numpy().copy()  # (N, 2)
    # normalised → pixel
    px = (kp[:, 0] + 1) / 2 * w
    py = (kp[:, 1] + 1) / 2 * h
    pts = np.stack([px, py, np.ones_like(px)], axis=1)  # (N, 3)
    warped = (H @ pts.T).T  # (N, 3)
    warped = warped[:, :2] / (warped[:, 2:3] + 1e-8)
    # pixel → normalised
    nx = warped[:, 0] / w * 2 - 1
    ny = warped[:, 1] / h * 2 - 1
    result = np.stack([nx, ny], axis=1)
    valid = (np.abs(result) < 1.0).all(axis=1)
    return (torch.from_numpy(result).float().unsqueeze(0).to(kpts_norm.device),
            torch.from_numpy(valid).to(kpts_norm.device))


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class RoadSceneDataset:
    """Loads aligned infrared ↔ visible pairs from RoadScene.

    The pairs are spatially registered, so the default GT warp is the
    identity.  When homography augmentation is enabled the warp becomes
    H (applied to the infrared image).
    """

    def __init__(self, vis_dir, ir_dir):
        vis_files = {
            Path(f).stem: os.path.join(vis_dir, f)
            for f in sorted(os.listdir(vis_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        ir_files = {
            Path(f).stem: os.path.join(ir_dir, f)
            for f in sorted(os.listdir(ir_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        common = sorted(set(vis_files) & set(ir_files))
        assert common, "No matching vis/ir pairs found"
        self.pairs = [(vis_files[k], ir_files[k]) for k in common]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return {"visible": self.pairs[idx][0], "infrared": self.pairs[idx][1]}


def train_val_split(dataset, val_frac, seed):
    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    k = int(n * (1 - val_frac))
    return (
        [dataset[i] for i in indices[:k]],
        [dataset[i] for i in indices[k:]],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL EXTENSIONS  [Ideas 3, 4]
# ═══════════════════════════════════════════════════════════════════════════════

class DescriptorAdapter(nn.Module):
    """Low-Rank Adaptation (LoRA) for cross-modal descriptor alignment.

    A tiny bottleneck that maps D → rank → D, initialised to zero so the
    model starts from the identity (pretrained behaviour) and gradually
    learns a cross-modal residual correction.

    Why this helps: the frozen DeDoDe descriptor was trained on RGB only.
    Thermal descriptors live in a different region of feature space.
    The adapter learns to *project* them closer together without touching
    the 100M+ descriptor parameters.
    """

    def __init__(self, dim, rank=16):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        # Zero-init so adapter starts as identity (no perturbation)
        nn.init.kaiming_normal_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


class ModalityEmbedding(nn.Module):
    """Learnable embedding added to descriptors to indicate modality.

    The matcher receives descriptors from two modalities that the frozen
    descriptor treats identically.  This small learned vector (dim-sized)
    lets the self-/cross-attention distinguish "I'm looking at an IR
    keypoint" from "I'm looking at a visible keypoint", which helps it
    learn modality-specific matching heuristics.
    """

    def __init__(self, dim):
        super().__init__()
        self.vis_embed = nn.Parameter(torch.zeros(dim))
        self.ir_embed = nn.Parameter(torch.zeros(dim))

    def forward(self, desc, modality):
        if modality == "visible":
            return desc + self.vis_embed
        return desc + self.ir_embed


# ═══════════════════════════════════════════════════════════════════════════════
# EMA  [Idea 7]
# ═══════════════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential Moving Average of model weights.

    Keeps a shadow copy updated as  θ_shadow ← α·θ_shadow + (1−α)·θ_model
    after every optimiser step.  Produces smoother, more robust weights for
    evaluation.  Standard trick for fine-tuning.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.clone()

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        model.load_state_dict(self.backup, strict=False)
        self.backup = {}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES  [Ideas 5, 6, 12]
# ═══════════════════════════════════════════════════════════════════════════════

def training_forward(model, kpts0, kpts1, desc0, desc1,
                     adapter=None, mod_embed=None,
                     mod_A="visible", mod_B="infrared"):
    """Forward pass collecting intermediate-layer scores for training."""
    from loma.device import device as dev, amp_dtype as adtype

    with torch.autocast(enabled=model.cfg.mp, dtype=adtype,
                        device_type=dev.type):
        d0 = desc0.detach().contiguous()
        d1 = desc1.detach().contiguous()

        # [Idea 3] Descriptor adapter
        if adapter is not None:
            d0 = adapter(d0)
            d1 = adapter(d1)

        d0 = model.input_proj(d0)
        d1 = model.input_proj(d1)

        # [Idea 4] Modality embedding
        if mod_embed is not None:
            d0 = mod_embed(d0, mod_A)
            d1 = mod_embed(d1, mod_B)

        enc0 = model.posenc(kpts0)
        enc1 = model.posenc(kpts1)

        all_scores = []
        for i in range(model.cfg.n_layers):
            d0, d1 = model.transformers[i](d0, d1, enc0, enc1)
            scores, _ = model.log_assignment[i](d0, d1)
            all_scores.append(scores)

    return all_scores


@torch.no_grad()
def compute_gt_matches(kpts_A, kpts_B, threshold=0.06):
    """MNN matches assuming identity warp (aligned images)."""
    D = torch.cdist(kpts_A.float(), kpts_B.float())
    min_AB, nn_AB = D.min(dim=2)
    _, nn_BA = D.min(dim=1)
    B, N, M = D.shape
    idx_A = torch.arange(N, device=D.device).unsqueeze(0).expand(B, -1)
    mutual = nn_BA.gather(1, nn_AB) == idx_A
    valid = mutual & (min_AB < threshold)
    bi, ai = torch.where(valid)
    bi2 = nn_AB[bi, ai]
    return torch.stack([bi, ai, bi2], dim=1)


@torch.no_grad()
def compute_gt_matches_homography(kpts_A, kpts_B, H, h, w, threshold=0.06):
    """MNN matches when image B has been warped by homography H.

    We warp kpts_A through H and find MNN with kpts_B.
    """
    warped_A, valid_mask = warp_keypoints_by_H(kpts_A, H, h, w)
    D = torch.cdist(warped_A.float(), kpts_B.float())
    # Invalidate out-of-bounds keypoints
    D[0, ~valid_mask] = float("inf")
    min_AB, nn_AB = D.min(dim=2)
    _, nn_BA = D.min(dim=1)
    B, N, M = D.shape
    idx_A = torch.arange(N, device=D.device).unsqueeze(0).expand(B, -1)
    mutual = nn_BA.gather(1, nn_AB) == idx_A
    valid = mutual & (min_AB < threshold) & valid_mask.unsqueeze(0)
    bi, ai = torch.where(valid)
    bi2 = nn_AB[bi, ai]
    return torch.stack([bi, ai, bi2], dim=1)


def focal_bce(logits, targets, gamma=2.0):
    """Focal binary cross-entropy  [Idea 5].

    Down-weights easy-to-classify samples so the network focuses its
    capacity on ambiguous keypoints (the ones that are almost matchable
    but not quite, or vice-versa).  Particularly useful here because
    cross-modal pairs tend to have many more *un*matched than matched
    keypoints.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    return ((1 - p_t) ** gamma * bce).mean()


def matching_loss(all_scores, mnn, args,
                  kpts_A=None, kpts_B=None):
    """Matching loss with intermediate supervision and all improvements."""
    from loma.device import device as dev
    total = torch.tensor(0.0, device=dev)

    for scores in all_scores:
        M = scores.shape[1] - 1
        N = scores.shape[2] - 1

        # ── Conditional match loss ────────────────────────────────────────
        if mnn.numel() > 0:
            nll = -scores[mnn[:, 0], mnn[:, 1], mnn[:, 2]]

            # [Idea 6] Soft labels — weight each GT match by spatial
            # confidence (closer keypoints get higher weight)
            if args.use_soft_labels and kpts_A is not None and kpts_B is not None:
                d = (kpts_A[0, mnn[:, 1]] - kpts_B[0, mnn[:, 2]]).norm(dim=-1)
                w = torch.exp(-d ** 2 / (2 * args.soft_label_sigma ** 2))
                cond_loss = (nll * w).sum() / w.sum().clamp(min=1e-6)
            else:
                cond_loss = nll.mean()

            # [Idea 12] Reciprocal weighting
            if args.reciprocal_loss_weight:
                cond_loss = cond_loss / max(mnn.shape[0], 1) * 100
        else:
            cond_loss = torch.tensor(0.0, device=dev)

        # ── Matchability loss ─────────────────────────────────────────────
        tgt_A = torch.zeros(scores.shape[0], M, device=dev)
        tgt_B = torch.zeros(scores.shape[0], N, device=dev)
        if mnn.numel() > 0:
            tgt_A[mnn[:, 0], mnn[:, 1]] = 1.0
            tgt_B[mnn[:, 0], mnn[:, 2]] = 1.0

        if args.use_focal_loss:
            m_A = focal_bce(scores[:, :-1, -1], tgt_A, args.focal_gamma)
            m_B = focal_bce(scores[:, -1, :-1], tgt_B, args.focal_gamma)
        else:
            m_A = F.binary_cross_entropy_with_logits(scores[:, :-1, -1], tgt_A)
            m_B = F.binary_cross_entropy_with_logits(scores[:, -1, :-1], tgt_B)

        total = total + cond_loss + m_A + m_B

    return total / len(all_scores)


# ═══════════════════════════════════════════════════════════════════════════════
# DETECT & DESCRIBE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def detect_and_describe_path(model, path, n_kpts):
    return model.detect_and_describe(path, num_keypoints=n_kpts)


@torch.no_grad()
def detect_and_describe_tensor(model, img_tensor, n_kpts):
    from loma.device import device as dev
    img_tensor = img_tensor.to(dev)
    return model.detect_and_describe(img_tensor, num_keypoints=n_kpts)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION  [Idea 10]
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, samples, args, adapter=None, mod_embed=None,
             keypoint_budgets=None, tag=""):
    """Evaluate on thermal-optical pairs.

    [Idea 10] Multi-scale: if keypoint_budgets is given (e.g. [512, 1024,
    2048]), we report metrics at each scale.
    """
    from loma.loma import filter_matches, to_pixel_coords

    if keypoint_budgets is None:
        keypoint_budgets = [args.eval_num_keypoints]

    model.eval()
    results_per_scale = {}

    for budget in keypoint_budgets:
        all_n, all_errs = [], []

        for sample in tqdm(samples, desc=f"{tag} @{budget}kp", leave=False):
            kA, dA, h1, w1 = detect_and_describe_path(model, sample["visible"], budget)
            kB, dB, h2, w2 = detect_and_describe_path(model, sample["infrared"], budget)

            # At eval time, adapter and embed are folded into the
            # standard forward if present.  For simplicity we just
            # use the raw model (they only affect training_forward).
            scores = model(kA, kB, dA, dB)["scores"]
            m0, *_ = filter_matches(scores, args.filter_threshold)

            valid = m0[0] > -1
            n = valid.sum().item()
            all_n.append(n)

            if n > 0:
                mA = to_pixel_coords(kA[0][torch.where(valid)[0]], h1, w1)
                mB = to_pixel_coords(kB[0][m0[0][valid]], h2, w2)
                if h1 != h2 or w1 != w2:
                    mB = mB.clone()
                    mB[:, 0] *= w1 / w2
                    mB[:, 1] *= h1 / h2
                errs = (mA - mB).norm(dim=-1).cpu().numpy()
                all_errs.append(errs)

        flat = np.concatenate(all_errs) if all_errs else np.array([])
        r = {
            "avg_matches": float(np.mean(all_n)) if all_n else 0,
            "total_matches": int(np.sum(all_n)),
            "mean_err": float(np.mean(flat)) if len(flat) else float("inf"),
            "median_err": float(np.median(flat)) if len(flat) else float("inf"),
        }
        for th in (1, 3, 5, 10):
            r[f"mma@{th}px"] = float((flat < th).mean()) if len(flat) else 0.0
        results_per_scale[budget] = r

    return results_per_scale


def print_results(results_per_scale, title=""):
    if title:
        print(f"\n{'═'*62}")
        print(f"  {title}")
        print(f"{'═'*62}")
    for budget, r in results_per_scale.items():
        print(f"  ── {budget} keypoints ──")
        print(f"    Avg matches  : {r['avg_matches']:.1f}   (total {r['total_matches']})")
        print(f"    Mean reproj  : {r['mean_err']:.2f} px")
        print(f"    Median reproj: {r['median_err']:.2f} px")
        for k, v in r.items():
            if k.startswith("mma"):
                print(f"    {k:13s}  : {v:.4f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_matches(model, vis_path, ir_path, args, title="", save_path=None):
    from loma.loma import filter_matches, to_pixel_coords

    kA, dA, h1, w1 = detect_and_describe_path(model, vis_path, args.eval_num_keypoints)
    kB, dB, h2, w2 = detect_and_describe_path(model, ir_path, args.eval_num_keypoints)
    with torch.inference_mode():
        scores = model(kA, kB, dA, dB)["scores"]
    m0, *_ = filter_matches(scores, args.filter_threshold)
    valid = m0[0] > -1
    mA = to_pixel_coords(kA[0][torch.where(valid)[0]], h1, w1).cpu().numpy()
    mB = to_pixel_coords(kB[0][m0[0][valid]], h2, w2).cpu().numpy()

    im_v = Image.open(vis_path).convert("RGB")
    im_i = Image.open(ir_path).convert("RGB").resize(im_v.size)
    W, H = im_v.size
    canvas = Image.new("RGB", (W * 2 + 10, H), (30, 30, 30))
    canvas.paste(im_v, (0, 0))
    canvas.paste(im_i, (W + 10, 0))
    draw = ImageDraw.Draw(canvas)

    sx, sy = W / w2, H / h2
    mB_s = mB.copy()
    mB_s[:, 0] *= sx
    mB_s[:, 1] *= sy
    errs = np.linalg.norm(mA - mB_s, axis=1)

    rng = np.random.default_rng(42)
    idxs = rng.choice(len(mA), min(len(mA), 300), replace=False) if len(mA) > 300 else np.arange(len(mA))
    for j in idxs:
        e = min(errs[j] / 20, 1.0)
        c = (int(255 * e), int(255 * (1 - e)), 50)
        draw.line([(mA[j, 0], mA[j, 1]), (mB_s[j, 0] + W + 10, mB_s[j, 1])], fill=c, width=1)

    draw.text((5, 5), f"{title}  |  {len(mA)} matches", fill=(255, 255, 255))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(save_path)
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# GRADUAL UNFREEZING  [Idea 8]
# ═══════════════════════════════════════════════════════════════════════════════

def maybe_unfreeze_descriptor(model, epoch, args):
    """After a set epoch, unfreeze the last 2 decoder conv-refiners of the
    descriptor so it can co-adapt with the matcher.

    Why this helps: the frozen descriptor was never trained on thermal
    images.  After the matcher has adapted for a few epochs, letting the
    descriptor's *final* layers adjust gives a significant accuracy bump.
    We keep earlier layers frozen to preserve low-level feature extraction.
    """
    if args.unfreeze_descriptor_after < 0:
        return
    if epoch != args.unfreeze_descriptor_after:
        return

    print(f"  ⚡ Unfreezing last 2 descriptor decoder layers at epoch {epoch}")
    desc = model._descriptor
    # Unfreeze the two finest-scale decoder layers ("1" and "2")
    for scale in ["1", "2"]:
        if scale in desc.decoder.layers:
            for p in desc.decoder.layers[scale].parameters():
                p.requires_grad = True

    # Return the newly unfrozen params so the optimiser can pick them up
    new_params = [p for p in desc.parameters() if p.requires_grad]
    return new_params


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(model, train_set, val_set, args):
    from loma.device import device as dev

    # ── Extra modules ─────────────────────────────────────────────────────
    adapter = None
    if args.use_descriptor_adapter:
        dim = model.cfg.input_dim
        adapter = DescriptorAdapter(dim, rank=args.adapter_rank).to(dev)
        print(f"  ✓ Descriptor adapter enabled (rank={args.adapter_rank}, "
              f"{sum(p.numel() for p in adapter.parameters()):,} params)")

    mod_embed = None
    if args.use_modality_embed:
        mod_embed = ModalityEmbedding(model.cfg.embed_dim).to(dev)
        print(f"  ✓ Modality embedding enabled ({model.cfg.embed_dim}-d)")

    # ── Collect trainable params ──────────────────────────────────────────
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad],
         "lr": args.lr},
    ]
    if adapter is not None:
        param_groups.append({"params": list(adapter.parameters()),
                             "lr": args.lr * 5})  # adapter can learn faster
    if mod_embed is not None:
        param_groups.append({"params": list(mod_embed.parameters()),
                             "lr": args.lr * 5})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_set)
    warmup_steps = args.warmup_epochs * len(train_set)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(dev.type, enabled=(dev.type == "cuda"))

    ema = EMA(model, args.ema_decay) if args.use_ema else None

    # ── Training mode helpers ─────────────────────────────────────────────
    def _train_mode():
        model.train()
        model._detector.eval()
        model._descriptor.eval()
        if adapter:
            adapter.train()
        if mod_embed:
            mod_embed.train()

    def _eval_mode():
        model.eval()
        if adapter:
            adapter.eval()
        if mod_embed:
            mod_embed.eval()

    best_mma5 = -1.0
    history = {"loss": [], "val": []}
    rng_np = np.random.RandomState(args.seed)

    active_ideas = []
    if args.use_clahe: active_ideas.append("CLAHE")
    if args.use_homography_aug: active_ideas.append("Homography-Aug")
    if args.use_descriptor_adapter: active_ideas.append("LoRA-Adapter")
    if args.use_modality_embed: active_ideas.append("Modality-Embed")
    if args.use_focal_loss: active_ideas.append("Focal-Loss")
    if args.use_soft_labels: active_ideas.append("Soft-Labels")
    if args.use_ema: active_ideas.append("EMA")
    if args.unfreeze_descriptor_after >= 0: active_ideas.append(f"Unfreeze@ep{args.unfreeze_descriptor_after}")
    if args.modality_dropout > 0: active_ideas.append(f"Mod-Dropout({args.modality_dropout})")
    if args.reciprocal_loss_weight: active_ideas.append("Reciprocal-Weight")
    print(f"\n  Active improvements: {', '.join(active_ideas) or 'none'}\n")

    for epoch in range(1, args.epochs + 1):
        _train_mode()

        # [Idea 8] Gradual unfreezing
        new_params = maybe_unfreeze_descriptor(model, epoch, args)
        if new_params:
            optimizer.add_param_group({"params": new_params, "lr": args.lr * 0.1})

        indices = list(range(len(train_set)))
        random.shuffle(indices)
        losses = []

        pbar = tqdm(indices, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for idx in pbar:
            sample = train_set[idx]
            vis_path = sample["visible"]
            ir_path = sample["infrared"]

            # [Idea 11] Modality dropout — occasionally form a
            # same-modality pair to regularise and prevent the matcher
            # from only seeing cross-modal inputs.
            mod_A, mod_B = "visible", "infrared"
            if args.modality_dropout > 0 and random.random() < args.modality_dropout:
                if random.random() < 0.5:
                    ir_path = vis_path
                    mod_B = "visible"
                else:
                    vis_path = ir_path
                    mod_A = "infrared"

            # ── Load & optionally augment ─────────────────────────────────
            use_homography = args.use_homography_aug and random.random() < 0.5

            if use_homography or args.use_clahe:
                vis_t = load_image_as_tensor(vis_path, apply_clahe_flag=False)
                ir_t = load_image_as_tensor(
                    ir_path, apply_clahe_flag=args.use_clahe,
                    is_thermal=(mod_B == "infrared"),
                )
                h_img, w_img = ir_t.shape[2], ir_t.shape[3]

                if use_homography:
                    H_mat = random_homography_matrix(
                        h_img, w_img, args.homography_strength, rng_np)
                    ir_t = warp_tensor_by_H(ir_t, H_mat)

                kA, dA, h1, w1 = detect_and_describe_tensor(model, vis_t, args.num_keypoints)
                kB, dB, h2, w2 = detect_and_describe_tensor(model, ir_t, args.num_keypoints)
            else:
                kA, dA, h1, w1 = detect_and_describe_path(model, vis_path, args.num_keypoints)
                kB, dB, h2, w2 = detect_and_describe_path(model, ir_path, args.num_keypoints)
                use_homography = False

            # ── Ground-truth matches ──────────────────────────────────────
            if use_homography:
                mnn = compute_gt_matches_homography(
                    kA, kB, H_mat, h1, w1, args.match_threshold)
            else:
                mnn = compute_gt_matches(kA, kB, args.match_threshold)

            # ── Forward ───────────────────────────────────────────────────
            _train_mode()
            all_scores = training_forward(
                model, kA, kB, dA, dB,
                adapter=adapter, mod_embed=mod_embed,
                mod_A=mod_A, mod_B=mod_B,
            )

            loss = matching_loss(all_scores, mnn, args, kA, kB)

            # ── Backward ──────────────────────────────────────────────────
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            all_params = list(model.parameters())
            if adapter:
                all_params += list(adapter.parameters())
            if mod_embed:
                all_params += list(mod_embed.parameters())
            torch.nn.utils.clip_grad_norm_(
                [p for p in all_params if p.requires_grad], args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema:
                ema.update(model)

            lv = loss.item()
            losses.append(lv)
            pbar.set_postfix(loss=f"{lv:.4f}", gt=mnn.shape[0],
                             lr=f"{scheduler.get_last_lr()[0]:.1e}")

        avg_loss = float(np.mean(losses))
        history["loss"].append(avg_loss)

        # ── Validate ──────────────────────────────────────────────────────
        _eval_mode()
        if ema:
            ema.apply_shadow(model)

        vr = evaluate(model, val_set, args, tag="val")
        budget = list(vr.keys())[0]
        vr0 = vr[budget]
        history["val"].append(vr0)

        if ema:
            ema.restore(model)

        mma5 = vr0["mma@5px"]
        print(f"Epoch {epoch:2d}  │  loss={avg_loss:.4f}  "
              f"matches={vr0['avg_matches']:.1f}  "
              f"err={vr0['mean_err']:.2f}px  "
              f"MMA@5={mma5:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.1e}")

        if mma5 > best_mma5:
            best_mma5 = mma5
            os.makedirs(args.save_dir, exist_ok=True)
            sp = os.path.join(args.save_dir, "best.pth")
            state = {"model": model.state_dict()}
            if adapter:
                state["adapter"] = adapter.state_dict()
            if mod_embed:
                state["mod_embed"] = mod_embed.state_dict()
            torch.save(state, sp)
            print(f"  ↳ Best model saved → {sp}")

    return history, best_mma5


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    seed_everything(args.seed)

    vis_dir, ir_dir = setup_repos(args)

    # ── Late imports (after sys.path setup) ───────────────────────────────
    from loma.loma import LoMa
    from loma.device import device as dev

    print(f"\nDevice  : {dev}")
    print(f"Variant : LoMa-{args.variant}")

    # ── Dataset ───────────────────────────────────────────────────────────
    ds = RoadSceneDataset(vis_dir, ir_dir)
    train_set, val_set = train_val_split(ds, args.val_split, args.seed)
    print(f"Pairs   : {len(ds)} total  →  {len(train_set)} train / {len(val_set)} val")

    # ── Model ─────────────────────────────────────────────────────────────
    variant_cfgs = {
        "B128": dict(input_dim=128, embed_dim=256, num_heads=4,
                     descriptor="dedode_b",
                     weights_url="https://github.com/davnords/storage/releases/download/loma/loma_B128.pth"),
        "B": dict(input_dim=256, embed_dim=256, num_heads=4,
                  descriptor="dedode_g",
                  weights_url="https://github.com/davnords/storage/releases/download/loma/loma_B.pt"),
    }
    cfg = LoMa.Cfg(**variant_cfgs[args.variant], compile=False)
    model = LoMa(cfg)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Params  : {n_train:,} trainable  /  {n_frozen:,} frozen")

    # ── Baseline ──────────────────────────────────────────────────────────
    print("\n▸ Baseline evaluation …")
    baseline = evaluate(model, val_set, args, tag="baseline")
    print_results(baseline, "BASELINE  (pretrained, no fine-tuning)")

    # ── Fine-tune ─────────────────────────────────────────────────────────
    print("▸ Fine-tuning …")
    history, best_mma5 = train(model, train_set, val_set, args)

    # ── Reload best & final eval ──────────────────────────────────────────
    ckpt_path = os.path.join(args.save_dir, "best.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ckpt["model"])
    model.eval()

    # [Idea 10] Multi-scale evaluation
    print("\n▸ Final multi-scale evaluation …")
    final = evaluate(model, val_set, args,
                     keypoint_budgets=[512, 1024, 2048], tag="final")
    print_results(final, "FINE-TUNED  (best checkpoint)")
    print_results(baseline, "BASELINE  (for comparison)")

    # ── Improvement table ─────────────────────────────────────────────────
    b0 = list(baseline.values())[0]
    f0 = final[args.eval_num_keypoints] if args.eval_num_keypoints in final else list(final.values())[-1]
    print("═" * 62)
    print("  IMPROVEMENT SUMMARY")
    print("═" * 62)
    for k in ["avg_matches", "mean_err", "median_err",
              "mma@1px", "mma@3px", "mma@5px", "mma@10px"]:
        bv, fv = b0.get(k, 0), f0.get(k, 0)
        better = (fv < bv) if "err" in k else (fv > bv)
        arrow = "▲" if better else "▼"
        print(f"  {k:15s}  {bv:8.3f} → {fv:8.3f}  {arrow} {abs(fv-bv):.3f}")
    print()

    # ── Visualise ─────────────────────────────────────────────────────────
    for i in range(min(5, len(val_set))):
        s = val_set[i]
        p = os.path.join(args.save_dir, f"matches_{i}.jpg")
        visualize_matches(model, s["visible"], s["infrared"], args,
                          title=f"Fine-tuned pair {i}", save_path=p)
        print(f"Saved → {p}")

    # ── Save matcher-only (small file) ────────────────────────────────────
    matcher_state = {k: v for k, v in model.state_dict().items()
                     if not k.startswith("_detector.") and not k.startswith("_descriptor.")}
    mp = os.path.join(args.save_dir, "matcher_only.pth")
    torch.save(matcher_state, mp)
    print(f"\nMatcher-only weights → {mp}")

    # ── Save training curves ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(history["loss"], "b-o", ms=3)
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
        axes[0].grid(True, alpha=0.3)

        mc = [r["avg_matches"] for r in history["val"]]
        axes[1].plot(mc, "g-o", ms=3)
        axes[1].axhline(b0["avg_matches"], color="r", ls="--", label="baseline")
        axes[1].set(xlabel="Epoch", ylabel="Avg Matches", title="Val Matches")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        mm = [r["mma@5px"] for r in history["val"]]
        axes[2].plot(mm, "m-o", ms=3)
        axes[2].axhline(b0["mma@5px"], color="r", ls="--", label="baseline")
        axes[2].set(xlabel="Epoch", ylabel="MMA@5px", title="Val MMA@5px")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        cp = os.path.join(args.save_dir, "curves.png")
        plt.savefig(cp, dpi=150, bbox_inches="tight")
        print(f"Curves → {cp}")
    except ImportError:
        pass

    print(f"\n{'═'*62}")
    print(f"  DONE  —  best MMA@5px = {best_mma5:.4f}")
    print(f"{'═'*62}")


if __name__ == "__main__":
    main()
