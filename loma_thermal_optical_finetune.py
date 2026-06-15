#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoMa Fine-tuning for Thermal-Optical Image Matching
Single executable Kaggle script.

Run directly as: python loma_thermal_optical_finetune.py
Assumes repository is forked to mahdi13830510/LoMa and will be cloned fresh.
"""

# ===========================================================================
# SECTION 0 — IMPORTS & INSTALL
# ===========================================================================

import os, sys, subprocess, shutil, logging, json, random, math, time, gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("loma_thermal")

IS_KAGGLE = os.path.exists("/kaggle")
WORK_DIR  = Path("/kaggle/working") if IS_KAGGLE else Path("/tmp/loma_work")
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR   = WORK_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR  = WORK_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
VIZ_DIR   = OUT_DIR / "viz"
VIZ_DIR.mkdir(exist_ok=True)

def _run(cmd, **kw):
    log.info("$ %s", " ".join(cmd) if isinstance(cmd, list) else cmd)
    return subprocess.run(cmd, check=True, **kw)

# Install dependencies
_run([sys.executable, "-m", "pip", "install", "-q",
      "einops>=0.8.1", "tyro", "tqdm", "opencv-python-headless",
      "matplotlib", "scipy"])

# Clone LoMa from forked repo (change to upstream if not forking)
LOMA_REPO = "https://github.com/mahdi13830510/LoMa.git"
LOMA_DIR  = WORK_DIR / "LoMa"
if not LOMA_DIR.exists():
    _run(["git", "clone", "--depth", "1", LOMA_REPO, str(LOMA_DIR)])
else:
    log.info("LoMa already cloned at %s", LOMA_DIR)

# Install LoMa
_run([sys.executable, "-m", "pip", "install", "-q", "-e", str(LOMA_DIR)])

# ── runtime imports (after install) ─────────────────────────────────────────
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ===========================================================================
# SECTION 1 — PATCH LoMa FOR TRAINING
# ===========================================================================
# The released forward() only returns scores for the final layer.
# We monkey-patch to also return per-layer scores (all_scores) when in
# train() mode, which the GlueLoss-style supervision requires.

log.info("Patching LoMa.forward for training support …")

from loma.loma import LoMa, MatchAssignment, filter_matches, to_pixel_coords
from loma.device import device, amp_dtype

_orig_match_assignment_fwd = MatchAssignment.forward


def _ma_forward_train_aware(self, desc0, desc1):
    """Return log_double_softmax in train mode, product-softmax in eval mode."""
    mdesc0 = self.final_proj(desc0)
    mdesc1 = self.final_proj(desc1)
    _, _, d = mdesc0.shape
    mdesc0 = mdesc0 / d**0.25
    mdesc1 = mdesc1 / d**0.25
    sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
    if self.training:
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        from loma.loma import log_double_softmax
        scores = log_double_softmax(sim, z0, z1)
    else:
        scores = F.softmax(sim, dim=2) * F.softmax(sim, dim=1)
    return scores, sim


MatchAssignment.forward = _ma_forward_train_aware


def _loma_forward_patched(self, kpts0, kpts1, desc0, desc1):
    """
    Forward pass. In training mode, runs all layers and returns all_scores
    for layer-wise supervision. In eval mode, behaves as the original.
    """
    with torch.autocast(
        enabled=self.cfg.mp, dtype=amp_dtype, device_type=device.type
    ):
        kpts0 = kpts0.to(device)
        kpts1 = kpts1.to(device)
        desc0 = desc0.to(device).detach().contiguous()
        desc1 = desc1.to(device).detach().contiguous()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        enc0 = self.posenc(kpts0)
        enc1 = self.posenc(kpts1)

        all_scores: List[torch.Tensor] = []
        for i in range(self.cfg.n_layers):
            desc0, desc1 = self.transformers[i](desc0, desc1, enc0, enc1)
            if self.training:
                s, _ = self.log_assignment[i](desc0, desc1)
                all_scores.append(s)

        if not self.training:
            scores, _ = self.log_assignment[i](desc0, desc1)
            return {"scores": scores}

    return {"scores": all_scores[-1], "all_scores": all_scores}


LoMa.forward = _loma_forward_patched
log.info("LoMa patched ✓")

# ===========================================================================
# SECTION 2 — DATASET PREPARATION  (fully automated, no manual uploads)
# ===========================================================================
# Priority order:
#   1. LasHeR    — clone metadata (split lists) from GitHub; uses image data
#                  only if already present (requires BaiduNetdisk/TeraBox manually).
#   2. RoadScene — clone from https://github.com/hanna-xu/road-scene-infrared-
#                  visible-images (public repo, images committed directly, ~63 MB).
# ===========================================================================

DATA_DIR = WORK_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_clone(url: str, target: Path, depth: int = 1) -> bool:
    """Clone url → target. Returns True on success."""
    if target.exists():
        log.info("Already cloned: %s", target)
        return True
    try:
        _run(["git", "clone", f"--depth={depth}", url, str(target)])
        return True
    except Exception as e:
        log.warning("git clone %s failed: %s", url, e)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        return False



def _count_images(directory: Path, exts=("*.jpg", "*.png", "*.jpeg")) -> int:
    return sum(len(list(directory.rglob(e))) for e in exts)


# ---------------------------------------------------------------------------
# LasHeR acquisition
# ---------------------------------------------------------------------------
# The full LasHeR dataset (730K frames) requires BaiduNetdisk/TeraBox
# authentication, which cannot be automated.  We therefore:
#   (a) Clone the GitHub metadata repo for the protocol-2 split lists.
#   (b) Attempt to fetch a small curated subset via direct HTTP if any
#       publicly mirrored zip is reachable.
#   (c) Fall through to RoadScene if no images are obtained.

LASHER_META_URL = "https://github.com/BUGPLEASEOUT/LasHeR.git"
LASHER_DIR      = DATA_DIR / "LasHeR"
LASHER_META     = DATA_DIR / "LasHeR_meta"


def _acquire_lasher() -> Optional[Path]:
    """
    LasHeR images require BaiduNetdisk/TeraBox credentials and cannot be
    downloaded automatically.  We clone only the metadata repo (split lists)
    so that build_lasher_pairs() can use the official protocol-2 splits if
    the user has pre-populated LASHER_DIR with the image sequences.
    Returns LASHER_DIR if image data is already present, else None.
    """
    _git_clone(LASHER_META_URL, LASHER_META)

    if LASHER_DIR.exists() and _count_images(LASHER_DIR) > 50:
        log.info("LasHeR images found at %s", LASHER_DIR)
        return LASHER_DIR

    log.info(
        "LasHeR images not present (BaiduNetdisk/TeraBox required). "
        "Skipping LasHeR; will use RoadScene."
    )
    return None


# ---------------------------------------------------------------------------
# RoadScene acquisition  (public GitHub repo, images committed directly)
# ---------------------------------------------------------------------------
# Primary source: https://github.com/hanna-xu/road-scene-infrared-visible-images
#   — 221 aligned visible/thermal pairs, ~18 MB total, no LFS needed.
# Secondary:      https://github.com/jiayi-ma/RoadScene  (same data, SSH origin
#   of the local copy; HTTPS clone should work identically)

ROADSCENE_URLS = [
    "https://github.com/hanna-xu/road-scene-infrared-visible-images.git",
    "https://github.com/jiayi-ma/RoadScene.git",
]
ROADSCENE_DIR = DATA_DIR / "RoadScene"


def _acquire_roadscene() -> Optional[Path]:
    """Clone RoadScene from GitHub. Returns the root path on success."""
    if ROADSCENE_DIR.exists() and _count_images(ROADSCENE_DIR) > 50:
        log.info("RoadScene already present at %s", ROADSCENE_DIR)
        return ROADSCENE_DIR

    for url in ROADSCENE_URLS:
        log.info("Cloning RoadScene from %s …", url)
        if _git_clone(url, ROADSCENE_DIR):
            n = _count_images(ROADSCENE_DIR)
            if n > 50:
                log.info("RoadScene cloned: %d images", n)
                return ROADSCENE_DIR
            log.warning("Clone succeeded but found only %d images — trying next source", n)
            shutil.rmtree(ROADSCENE_DIR, ignore_errors=True)

    return None




# ---------------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------------

@dataclass
class ImagePair:
    visible: Path
    thermal: Path
    split: str  # train / val / test


def _split_pairs(pairs: List[Tuple[Path, Path]]) -> List[ImagePair]:
    random.seed(42)
    random.shuffle(pairs)
    n       = len(pairs)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    result  = []
    for i, (v, t) in enumerate(pairs):
        split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        result.append(ImagePair(v, t, split))
    return result


def build_roadscene_pairs(root: Path) -> List[ImagePair]:
    vis_dir = root / "crop_HR_visible"
    thr_dir = root / "cropinfrared"
    if not vis_dir.exists():
        vis_dir = root / "crop_LR_visible"
    if not thr_dir.exists():
        thr_dir = root / "infrared"

    raw: List[Tuple[Path, Path]] = []
    for v in sorted(vis_dir.glob("*.jpg")) + sorted(vis_dir.glob("*.png")):
        for ext in (".jpg", ".png"):
            t = thr_dir / (v.stem + ext)
            if t.exists():
                raw.append((v, t))
                break

    result = _split_pairs(raw)
    log.info("RoadScene: %d train / %d val / %d test",
             sum(1 for p in result if p.split == "train"),
             sum(1 for p in result if p.split == "val"),
             sum(1 for p in result if p.split == "test"))
    return result


def build_lasher_pairs(root: Path, max_per_seq: int = 15) -> List[ImagePair]:
    # Locate protocol-2 split lists (may be in meta-only clone)
    for list_dir in (LASHER_META / "List_for_Protocol2",
                     root / "List_for_Protocol2",
                     root.parent / "LasHeR_meta" / "List_for_Protocol2"):
        if list_dir.exists():
            break
    else:
        list_dir = None

    train_seqs: Optional[set] = None
    test_seqs:  Optional[set] = None
    if list_dir and list_dir.exists():
        tf = list_dir / "trainingsetList.txt"
        ef = list_dir / "testingsetList.txt"
        if tf.exists():
            train_seqs = set(tf.read_text().splitlines())
        if ef.exists():
            test_seqs  = set(ef.read_text().splitlines())

    sequences = [d for d in root.iterdir()
                 if d.is_dir()
                 and (d / "infrared").exists()
                 and (d / "visible").exists()]
    random.seed(42)
    random.shuffle(sequences)

    raw: List[Tuple[Path, Path]] = []
    splits: List[str] = []
    for seq in sequences:
        name = seq.name
        if train_seqs is not None:
            sp = ("train" if name in train_seqs
                  else "test" if (test_seqs and name in test_seqs)
                  else "val")
        else:
            idx = sequences.index(seq)
            n   = len(sequences)
            sp  = "train" if idx < int(0.70*n) else ("val" if idx < int(0.85*n) else "test")

        ir_frames  = sorted((seq / "infrared").glob("i*.jpg"))
        vis_frames = sorted((seq / "visible").glob("v*.jpg"))
        frame_pairs = list(zip(vis_frames, ir_frames))
        step = max(1, len(frame_pairs) // max_per_seq)
        for v, t in frame_pairs[::step][:max_per_seq]:
            raw.append((v, t))
            splits.append(sp)

    result = [ImagePair(v, t, sp) for (v, t), sp in zip(raw, splits)]
    log.info("LasHeR: %d train / %d val / %d test",
             sum(1 for p in result if p.split == "train"),
             sum(1 for p in result if p.split == "val"),
             sum(1 for p in result if p.split == "test"))
    return result


# ---------------------------------------------------------------------------
# Orchestrate acquisition
# ---------------------------------------------------------------------------

ALL_PAIRS:    List[ImagePair] = []
DATASET_NAME: str = "unknown"

# 1 — LasHeR (skips automatically if images not pre-populated)
lasher_root = _acquire_lasher()
if lasher_root is not None:
    ALL_PAIRS    = build_lasher_pairs(lasher_root, max_per_seq=15)
    DATASET_NAME = "LasHeR"

# 2 — RoadScene (cloned automatically from GitHub)
if len(ALL_PAIRS) < 30:
    roadscene_root = _acquire_roadscene()
    if roadscene_root is not None:
        rsp = build_roadscene_pairs(roadscene_root)
        if not ALL_PAIRS:
            ALL_PAIRS    = rsp
            DATASET_NAME = "RoadScene"
        else:
            ALL_PAIRS.extend(rsp)
            DATASET_NAME = "LasHeR+RoadScene"

if len(ALL_PAIRS) == 0:
    raise RuntimeError(
        "No thermal-optical pairs found.\n"
        "  RoadScene: git clone of https://github.com/hanna-xu/road-scene-infrared-visible-images failed.\n"
        "  LasHeR:    images require manual download from BaiduNetdisk/TeraBox "
        "(place sequences under /kaggle/working/data/LasHeR/)."
    )

log.info("Using dataset: %s  total pairs: %d", DATASET_NAME, len(ALL_PAIRS))
TRAIN_PAIRS = [p for p in ALL_PAIRS if p.split == "train"]
VAL_PAIRS   = [p for p in ALL_PAIRS if p.split == "val"]
TEST_PAIRS  = [p for p in ALL_PAIRS if p.split == "test"]

# ===========================================================================
# SECTION 3 — DATASET CLASS
# ===========================================================================

TRAIN_SIZE = 512      # resize images to this for training
NUM_CORRESP = 200     # grid correspondences per pair
MAX_DISTORT = 0.12    # max relative homography distortion


def load_rgb(path: Path, size: int = TRAIN_SIZE) -> np.ndarray:
    """Load any image as float32 RGB [0,1], resized to (size × size)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_gray_as_rgb(path: Path, size: int = TRAIN_SIZE) -> np.ndarray:
    """Load grayscale thermal as 3-channel float32 RGB [0,1]."""
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.stack([arr, arr, arr], axis=-1)


def random_homography(H: int, W: int, max_distort: float = MAX_DISTORT,
                      rng: Optional[np.random.Generator] = None
                      ) -> np.ndarray:
    """Random 4-corner perspective homography in pixel space."""
    if rng is None:
        rng = np.random.default_rng()
    pts_src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    delta = (rng.random((4, 2)).astype(np.float32) - 0.5) * 2 * max_distort
    pts_dst = pts_src + delta * np.array([[W, H]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return M.astype(np.float64)


def warp_image(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    return cv2.warpPerspective(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def correspondences_from_homography(
    M: np.ndarray,
    H: int, W: int,
    n_grid: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (pts_A, pts_B) in NORMALIZED [-1,1] coordinates.
    pts_A: grid points in image A.
    pts_B: same points projected through M (A→B) in image B.
    """
    margin = 0.1
    xs = np.linspace(margin * W, (1 - margin) * W, n_grid, dtype=np.float32)
    ys = np.linspace(margin * H, (1 - margin) * H, n_grid, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    pts_A_px = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N,2)

    ones = np.ones((len(pts_A_px), 1), dtype=np.float32)
    pts_A_h = np.concatenate([pts_A_px, ones], axis=1)  # (N,3)
    pts_B_h = (M.astype(np.float32) @ pts_A_h.T).T       # (N,3)
    pts_B_px = pts_B_h[:, :2] / (pts_B_h[:, 2:3] + 1e-8)

    # Keep only points whose projected location is inside image B
    valid = (
        (pts_B_px[:, 0] >= 0) & (pts_B_px[:, 0] < W) &
        (pts_B_px[:, 1] >= 0) & (pts_B_px[:, 1] < H)
    )
    pts_A_px = pts_A_px[valid]
    pts_B_px = pts_B_px[valid]

    # Normalize to [-1, 1]
    pts_A_norm = np.stack([2 * pts_A_px[:, 0] / W - 1,
                           2 * pts_A_px[:, 1] / H - 1], axis=1)
    pts_B_norm = np.stack([2 * pts_B_px[:, 0] / W - 1,
                           2 * pts_B_px[:, 1] / H - 1], axis=1)
    return pts_A_norm.astype(np.float32), pts_B_norm.astype(np.float32)


class ThermalOpticalDataset(Dataset):
    """
    Each sample is an aligned (visible, thermal) pair.
    We apply a random homography to the thermal image to create a known
    geometric transformation and derive ground-truth correspondences from it.
    This enables learning cross-modal matching without depth/pose supervision.
    """

    def __init__(
        self,
        pairs: List[ImagePair],
        img_size: int = TRAIN_SIZE,
        augment_color: bool = True,
        rng_seed: int = 0,
    ):
        self.pairs = pairs
        self.img_size = img_size
        self.augment_color = augment_color
        self._rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return len(self.pairs)

    def _color_jitter(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Mild brightness/contrast augmentation for visible images."""
        alpha = float(rng.uniform(0.8, 1.2))   # contrast
        beta  = float(rng.uniform(-0.1, 0.1))   # brightness
        return np.clip(alpha * img + beta, 0.0, 1.0).astype(np.float32)

    def _thermal_augment(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Gain + offset noise for thermal images."""
        gain   = float(rng.uniform(0.85, 1.15))
        offset = float(rng.uniform(-0.05, 0.05))
        return np.clip(gain * img + offset, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        rng  = np.random.default_rng(int(self._rng.integers(0, 2**31)) + idx)
        S    = self.img_size

        vis = load_rgb(pair.visible, S)
        thr = load_gray_as_rgb(pair.thermal, S)

        # Random homography applied to the thermal image
        M = random_homography(S, S, rng=rng)
        thr_warped = warp_image(thr, M)

        # GT correspondences in normalized coords
        pts_A, pts_B = correspondences_from_homography(M, S, S, n_grid=14)
        # pts_A[i] in visible maps to pts_B[i] in warped thermal

        if self.augment_color:
            vis        = self._color_jitter(vis, rng)
            thr_warped = self._thermal_augment(thr_warped, rng)

        # HWC→CHW tensors
        vis_t = torch.from_numpy(vis.transpose(2, 0, 1))           # (3,S,S)
        thr_t = torch.from_numpy(thr_warped.transpose(2, 0, 1))    # (3,S,S)
        pts_A_t = torch.from_numpy(pts_A)                           # (N,2)
        pts_B_t = torch.from_numpy(pts_B)                           # (N,2)
        M_t = torch.from_numpy(M.astype(np.float32))               # (3,3)

        return {
            "vis":   vis_t,
            "thr":   thr_t,
            "pts_A": pts_A_t,
            "pts_B": pts_B_t,
            "H":     M_t,
            "img_h": S,
            "img_w": S,
        }


def collate_fn(batch):
    """Collate samples, padding GT correspondences to the same length."""
    max_n = max(b["pts_A"].shape[0] for b in batch)
    result = {
        "vis":   torch.stack([b["vis"]   for b in batch]),
        "thr":   torch.stack([b["thr"]   for b in batch]),
        "H":     torch.stack([b["H"]     for b in batch]),
        "img_h": batch[0]["img_h"],
        "img_w": batch[0]["img_w"],
    }
    pts_A_list, pts_B_list, mask_list = [], [], []
    for b in batch:
        n = b["pts_A"].shape[0]
        pad = max_n - n
        pa = F.pad(b["pts_A"], (0, 0, 0, pad))
        pb = F.pad(b["pts_B"], (0, 0, 0, pad))
        mask = torch.zeros(max_n, dtype=torch.bool)
        mask[:n] = True
        pts_A_list.append(pa)
        pts_B_list.append(pb)
        mask_list.append(mask)
    result["pts_A"] = torch.stack(pts_A_list)       # (B, N, 2)
    result["pts_B"] = torch.stack(pts_B_list)       # (B, N, 2)
    result["corresp_mask"] = torch.stack(mask_list) # (B, N)
    return result

# ===========================================================================
# SECTION 4 — TRAINING LOSS
# ===========================================================================

@dataclass
class LossCfg:
    num_keypoints: int = 1024
    match_threshold: float = 5.0   # pixel distance for a GT match
    layer_weight: float = 1.0
    matchability_weight: float = 0.5


def compute_gt_mnn_from_homography(
    kpts_A: torch.Tensor,   # (B, N, 2) normalized
    kpts_B: torch.Tensor,   # (B, M, 2) normalized
    H_px: torch.Tensor,     # (B, 3, 3) pixel homography A→B
    img_h: int,
    img_w: int,
    threshold_px: float,
) -> torch.Tensor:
    """
    Returns MNN tensor of shape (K, 3): (batch_idx, idx_A, idx_B).
    Uses mutual nearest-neighbor constraint in projected pixel space.
    """
    B, N, _ = kpts_A.shape
    _, M, _ = kpts_B.shape

    # Pixel coords
    def to_px(norm, h, w):
        x = (norm[..., 0] + 1) / 2 * w
        y = (norm[..., 1] + 1) / 2 * h
        return torch.stack([x, y], dim=-1)

    kpts_A_px = to_px(kpts_A, img_h, img_w)  # (B, N, 2)
    kpts_B_px = to_px(kpts_B, img_h, img_w)  # (B, M, 2)

    # Project kpts_A through H_px to B's pixel space
    ones = torch.ones(B, N, 1, device=kpts_A.device, dtype=kpts_A.dtype)
    kpts_A_h = torch.cat([kpts_A_px, ones], dim=-1)  # (B, N, 3)
    kpts_A_in_B_h = torch.bmm(kpts_A_h, H_px.transpose(1, 2))  # (B, N, 3)
    denom = kpts_A_in_B_h[..., 2:3].clamp(min=1e-6)
    kpts_A_in_B = kpts_A_in_B_h[..., :2] / denom  # (B, N, 2)

    # Distance matrix: kpts_A projected → kpts_B
    dist = torch.cdist(kpts_A_in_B, kpts_B_px)  # (B, N, M)

    # Also project kpts_B back through H_px^{-1} for mutual consistency
    ones_m = torch.ones(B, M, 1, device=kpts_B.device, dtype=kpts_B.dtype)
    kpts_B_h = torch.cat([kpts_B_px, ones_m], dim=-1)
    H_inv = torch.linalg.inv(H_px.float()).to(kpts_B.dtype)
    kpts_B_in_A_h = torch.bmm(kpts_B_h, H_inv.transpose(1, 2))
    denom2 = kpts_B_in_A_h[..., 2:3].clamp(min=1e-6)
    kpts_B_in_A = kpts_B_in_A_h[..., :2] / denom2  # (B, M, 2)

    dist_ba = torch.cdist(kpts_A_px, kpts_B_in_A)  # (B, N, M)

    # Mutual nearest neighbors within threshold
    nn_A2B = dist.argmin(dim=2)    # (B, N)
    nn_B2A = dist_ba.argmin(dim=1) # (B, M)

    all_mnn = []
    for b in range(B):
        for i in range(N):
            j = int(nn_A2B[b, i])
            if nn_B2A[b, j] == i:
                d = float(dist[b, i, j])
                if d < threshold_px:
                    all_mnn.append((b, i, j))
    if not all_mnn:
        return torch.zeros((0, 3), dtype=torch.long, device=kpts_A.device)
    return torch.tensor(all_mnn, dtype=torch.long, device=kpts_A.device)


def thermal_optical_loss(
    batch: Dict,
    model: LoMa,
    cfg: LossCfg,
) -> Tuple[torch.Tensor, Dict]:
    """
    1. Detect keypoints (frozen DaD) on visible and thermal images.
    2. Describe keypoints (frozen DeDoDeDescriptor).
    3. Compute GT matches from known homography.
    4. Run trainable transformer matcher → all_scores.
    5. Compute GlueLoss-style NLL + matchability BCE.
    """
    vis  = batch["vis"].to(device)    # (B,3,H,W)
    thr  = batch["thr"].to(device)    # (B,3,H,W)
    H_px = batch["H"].to(device)      # (B,3,3) pixel homography
    img_h = int(batch["img_h"])
    img_w = int(batch["img_w"])
    B = vis.shape[0]

    # ── Detect (frozen, no grad) ───────────────────────────────────────────
    with torch.inference_mode():
        kpts_A = model._detector.detect(
            {"image": vis}, num_keypoints=cfg.num_keypoints
        )["keypoints"]   # (B, N, 2)
        kpts_B = model._detector.detect(
            {"image": thr}, num_keypoints=cfg.num_keypoints
        )["keypoints"]   # (B, M, 2)

        images_cat = torch.cat([vis, thr], dim=0)   # (2B, 3, H, W)
        kpts_cat   = torch.cat([kpts_A, kpts_B], dim=0)  # (2B, N, 2)
        descs = model._descriptor.describe_keypoints(
            images_cat, kpts_cat
        )["descriptions"]   # (2B, N, D)
        desc_A, desc_B = descs[:B], descs[B:]

    # ── GT MNN from homography ─────────────────────────────────────────────
    mnn = compute_gt_mnn_from_homography(
        kpts_A, kpts_B, H_px, img_h, img_w,
        threshold_px=cfg.match_threshold,
    )

    # ── Forward (trainable) ───────────────────────────────────────────────
    result = model(kpts_A, kpts_B, desc_A, desc_B)
    all_scores = result["all_scores"]  # list of (B, M+1, N+1)

    if mnn.numel() == 0:
        dummy = sum(0.0 * s.mean() for s in all_scores)
        return dummy, {"loss": float(dummy), "n_gt_matches": 0}

    # ── GlueLoss ──────────────────────────────────────────────────────────
    total_loss = 0.0
    n_layers = len(all_scores)
    for scores in all_scores:
        S_M = scores.shape[1] - 1  # number of keypoints in A
        S_N = scores.shape[2] - 1  # number of keypoints in B

        matchable_A = torch.zeros(B, S_M, device=device)
        matchable_B = torch.zeros(B, S_N, device=device)

        # Clamp indices to valid range (safety guard)
        bi  = mnn[:, 0].clamp(0, B - 1)
        idx_a = mnn[:, 1].clamp(0, S_M - 1)
        idx_b = mnn[:, 2].clamp(0, S_N - 1)

        matchable_A[bi, idx_a] = 1.0
        matchable_B[bi, idx_b] = 1.0

        # NLL on GT matched pairs
        loss_cond = -scores[bi, idx_a, idx_b].mean()

        # Matchability BCE (dustbin column/row)
        loss_ma = F.binary_cross_entropy_with_logits(
            scores[:, :S_M, -1], matchable_A
        )
        loss_mb = F.binary_cross_entropy_with_logits(
            scores[:, -1, :S_N], matchable_B
        )

        total_loss = (total_loss
                      + cfg.layer_weight * loss_cond
                      + cfg.matchability_weight * (loss_ma + loss_mb))

    total_loss = total_loss / n_layers
    return total_loss, {
        "loss": float(total_loss),
        "n_gt_matches": int(mnn.shape[0]),
    }

# ===========================================================================
# SECTION 5 — EVALUATION UTILITIES
# ===========================================================================

@torch.inference_mode()
def eval_pair_metrics(
    model: LoMa,
    vis_path: Path,
    thr_path: Path,
    img_size: int = TRAIN_SIZE,
    num_keypoints: int = 1024,
    filter_threshold: float = 0.1,
) -> Dict:
    """
    Evaluate matching on a single aligned pair (identity transform expected).
    Returns metrics without requiring GT pose.
    """
    model.eval()

    vis = load_rgb(vis_path, img_size)       # (H,W,3) float32
    thr = load_gray_as_rgb(thr_path, img_size)

    vis_t = torch.from_numpy(vis.T.swapaxes(0, 1).transpose(2, 0, 1)).unsqueeze(0).to(device)
    thr_t = torch.from_numpy(thr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    kpts_A = model._detector.detect({"image": vis_t}, num_keypoints=num_keypoints)["keypoints"]
    kpts_B = model._detector.detect({"image": thr_t}, num_keypoints=num_keypoints)["keypoints"]
    imgs   = torch.cat([vis_t, thr_t], dim=0)
    kpts_c = torch.cat([kpts_A, kpts_B], dim=0)
    descs  = model._descriptor.describe_keypoints(imgs, kpts_c)["descriptions"]
    desc_A, desc_B = descs[:1], descs[1:]

    scores = model(kpts_A, kpts_B, desc_A, desc_B)["scores"]
    m0, _, msc0, _ = filter_matches(scores, filter_threshold)

    valid = m0[0] > -1
    n_matches = int(valid.sum())
    n_kpts_A  = kpts_A.shape[1]
    n_kpts_B  = kpts_B.shape[1]
    match_ratio = n_matches / max(min(n_kpts_A, n_kpts_B), 1)

    # For aligned pairs: compute reprojection error assuming identity warp
    if n_matches > 0:
        mA = kpts_A[0][valid]
        mB = kpts_B[0][m0[0][valid]]
        err_norm = (mA - mB).norm(dim=-1)  # normalized distance
        err_px   = err_norm * img_size / 2  # pixel distance
        mean_err_px = float(err_px.mean())
        med_err_px  = float(err_px.median())
        inlier_ratio_1px = float((err_px < (1.0 / img_size * 2)).float().mean())
        inlier_ratio_5px = float((err_px < (5.0 / img_size * 2)).float().mean())
        conf_scores = msc0[0][valid]
        mean_conf = float(conf_scores.mean())
    else:
        mean_err_px = float("nan")
        med_err_px  = float("nan")
        inlier_ratio_1px = 0.0
        inlier_ratio_5px = 0.0
        mean_conf = 0.0

    return {
        "n_matches":    n_matches,
        "n_kpts_A":     n_kpts_A,
        "n_kpts_B":     n_kpts_B,
        "match_ratio":  match_ratio,
        "mean_err_px":  mean_err_px,
        "med_err_px":   med_err_px,
        "inlier_1px":   inlier_ratio_1px,
        "inlier_5px":   inlier_ratio_5px,
        "mean_conf":    mean_conf,
    }


def run_evaluation(
    model: LoMa,
    pairs: List[ImagePair],
    tag: str,
    max_pairs: int = 50,
    img_size: int = TRAIN_SIZE,
) -> Dict:
    """Evaluate model on a list of pairs, return averaged metrics."""
    model.eval()
    eval_pairs = pairs[:max_pairs]
    all_m: List[Dict] = []
    for p in tqdm(eval_pairs, desc=f"Eval [{tag}]"):
        try:
            m = eval_pair_metrics(model, p.visible, p.thermal, img_size)
            all_m.append(m)
        except Exception as e:
            log.warning("Eval error on %s: %s", p.visible.name, e)

    if not all_m:
        return {}

    def _nanmean(key):
        vals = [m[key] for m in all_m if not math.isnan(m.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    result = {
        k: _nanmean(k) for k in all_m[0]
    }
    result["n_pairs"] = len(all_m)
    log.info("[%s] matches=%.1f  match_ratio=%.3f  inlier_5px=%.3f  conf=%.3f",
             tag,
             result.get("n_matches", 0),
             result.get("match_ratio", 0),
             result.get("inlier_5px", 0),
             result.get("mean_conf", 0))
    return result

# ===========================================================================
# SECTION 6 — VISUALIZATIONS
# ===========================================================================

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert float [0,1] to uint8."""
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def visualize_matches(
    vis: np.ndarray, thr: np.ndarray,
    kpts_A: np.ndarray, kpts_B: np.ndarray,
    matches_A: np.ndarray, matches_B: np.ndarray,
    correct_mask: Optional[np.ndarray] = None,
    title: str = "Matches",
    save_path: Optional[Path] = None,
    max_draw: int = 100,
):
    """Side-by-side match visualization with correct/incorrect coloring."""
    H, W = vis.shape[:2]
    canvas = np.zeros((H, 2 * W, 3), dtype=np.uint8)
    canvas[:, :W] = _to_uint8(vis)
    canvas[:, W:] = _to_uint8(thr)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=100)
    ax.imshow(canvas)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    n = min(len(matches_A), max_draw)
    idx = np.random.choice(len(matches_A), n, replace=False) if len(matches_A) > n else np.arange(n)
    for i in idx:
        x1, y1 = float(matches_A[i, 0]), float(matches_A[i, 1])
        x2, y2 = float(matches_B[i, 0]) + W, float(matches_B[i, 1])
        color = "lime" if (correct_mask is not None and correct_mask[i]) else "red"
        lw    = 0.8 if correct_mask is not None else 0.6
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=0.7)
        ax.plot(x1, y1, "o", color=color, ms=2)
        ax.plot(x2, y2, "o", color=color, ms=2)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def visualize_keypoints(
    vis: np.ndarray, thr: np.ndarray,
    kpts_A_px: np.ndarray, kpts_B_px: np.ndarray,
    title: str = "Keypoints",
    save_path: Optional[Path] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    axes[0].imshow(_to_uint8(vis))
    axes[0].scatter(kpts_A_px[:, 0], kpts_A_px[:, 1], s=4, c="cyan", alpha=0.7)
    axes[0].set_title(f"Visible — {len(kpts_A_px)} kpts")
    axes[0].axis("off")
    axes[1].imshow(_to_uint8(thr))
    axes[1].scatter(kpts_B_px[:, 0], kpts_B_px[:, 1], s=4, c="orange", alpha=0.7)
    axes[1].set_title(f"Thermal — {len(kpts_B_px)} kpts")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=12)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def visualize_confidence_distribution(
    conf_pre: np.ndarray, conf_post: np.ndarray,
    save_path: Optional[Path] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
    for ax, conf, label in zip(axes, [conf_pre, conf_post], ["Pretrained", "Fine-tuned"]):
        if len(conf) == 0:
            ax.set_title(f"{label} (no matches)")
            continue
        ax.hist(conf, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(float(np.median(conf)), color="red", lw=1.5, label=f"median={np.median(conf):.3f}")
        ax.set_xlabel("Match confidence")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    train_losses: List[float],
    val_metrics:  List[Dict],
    save_path: Optional[Path] = None,
):
    steps_per_epoch = max(1, len(train_losses) // max(len(val_metrics), 1))
    epochs = list(range(1, len(val_metrics) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=100)

    # Training loss
    ax = axes[0, 0]
    ax.plot(train_losses, lw=1, alpha=0.8, color="steelblue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # Match ratio
    ax = axes[0, 1]
    mr = [m.get("match_ratio", 0) for m in val_metrics]
    ax.plot(epochs, mr, "o-", color="forestgreen")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Match Ratio")
    ax.set_title("Validation Match Ratio")
    ax.grid(True, alpha=0.3)

    # Inlier 5px
    ax = axes[1, 0]
    ir = [m.get("inlier_5px", 0) for m in val_metrics]
    ax.plot(epochs, ir, "o-", color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Inlier Ratio @5px")
    ax.set_title("Validation Inlier Ratio")
    ax.grid(True, alpha=0.3)

    # Mean confidence
    ax = axes[1, 1]
    mc = [m.get("mean_conf", 0) for m in val_metrics]
    ax.plot(epochs, mc, "o-", color="purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Confidence")
    ax.set_title("Validation Mean Confidence")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fine-tuning Progress", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_recall_precision(
    metrics_pre: Dict,
    metrics_post: Dict,
    save_path: Optional[Path] = None,
):
    thresholds = [1, 2, 3, 5, 10, 15, 20]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    for metrics, label, color in [
        (metrics_pre,  "Pretrained",  "steelblue"),
        (metrics_post, "Fine-tuned",  "darkorange"),
    ]:
        # Approximate recall at different error thresholds using inlier metrics
        # (two points from our eval: 1px and 5px)
        pts = [(1, metrics.get("inlier_1px", 0)),
               (5, metrics.get("inlier_5px", 0))]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=color, lw=2, label=label)
    ax.set_xlabel("Error threshold (px)")
    ax.set_ylabel("Inlier Ratio")
    ax.set_title("Inlier Recall vs Error Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_table(
    metrics_pre: Dict,
    metrics_post: Dict,
    save_path: Optional[Path] = None,
):
    keys = ["n_matches", "match_ratio", "inlier_1px", "inlier_5px",
            "mean_err_px", "med_err_px", "mean_conf"]
    labels = ["Num matches", "Match ratio", "Inlier @1px", "Inlier @5px",
              "Mean err (px)", "Median err (px)", "Mean confidence"]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
    ax.axis("off")
    table_data = [[lbl,
                   f"{metrics_pre.get(k, float('nan')):.4f}",
                   f"{metrics_post.get(k, float('nan')):.4f}",
                   f"{(metrics_post.get(k, 0) - metrics_pre.get(k, 0)):+.4f}"]
                  for k, lbl in zip(keys, labels)]
    col_labels = ["Metric", "Pretrained", "Fine-tuned", "Δ"]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)
    ax.set_title("Pre-trained vs Fine-tuned — Thermal-Optical Matching", fontsize=12, pad=20)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def generate_full_visualizations(
    model: LoMa,
    pairs: List[ImagePair],
    tag: str,
    n_examples: int = 8,
    img_size: int = TRAIN_SIZE,
    filter_threshold: float = 0.1,
):
    """Generate per-pair match visualizations including best/worst cases."""
    model.eval()
    sample = pairs[:n_examples]
    stats = []
    for idx, p in enumerate(sample):
        vis = load_rgb(p.visible, img_size)
        thr = load_gray_as_rgb(p.thermal, img_size)
        vis_t = torch.from_numpy(vis.transpose(2, 0, 1)).unsqueeze(0).to(device)
        thr_t = torch.from_numpy(thr.transpose(2, 0, 1)).unsqueeze(0).to(device)

        kpts_A = model._detector.detect({"image": vis_t}, num_keypoints=1024)["keypoints"]
        kpts_B = model._detector.detect({"image": thr_t}, num_keypoints=1024)["keypoints"]
        imgs   = torch.cat([vis_t, thr_t], dim=0)
        kpts_c = torch.cat([kpts_A, kpts_B], dim=0)
        descs  = model._descriptor.describe_keypoints(imgs, kpts_c)["descriptions"]
        desc_A, desc_B = descs[:1], descs[1:]

        scores = model(kpts_A, kpts_B, desc_A, desc_B)["scores"]
        m0, _, msc0, _ = filter_matches(scores, filter_threshold)
        valid = m0[0] > -1

        def norm_to_px(kpts_norm, h, w):
            px = to_pixel_coords(kpts_norm, h, w)
            return px.cpu().numpy()

        kA_px = norm_to_px(kpts_A[0], img_size, img_size)
        kB_px = norm_to_px(kpts_B[0], img_size, img_size)

        if valid.any():
            mA_px = norm_to_px(kpts_A[0][valid], img_size, img_size)
            mB_px = norm_to_px(kpts_B[0][m0[0][valid]], img_size, img_size)
            err_px = np.linalg.norm(mA_px - mB_px, axis=1)
            correct_mask = err_px < 10.0  # aligned pairs: low err = correct
        else:
            mA_px = mB_px = np.zeros((0, 2))
            correct_mask = np.zeros(0, dtype=bool)
            err_px = np.zeros(0)

        n_m = int(valid.sum())
        stats.append({"idx": idx, "n_matches": n_m, "err_px": float(np.mean(err_px)) if len(err_px) else 999})

        # Keypoints visualization
        visualize_keypoints(
            vis, thr, kA_px, kB_px,
            title=f"{tag} — {p.visible.name}",
            save_path=VIZ_DIR / f"{tag}_{idx:02d}_kpts.png",
        )

        # Match visualization
        if n_m > 0:
            visualize_matches(
                vis, thr, kA_px, kB_px, mA_px, mB_px,
                correct_mask=correct_mask,
                title=f"{tag} — {n_m} matches — {p.visible.name}",
                save_path=VIZ_DIR / f"{tag}_{idx:02d}_matches.png",
            )

    # Best/worst by n_matches
    if stats:
        best  = min(stats, key=lambda x: x["err_px"])
        worst = max(stats, key=lambda x: x["err_px"])
        log.info("[%s] Best pair idx=%d (err=%.1f px)  Worst idx=%d (err=%.1f px)",
                 tag, best["idx"], best["err_px"], worst["idx"], worst["err_px"])

# ===========================================================================
# SECTION 7 — LOAD PRETRAINED MODEL
# ===========================================================================

log.info("Loading pretrained LoMa (LoMaB128) …")
from loma.loma import LoMaB128

_cfg = LoMaB128()
pretrained_model = LoMa(_cfg)
pretrained_model.to(device)
pretrained_model.eval()
log.info("Pretrained LoMa loaded. device=%s  embed_dim=%d  n_layers=%d",
         device, _cfg.embed_dim, _cfg.n_layers)

# ===========================================================================
# SECTION 8 — PRE-TRAINING EVALUATION
# ===========================================================================

log.info("=" * 60)
log.info("PRE-TRAINING EVALUATION")
log.info("=" * 60)

metrics_pre = run_evaluation(
    pretrained_model,
    TEST_PAIRS if TEST_PAIRS else VAL_PAIRS,
    tag="pretrained",
    max_pairs=min(50, len(TEST_PAIRS or VAL_PAIRS)),
)

log.info("Pre-training metrics: %s", json.dumps(
    {k: round(v, 4) for k, v in metrics_pre.items() if isinstance(v, float)}, indent=2
))

# Save pre-training eval
with open(OUT_DIR / "metrics_pretrained.json", "w") as f:
    json.dump(metrics_pre, f, indent=2)

# Generate pre-training visualizations
log.info("Generating pre-training visualizations …")
generate_full_visualizations(
    pretrained_model,
    (TEST_PAIRS or VAL_PAIRS)[:8],
    tag="pretrained",
    n_examples=min(8, len(TEST_PAIRS or VAL_PAIRS)),
)

# Collect confidence scores from a batch for distribution plot
_conf_pre: List[float] = []
with torch.inference_mode():
    for p in (TEST_PAIRS or VAL_PAIRS)[:20]:
        try:
            vis_t = torch.from_numpy(
                load_rgb(p.visible, TRAIN_SIZE).transpose(2, 0, 1)
            ).unsqueeze(0).to(device)
            thr_t = torch.from_numpy(
                load_gray_as_rgb(p.thermal, TRAIN_SIZE).transpose(2, 0, 1)
            ).unsqueeze(0).to(device)
            kA = pretrained_model._detector.detect({"image": vis_t}, num_keypoints=512)["keypoints"]
            kB = pretrained_model._detector.detect({"image": thr_t}, num_keypoints=512)["keypoints"]
            imgs = torch.cat([vis_t, thr_t], dim=0)
            kc   = torch.cat([kA, kB], dim=0)
            descs = pretrained_model._descriptor.describe_keypoints(imgs, kc)["descriptions"]
            dA, dB = descs[:1], descs[1:]
            scores = pretrained_model(kA, kB, dA, dB)["scores"]
            m0, _, msc0, _ = filter_matches(scores, 0.1)
            valid = m0[0] > -1
            if valid.any():
                _conf_pre.extend(msc0[0][valid].cpu().tolist())
        except Exception:
            pass
_conf_pre_arr = np.array(_conf_pre)

# ===========================================================================
# SECTION 9 — FINE-TUNING
# ===========================================================================

log.info("=" * 60)
log.info("FINE-TUNING")
log.info("=" * 60)

# ── Hyperparameters ───────────────────────────────────────────────────────
@dataclass
class TrainCfg:
    num_epochs:          int   = 30
    batch_size:          int   = 4
    grad_accum_steps:    int   = 8          # effective batch = 32
    lr:                  float = 5e-5
    lr_min:              float = 1e-6
    warmup_steps:        int   = 50
    grad_clip:           float = 1.0
    num_keypoints:       int   = 1024
    match_threshold_px:  float = 8.0
    filter_threshold:    float = 0.1
    val_every:           int   = 5         # epochs
    log_every:           int   = 20        # steps
    amp:                 bool  = True
    num_workers:         int   = 2


train_cfg = TrainCfg()
loss_cfg  = LossCfg(
    num_keypoints=train_cfg.num_keypoints,
    match_threshold=train_cfg.match_threshold_px,
)

# ── DataLoaders ────────────────────────────────────────────────────────────
train_ds = ThermalOpticalDataset(TRAIN_PAIRS, img_size=TRAIN_SIZE, augment_color=True)
val_ds   = ThermalOpticalDataset(VAL_PAIRS[:50], img_size=TRAIN_SIZE, augment_color=False)

train_loader = DataLoader(
    train_ds, batch_size=train_cfg.batch_size, shuffle=True,
    num_workers=train_cfg.num_workers, collate_fn=collate_fn,
    pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_ds, batch_size=train_cfg.batch_size, shuffle=False,
    num_workers=0, collate_fn=collate_fn,
)

# ── Build fine-tune model (start from pretrained weights) ─────────────────
finetune_model = LoMa(_cfg)
finetune_model.to(device)

# Only train the transformer matcher layers (keep DaD + DeDoDeDescriptor frozen)
trainable_params = [
    p for name, p in finetune_model.named_parameters()
    if not name.startswith("_detector") and not name.startswith("_descriptor")
]
log.info(
    "Trainable parameters: %d / %d total",
    sum(p.numel() for p in trainable_params),
    sum(p.numel() for p in finetune_model.parameters()),
)

optimizer = torch.optim.AdamW(
    trainable_params, lr=train_cfg.lr, weight_decay=1e-4, betas=(0.9, 0.999)
)

def warmup_cosine_schedule(step: int, total_steps: int, warmup: int, lr_min: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return lr_min / train_cfg.lr + (1 - lr_min / train_cfg.lr) * 0.5 * (1 + math.cos(math.pi * progress))

total_steps   = train_cfg.num_epochs * len(train_loader) // train_cfg.grad_accum_steps
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda s: warmup_cosine_schedule(
        s, total_steps, train_cfg.warmup_steps, train_cfg.lr_min
    ),
)

scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.amp and torch.cuda.is_available())

# ── Training loop ──────────────────────────────────────────────────────────
train_losses: List[float] = []
val_metrics_history: List[Dict] = []
best_val_ratio = -1.0
global_step = 0

log.info("Starting training: %d epochs, %d steps/epoch, eff. batch=%d",
         train_cfg.num_epochs,
         len(train_loader),
         train_cfg.batch_size * train_cfg.grad_accum_steps)

for epoch in range(1, train_cfg.num_epochs + 1):
    finetune_model.train()
    epoch_losses: List[float] = []
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg.num_epochs}", leave=False)
    for step_in_epoch, batch in enumerate(pbar):
        try:
            with torch.cuda.amp.autocast(enabled=train_cfg.amp and torch.cuda.is_available()):
                loss, info = thermal_optical_loss(batch, finetune_model, loss_cfg)
                loss = loss / train_cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (step_in_epoch + 1) % train_cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            loss_val = float(loss) * train_cfg.grad_accum_steps
            epoch_losses.append(loss_val)
            train_losses.append(loss_val)

            if (step_in_epoch + 1) % train_cfg.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    gt_m=info.get("n_gt_matches", 0),
                    lr=f"{lr_now:.2e}",
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log.warning("OOM at step %d — skipping", step_in_epoch)
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            raise

    mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    log.info("Epoch %d/%d  loss=%.4f  lr=%.2e",
             epoch, train_cfg.num_epochs,
             mean_epoch_loss,
             optimizer.param_groups[0]["lr"])

    # ── Validation ────────────────────────────────────────────────────────
    if epoch % train_cfg.val_every == 0 or epoch == train_cfg.num_epochs:
        val_m = run_evaluation(
            finetune_model,
            VAL_PAIRS[:30],
            tag=f"ft_epoch{epoch}",
            max_pairs=30,
        )
        val_metrics_history.append(val_m)

        # Checkpoint best
        match_ratio = val_m.get("match_ratio", 0.0)
        ckpt_path = CKPT_DIR / f"loma_ft_epoch{epoch:03d}.pth"
        torch.save({
            "epoch":       epoch,
            "global_step": global_step,
            "model_state": {
                k: v for k, v in finetune_model.state_dict().items()
                if not k.startswith("_detector") and not k.startswith("_descriptor")
            },
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": val_m,
            "train_cfg":   train_cfg.__dict__,
        }, ckpt_path)
        log.info("Saved checkpoint: %s", ckpt_path)

        if match_ratio > best_val_ratio:
            best_val_ratio = match_ratio
            best_ckpt = CKPT_DIR / "loma_ft_best.pth"
            shutil.copy(ckpt_path, best_ckpt)
            log.info("New best match_ratio=%.4f → %s", match_ratio, best_ckpt)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

log.info("Training complete. Best match ratio: %.4f", best_val_ratio)

# ── Save training curves ──────────────────────────────────────────────────
plot_training_curves(
    train_losses,
    val_metrics_history,
    save_path=VIZ_DIR / "training_curves.png",
)

# ===========================================================================
# SECTION 10 — POST-TRAINING EVALUATION
# ===========================================================================

log.info("=" * 60)
log.info("POST-TRAINING EVALUATION")
log.info("=" * 60)

# Load best checkpoint
best_ckpt = CKPT_DIR / "loma_ft_best.pth"
if best_ckpt.exists():
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    missing, unexpected = finetune_model.load_state_dict(
        ckpt["model_state"], strict=False
    )
    if missing:
        log.warning("Missing keys in best ckpt: %s", missing[:5])
    log.info("Loaded best checkpoint from epoch %d", ckpt.get("epoch", "?"))
else:
    log.warning("No best checkpoint found — using final model weights")

finetune_model.eval()

metrics_post = run_evaluation(
    finetune_model,
    TEST_PAIRS if TEST_PAIRS else VAL_PAIRS,
    tag="finetuned",
    max_pairs=min(50, len(TEST_PAIRS or VAL_PAIRS)),
)
log.info("Post-training metrics: %s", json.dumps(
    {k: round(v, 4) for k, v in metrics_post.items() if isinstance(v, float)}, indent=2
))

with open(OUT_DIR / "metrics_finetuned.json", "w") as f:
    json.dump(metrics_post, f, indent=2)

# ===========================================================================
# SECTION 11 — FULL VISUALIZATIONS
# ===========================================================================

log.info("Generating post-training visualizations …")
generate_full_visualizations(
    finetune_model,
    (TEST_PAIRS or VAL_PAIRS)[:8],
    tag="finetuned",
    n_examples=min(8, len(TEST_PAIRS or VAL_PAIRS)),
)

# Collect post-training confidence scores
_conf_post: List[float] = []
with torch.inference_mode():
    for p in (TEST_PAIRS or VAL_PAIRS)[:20]:
        try:
            vis_t = torch.from_numpy(
                load_rgb(p.visible, TRAIN_SIZE).transpose(2, 0, 1)
            ).unsqueeze(0).to(device)
            thr_t = torch.from_numpy(
                load_gray_as_rgb(p.thermal, TRAIN_SIZE).transpose(2, 0, 1)
            ).unsqueeze(0).to(device)
            kA = finetune_model._detector.detect({"image": vis_t}, num_keypoints=512)["keypoints"]
            kB = finetune_model._detector.detect({"image": thr_t}, num_keypoints=512)["keypoints"]
            imgs = torch.cat([vis_t, thr_t], dim=0)
            kc   = torch.cat([kA, kB], dim=0)
            descs = finetune_model._descriptor.describe_keypoints(imgs, kc)["descriptions"]
            dA, dB = descs[:1], descs[1:]
            scores = finetune_model(kA, kB, dA, dB)["scores"]
            m0, _, msc0, _ = filter_matches(scores, 0.1)
            valid = m0[0] > -1
            if valid.any():
                _conf_post.extend(msc0[0][valid].cpu().tolist())
        except Exception:
            pass
_conf_post_arr = np.array(_conf_post)

# ── Confidence distribution plot ──────────────────────────────────────────
visualize_confidence_distribution(
    _conf_pre_arr, _conf_post_arr,
    save_path=VIZ_DIR / "confidence_distribution.png",
)

# ── Recall/precision ──────────────────────────────────────────────────────
plot_recall_precision(
    metrics_pre, metrics_post,
    save_path=VIZ_DIR / "recall_precision.png",
)

# ── Comparison table ──────────────────────────────────────────────────────
plot_comparison_table(
    metrics_pre, metrics_post,
    save_path=VIZ_DIR / "comparison_table.png",
)

# ── Side-by-side qualitative: pretrained vs fine-tuned on same pair ───────
if TEST_PAIRS or VAL_PAIRS:
    eval_pairs_q = (TEST_PAIRS or VAL_PAIRS)[:4]
    for qi, p in enumerate(eval_pairs_q):
        vis = load_rgb(p.visible, TRAIN_SIZE)
        thr = load_gray_as_rgb(p.thermal, TRAIN_SIZE)
        vis_t = torch.from_numpy(vis.transpose(2, 0, 1)).unsqueeze(0).to(device)
        thr_t = torch.from_numpy(thr.transpose(2, 0, 1)).unsqueeze(0).to(device)

        fig, axes = plt.subplots(1, 2, figsize=(20, 6), dpi=90)
        H, W = TRAIN_SIZE, TRAIN_SIZE
        canvas = np.zeros((H, 2 * W, 3), dtype=np.uint8)
        canvas[:, :W] = _to_uint8(vis)
        canvas[:, W:] = _to_uint8(thr)

        for ax, model_q, label_q in [
            (axes[0], pretrained_model, "Pretrained"),
            (axes[1], finetune_model,   "Fine-tuned"),
        ]:
            with torch.inference_mode():
                kA = model_q._detector.detect({"image": vis_t}, num_keypoints=512)["keypoints"]
                kB = model_q._detector.detect({"image": thr_t}, num_keypoints=512)["keypoints"]
                imgs = torch.cat([vis_t, thr_t], dim=0)
                kc   = torch.cat([kA, kB], dim=0)
                descs = model_q._descriptor.describe_keypoints(imgs, kc)["descriptions"]
                dA, dB = descs[:1], descs[1:]
                sc = model_q(kA, kB, dA, dB)["scores"]
                m0_q, _, msc_q, _ = filter_matches(sc, 0.1)
                valid_q = m0_q[0] > -1

            ax.imshow(canvas)
            if valid_q.any():
                mA = to_pixel_coords(kA[0][valid_q], H, W).cpu().numpy()
                mB = to_pixel_coords(kB[0][m0_q[0][valid_q]], H, W).cpu().numpy()
                rng_q = np.random.default_rng(0)
                n_draw = min(80, len(mA))
                idx_q  = np.random.choice(len(mA), n_draw, replace=False) if len(mA) > n_draw else np.arange(len(mA))
                for ii in idx_q:
                    color_q = tuple(rng_q.integers(80, 255, 3).tolist())
                    ax.plot([mA[ii,0], mB[ii,0]+W], [mA[ii,1], mB[ii,1]], color=np.array(color_q)/255, lw=0.7, alpha=0.8)
            ax.set_title(f"{label_q} — {int(valid_q.sum())} matches", fontsize=11)
            ax.axis("off")
        fig.suptitle(f"Thermal-Optical Matching: {p.visible.name}", fontsize=13)
        fig.savefig(VIZ_DIR / f"comparison_q{qi:02d}.png", bbox_inches="tight")
        plt.close(fig)

# ── Print final summary ───────────────────────────────────────────────────
log.info("=" * 60)
log.info("FINAL RESULTS SUMMARY")
log.info("=" * 60)
log.info("Dataset: %s  (train=%d  val=%d  test=%d)",
         DATASET_NAME, len(TRAIN_PAIRS), len(VAL_PAIRS), len(TEST_PAIRS))
log.info("")
log.info("%-26s  %10s  %10s  %10s", "Metric", "Pretrained", "Fine-tuned", "Delta")
log.info("-" * 62)
for key, label in [
    ("n_matches",   "Num matches"),
    ("match_ratio", "Match ratio"),
    ("inlier_1px",  "Inlier @1px"),
    ("inlier_5px",  "Inlier @5px"),
    ("mean_err_px", "Mean err (px)"),
    ("mean_conf",   "Mean confidence"),
]:
    pre_v  = metrics_pre.get(key, float("nan"))
    post_v = metrics_post.get(key, float("nan"))
    delta  = post_v - pre_v if not (math.isnan(pre_v) or math.isnan(post_v)) else float("nan")
    log.info("%-26s  %10.4f  %10.4f  %+10.4f", label, pre_v, post_v, delta)
log.info("=" * 60)

# ── Save consolidated results ──────────────────────────────────────────────
consolidated = {
    "dataset": DATASET_NAME,
    "n_train": len(TRAIN_PAIRS),
    "n_val":   len(VAL_PAIRS),
    "n_test":  len(TEST_PAIRS),
    "pretrained": metrics_pre,
    "finetuned":  metrics_post,
    "train_cfg":  train_cfg.__dict__,
    "best_val_match_ratio": best_val_ratio,
}
with open(OUT_DIR / "results_consolidated.json", "w") as f:
    json.dump(consolidated, f, indent=2)

log.info("All outputs saved to %s", OUT_DIR)
log.info("Visualizations saved to %s", VIZ_DIR)
log.info("Best checkpoint: %s", CKPT_DIR / "loma_ft_best.pth")
log.info("Done.")
