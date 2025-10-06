import os, glob, numpy as np, torch
from torch.utils.data import Dataset


# -------------------------
# subject-id extraction
# -------------------------

def extract_sid(path: str) -> str:
    base = os.path.basename(path)

    # CHB-MIT: chb04_15__... -> "chb04"
    if base.startswith("chb"):
        return base.split("_")[0]

    # TUH/TUSZ: aaaaapwd_s001_t000__... -> "s001"
    if "_s" in base:
        try:
            return "s" + base.split("_s")[1].split("_")[0]  # keep "s001"
        except Exception:
            return "unknown"

    return "unknown"


# -------------------------
# dataset
# -------------------------

class NPZDataset(Dataset):
    """
    Load preprocessed EEG windows from .npz files.
    Each sample returns dict with:
      x: (C,T) EEG array
      mask_missing: (C,)
      mask_artifact: scalar (0.0/1.0)
      y: label (0 or 1)
      subject_id: string ID
      path: filename
    """

    def __init__(self, roots, subjects=None):
        self.files = []
        for r in roots:
            self.files += glob.glob(os.path.join(r, "**", "*.npz"), recursive=True)
        self.files.sort()

        if subjects is not None:
            self.files = [f for f in self.files if extract_sid(f) in subjects]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        d = np.load(p, allow_pickle=True)

        x = torch.tensor(d["x"], dtype=torch.float32)  # (C,T)
        mm = torch.tensor(d["mask_missing"], dtype=torch.bool)  # (C,)
        ma = d.get("mask_artifact", None)
        ma = torch.tensor(float(ma), dtype=torch.float32) if ma is not None else torch.tensor(0.0)
        y = torch.tensor(int(d["label"]), dtype=torch.long)

        sid = extract_sid(p)

        return {
            "x": x,
            "mask_missing": mm,
            "mask_artifact": ma,
            "y": y,
            "subject_id": sid,
            "path": p,
        }


# -------------------------
# helper for LOSO
# -------------------------

def list_subjects(roots):
    """
    Collect all subject IDs available under given roots.
    """
    subs = set()
    for r in roots:
        for p in glob.glob(os.path.join(r, "**", "*.npz"), recursive=True):
            subs.add(extract_sid(p))
    return sorted(list(subs))
