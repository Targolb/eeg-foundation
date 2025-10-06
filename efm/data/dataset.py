# efm/data/dataset.py
import os
import re
import inspect
from glob import glob
from typing import List, Optional, Dict, Any

from .npz_dataset import NPZDataset

SUBJ_RE = re.compile(r"^(chb\d{2})_")  # adjust if your prefix differs


def _list_subjects_from_files(root: str) -> List[str]:
    """Find unique subject prefixes (e.g., chb01) from files like chb01_*.npz."""
    files = glob(os.path.join(root, "*.npz"))
    subs = set()
    for f in files:
        m = SUBJ_RE.match(os.path.basename(f))
        if m:
            subs.add(m.group(1))
    return sorted(subs)


def _list_subjects(root: str) -> List[str]:
    """Try by filenames at root; if empty, look one level deeper."""
    subs = _list_subjects_from_files(root)
    if subs:
        return subs
    # one level down: e.g., /data/processed/*/*.npz
    for d in sorted(p for p in glob(os.path.join(root, "*")) if os.path.isdir(p)):
        subs = _list_subjects_from_files(d)
        if subs:
            return subs
    # last fallback: any top-level dir names (unlikely useful here)
    return sorted(os.path.basename(p.rstrip("/"))
                  for p in glob(os.path.join(root, "*"))
                  if os.path.isdir(p))


def _map_for_npz(npz_cls, root: str, subjects: List[str], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Map (root, subjects, extra) to NPZDataset.__init__ kwargs safely."""
    sig = inspect.signature(npz_cls.__init__)
    params = set(sig.parameters.keys())
    out: Dict[str, Any] = {}

    # ---- roots/root mapping ----
    if "roots" in params:
        out["roots"] = extra.get("roots", [root])  # expects list
    else:
        for nm in ["root", "data_root", "dataset_root", "base_dir", "data_dir", "path", "data_path"]:
            if nm in params:
                out[nm] = root
                break

    # ---- subjects mapping (IDs) ----
    # Many datasets accept 'subjects'/'subject_ids' and filter internally.
    for nm in ["subjects", "subject_ids", "patients", "ids", "subject_list"]:
        if nm in params:
            out[nm] = subjects
            break
    else:
        # If NPZDataset doesn't accept subject IDs directly but accepts patterns/globs,
        # build per-subject file patterns like /root/chb01_*.npz
        patterns = [os.path.join(root, f"{sid}_*.npz") for sid in subjects]
        for nm in ["patterns", "globs", "file_glob", "include", "includes"]:
            if nm in params:
                out[nm] = patterns
                break
        else:
            # As a last resort, build a regex filter if supported
            regex = r"^(" + "|".join(map(re.escape, subjects)) + r")_.*\.npz$"
            for nm in ["filter_regex", "filename_regex"]:
                if nm in params:
                    out[nm] = regex
                    break
            # If none of these exist, we can't filter by subject here; NPZDataset must handle it internally.

    # ---- copy only supported extras (don’t overwrite what we set) ----
    for k, v in (extra or {}).items():
        if k in params and k not in out:
            out[k] = v

    # Helpful log
    dbg = {k: (f"[{len(v)} subs]" if (
                k in {"subjects", "subject_ids", "patients", "ids", "subject_list"} and isinstance(v, list))
               else (v if k != "roots" else [str(p) for p in v]))
           for k, v in out.items()}
    print(f"[LOSODataset] NPZDataset kwargs: {dbg}")
    return out


class LOSODataset(NPZDataset):
    """
    LOSO adapter around NPZDataset for flat 'chbXX_*.npz' layouts.
    Splits by subject id parsed from filenames.
    """

    def __init__(
            self,
            split: str,  # "train" | "eval" | "test"
            test_subject: str,
            eval_subject: Optional[str] = None,
            root: str = "/data/processed/chbmit",  # sensible default for your layout
            subjects: Optional[List[str]] = None,
            **kwargs: Dict[str, Any],
    ):
        assert split in {"train", "eval", "test"}
        all_subs = subjects if subjects else _list_subjects(root)
        if test_subject not in all_subs:
            raise ValueError(f"test_subject {test_subject} not found in {root}. Available: {all_subs}")

        if split == "test":
            use_subs = [test_subject]
        elif split == "eval":
            if eval_subject is None:
                others = [s for s in all_subs if s != test_subject]
                eval_subject = (others[0] if others else test_subject)
            use_subs = [eval_subject]
        else:  # train
            use_subs = [s for s in all_subs if s not in {test_subject, eval_subject}]

        mapped = _map_for_npz(NPZDataset, root=root, subjects=use_subs, extra=kwargs)
        super().__init__(**mapped)
