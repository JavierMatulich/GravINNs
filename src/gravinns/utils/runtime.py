"""Device selection and global seeding (notebook cell 0, sections 0-1)."""
import numpy as np
import torch


def get_device(verbose: bool = False) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    return device


def set_global_seed(seed: int = 42) -> None:
    """The notebook's 'reproducibility lock'.  Every training run re-seeds
    itself from its own ``seed`` argument, so this only affects code that runs
    outside the trainer."""
    torch.manual_seed(seed)
    np.random.seed(seed)
