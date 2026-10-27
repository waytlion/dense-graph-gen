#!/usr/bin/env python3
"""Convert DeFoG .pt datasets into DiPhon .pkl format (dict of NetworkX graphs)."""

from pathlib import Path
import pickle
import networkx as nx
import torch

DATA_DIR = Path("data")
SPLITS = ("train", "val", "test")


def convert_dataset(ds_dir: Path):
    out_path = DATA_DIR / f"{ds_dir.name}.pkl"
    if out_path.exists():
        return print(f"Skipping {ds_dir.name}: {out_path} already exists.")
    if not all((ds_dir / f"{s}.pt").exists() for s in SPLITS):
        return print(f"Skipping {ds_dir.name}: missing split .pt files.")

    print(f"Converting {ds_dir.name} -> {out_path}...")
    dataset = {
        s: [nx.from_numpy_array(A.cpu().numpy() if torch.is_tensor(A) else A)
            for A in torch.load(ds_dir / f"{s}.pt", weights_only=False)]
        for s in SPLITS
    }
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {out_path} ({', '.join(f'{s}: {len(dataset[s])}' for s in SPLITS)}).")


def main():
    for d in sorted(p for p in DATA_DIR.glob("cograph_*") if p.is_dir()):
        convert_dataset(d)


if __name__ == "__main__":
    main()

