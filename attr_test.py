import os
import argparse
import torch
import torchvision.utils as vutils

from dataset import ZapposDataset
from models_film import Generator


def set_attrs_by_name(attrs: torch.Tensor, name_to_idx: dict, edits: dict) -> torch.Tensor:
    """
    attrs: [B, attr_dim]
    edits: {"Category.Boots": 1.0, ...}
    """
    out = attrs.clone()
    for name, val in edits.items():
        if name not in name_to_idx:
            raise ValueError(f"Unknown attribute name: {name}")
        out[:, name_to_idx[name]] = float(val)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="path to best.pth")
    ap.add_argument("--data_root", type=str, required=True, help="dataset root")
    ap.add_argument("--out_dir", type=str, default="./attr_test_out", help="where to save images")
    ap.add_argument("--sample_idx", type=int, default=0, help="which val sample to use")
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # attribute column order (must exist because you saved dataset_meta in Trainer)
    cols = ckpt["dataset_meta"]["attribute_columns"]
    name_to_idx = {n: i for i, n in enumerate(cols)}
    print("attr_dim:", ckpt["attr_dim"])
    print("num cols:", len(cols))

    # build generator and load weights
    g = Generator(attr_dim=ckpt["attr_dim"], ngf=ckpt["args"].ngf).to(device)
    g.load_state_dict(ckpt["generator_state_dict"])
    g.eval()

    # build val dataset using saved training args
    train_args = ckpt["args"]
    val_ds = ZapposDataset(
        data_root=args.data_root,
        image_size=train_args.image_size,
        split="val",
        train_ratio=train_args.train_ratio,
        min_attribute_freq=train_args.min_attr_freq,
        attribute_prefixes=train_args.attr_prefixes.split(",") if train_args.attr_prefixes else None,
        attribute_columns=cols,   # IMPORTANT: use same columns as training
    )

    edges, attr, real = val_ds[args.sample_idx]
    edges = edges.unsqueeze(0).to(device)   # [1,3,H,W]
    attr = attr.unsqueeze(0).to(device)     # [1,attr_dim]
    real = real.unsqueeze(0)                # keep on CPU for saving

    # 1) original
    gen_orig = g(edges, attr).cpu()

    # --- attribute edit presets (safe: turn on one category and turn off the others) ---
    # You can tweak these.
    boots = {
        "Category.Boots": 1.0,
        "Category.Sandals": 0.0,
        "Category.Slippers": 0.0,
        "Category.Shoes": 0.0,
    }
    sandals = {
        "Category.Sandals": 1.0,
        "Category.Boots": 0.0,
        "Category.Slippers": 0.0,
        "Category.Shoes": 0.0,
    }
    heels = {
        "SubCategory.Heels": 1.0,
        "HeelHeight.3in...3.3.4in": 1.0,   # optional, remove if not in your columns
        "HeelHeight.4in...4.3.4in": 0.0,
        "HeelHeight.Flat": 0.0,
    }

    # guard: drop any edits not present in cols (helps if a column name differs)
    def filter_edits(edits):
        return {k: v for k, v in edits.items() if k in name_to_idx}

    attr_boots = set_attrs_by_name(attr, name_to_idx, filter_edits(boots))
    attr_sandals = set_attrs_by_name(attr, name_to_idx, filter_edits(sandals))
    attr_heels = set_attrs_by_name(attr, name_to_idx, filter_edits(heels))

    gen_boots = g(edges, attr_boots).cpu()
    gen_sandals = g(edges, attr_sandals).cpu()
    gen_heels = g(edges, attr_heels).cpu()

    # save: real / gen_orig / gen_boots / gen_sandals / gen_heels
    os.makedirs(args.out_dir, exist_ok=True)
    grid = torch.cat([real, gen_orig, gen_boots, gen_sandals, gen_heels], dim=0)
    grid = (grid + 1) / 2  # [-1,1] -> [0,1]
    out_path = os.path.join(args.out_dir, f"sample_{args.sample_idx:04d}_attr_edits.png")
    vutils.save_image(grid, out_path, nrow=5)
    print("Saved:", out_path)

    # also print which edits actually applied
    print("\nApplied boots edits:", list(filter_edits(boots).keys()))
    print("Applied sandals edits:", list(filter_edits(sandals).keys()))
    print("Applied heels edits:", list(filter_edits(heels).keys()))


if __name__ == "__main__":
    main()