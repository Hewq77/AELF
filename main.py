# -*- coding: utf-8 -*-
import os
import time
import datetime
from pathlib import Path
import argparse
import torch
import torch.nn as nn

from models.AELF import AELF
from utils import initialize_logger, time2file_name
from data_loader import build_datasets
from train import train, validate
from loss_utils import CVLoss

# --------------------------- argparse ---------------------------

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='AELF')
    parser.add_argument('-dataset', type=str, default='Pavia',
                        choices=['Pavia', 'Houston2018', 'Chikusei'])
    parser.add_argument('--scale_ratio', type=int, default=4)
    parser.add_argument('--n_select_bands', type=int, default=4)
    parser.add_argument('--model_path', type=str, default='./checkpoints', help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='./data/dataset/train', help='directory for train data')
    parser.add_argument('--val_dir', type=str, default='./data/dataset/val', help='directory for val data')
    parser.add_argument('--pretrained_model_path', type=str, default=None)

    # training settings
    parser.add_argument('--n_epochs', type=int, default=10000, help='end epoch for training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=128)

    args = parser.parse_args()
    return args

# --------------------------- helpers ---------------------------

def dataset_bands(name: str) -> int:
    mapping = {"Pavia": 102, "Houston2018": 48, "Chikusei": 128}
    if name not in mapping:
        raise ValueError(f"Unknown dataset: {name}")
    return mapping[name]


def build_model(args) -> torch.nn.Module:
    
    if args.arch == "AELF":
        model = AELF(
            hsi_c=args.n_bands,
            msi_c=args.n_select_bands,
            scale_factor=args.scale_ratio,
            n_feat=64,
            scale_unetfeats=32,
            kernel_size=3,
            bias=True,
        )
    else:
        raise ValueError(f"Unknown architecture name: {args.arch}")
    return model.cuda()

# --------------------------- main ---------------------------

def main():
    args = args_parser()
    print(args)

    # dataset
    args.root = str(Path("../../Datasets") / args.dataset)
    args.n_bands = dataset_bands(args.dataset)

    train_list, test_list = build_datasets(
        args.root, args.dataset, args.image_size, args.n_select_bands, args.scale_ratio
    )

    # model / opt / loss
    model = build_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = {
        "CVLoss": CVLoss().cuda(),
        "l1": nn.L1Loss().cuda()
    }
    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Info] Model size: {num_params_m:.3f}M")

    # paths
    timestamp = time2file_name(str(datetime.datetime.now()))
    out_dir = Path(args.model_path) / args.arch / f"{args.dataset}×{args.scale_ratio}" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = initialize_logger(str(out_dir / "train.log"))

    # resume
    if args.pretrained_model_path and Path(args.pretrained_model_path).is_file():
        state = torch.load(args.pretrained_model_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[Info] Loaded checkpoint: {args.pretrained_model_path}")
        _, psnr, _, _ = validate(test_list, args.arch, args.scale_ratio, model)
        print(f"[Resume Init] psnr: {psnr:.4f}")

    # initial evaluation
    _, best_psnr, _, _ = validate(test_list, args.arch, args.scale_ratio, model)
    print(f"[Init Eval] psnr: {best_psnr:.4f}")

    print("[Info] Start Training")
    best_epoch = -1
    t0 = time.time()

    for epoch in range(args.n_epochs):
        print(f"[Train] Epoch {epoch}")
        train(train_list, args.image_size, args.scale_ratio, args.n_bands,
              args.arch, model, optimizer, criterion, epoch, args.n_epochs)

        rmse, psnr, ergas, sam = validate(test_list, args.arch, args.scale_ratio, model)
        print(f"[Val] rmse:{rmse:.4f} psnr:{psnr:.4f} ergas:{ergas:.4f} sam:{sam:.4f} best:{best_psnr:.4f}")

        if epoch % 20 == 0:
            logger.info(f"Epoch [{epoch}/{args.n_epochs}] rmse:{rmse:.6f} psnr:{psnr:.6f} "
                        f"ergas:{ergas:.6f} sam:{sam:.6f} best:{best_psnr:.6f}")

        if psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch
            if epoch >= 500:
                ckpt = out_dir / f"net_{epoch}epoch.pth"
                torch.save(model.state_dict(), ckpt)
                print(f"[Save] {ckpt}")

    elapsed = time.time() - t0
    print(f"[Done] best_psnr:{best_psnr:.4f} at epoch {best_epoch} | time(s): {elapsed:.1f}")
    print(f"[Info] Model size: {num_params_m:.3f}M")

if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()
