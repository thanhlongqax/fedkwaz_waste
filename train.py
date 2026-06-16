"""
FedKWAZ Main Training Script
==============================
Entry point để chạy toàn bộ FL pipeline cho quản lý rác thải đa nhà máy

Cách chạy:
  python train.py --mode demo          # Chạy demo với synthetic data
  python train.py --mode full          # Chạy với dataset thực
  python train.py --mode eval          # Chỉ evaluate checkpoint
  python train.py --num_rounds 100 --num_clients 4
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np
import random
from pathlib import Path
from utils.monitor_plot import plot_monitor_history
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import (
    get_fedkwaz_config, get_training_config,
    CLIENT_DATASET_MAP, CLIENT_MODEL_MAP, DATASET_REGISTRY
)
from datasets.waste_datasets import (
    build_dataset, build_dataloader, build_demo_dataset
)
from models.architectures import build_client_model, ProxyModel, count_parameters
from client.fl_client import FedKWAZClient
from server.fl_server import FedKWAZServer
from utils.visualization import plot_training_curves, print_round_summary
from utils.metrics import compute_per_class_metrics
from utils.visualization import (
    plot_training_curves,
    print_round_summary,
    create_attention_progress_figure
)

# ─── Logging Setup ───────────────────────────────────────────────────────────

def setup_logging(log_dir: str, level: str = "INFO"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "fedkwaz_training.log"),
        ],
    )


# ─── Seed ────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Build Clients ───────────────────────────────────────────────────────────

def build_clients(
    fl_cfg,
    train_cfg,
    device: torch.device,
    demo_mode: bool = False,
) -> list:
    """Khởi tạo tất cả FL clients với dataset và model tương ứng"""
    clients = []
    logger = logging.getLogger(__name__)

    dataset_num_classes = {
        name: info["num_classes"]
        for name, info in DATASET_REGISTRY.items()
    }

    for client_id, dataset_name in CLIENT_DATASET_MAP.items():
        
        # Chỉ chạy client_3 để debug TACO
        # if client_id != "client_3":
        #     continue
        model_name = CLIENT_MODEL_MAP[client_id]
        num_classes = dataset_num_classes.get(dataset_name, 4)

        logger.info(f"\n📦 Building {client_id} | Dataset: {dataset_name} | Model: {model_name}")
        print("="*60)
        print("CLIENT_ID:", client_id)
        print("DATASET_NAME:", dataset_name)
        print("MODEL_NAME:", model_name)
        print("NUM_CLASSES:", num_classes)
        # print("REGISTRY_KEYS:", list(dataset_num_classes.keys()))
        print("="*60)
        # Build model
        model_kwargs = {}
        if model_name == "efficientnet_b3":
            model_kwargs["use_hsi"] = True
        elif model_name == "mobilenetv3":
            model_kwargs["use_depth"] = True

        model = build_client_model(model_name, num_classes, fl_cfg.feature_dim, **model_kwargs)
        logger.info(f"  Model params: {count_parameters(model)}")

        # Build dataset
        if demo_mode:
            train_ds = build_demo_dataset(dataset_name, num_samples=200)
            val_ds = build_demo_dataset(dataset_name, num_samples=50)
        else:
            try:
                train_ds = build_dataset(
                    dataset_name, train_cfg.data_root,
                    split="train", img_size=train_cfg.img_size
                )
                val_ds = build_dataset(
                    dataset_name, train_cfg.data_root,
                    split="val", img_size=train_cfg.img_size
                )
                logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
            except FileNotFoundError as e:
                logger.warning(f"  ⚠️  {e}")
                logger.warning(f"  → Switching to demo mode for {client_id}")
                train_ds = build_demo_dataset(dataset_name, num_samples=200)
                val_ds = build_demo_dataset(dataset_name, num_samples=50)

        from torch.utils.data import DataLoader

        def collate_fn(batch):
            images = torch.stack([b["image"] for b in batch])
            boxes = [b["boxes"] for b in batch]
            labels = [b["labels"] for b in batch]
            meta = {
                "image_paths": [b.get("image_path", "") for b in batch],
                "dataset": batch[0].get("dataset", dataset_name),
                "camera_type": batch[0].get("camera_type", "unknown"),
            }
            if "hsi" in batch[0]:
                meta["hsi"] = torch.stack([b["hsi"] for b in batch])
            if "depth" in batch[0]:
                meta["depth"] = torch.stack([b["depth"] for b in batch])
            return {"images": images, "boxes": boxes, "labels": labels, "meta": meta}

        train_loader = DataLoader(
            train_ds,
            batch_size=fl_cfg.local_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True, # 0 for demo mode compatibility
            collate_fn=collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=fl_cfg.local_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

        client = FedKWAZClient(
            client_id=client_id,
            dataset_name=dataset_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=fl_cfg,
            device=device,
            output_dir=train_cfg.output_dir,
        )
        clients.append(client)
        client.monitor_image = train_ds[0]
        client.save_monitor_image()
        logger.info(f"  ✅ {client_id} ready")

    return clients


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(args):
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    # Configs
    fl_cfg = get_fedkwaz_config(
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        local_batch_size=args.batch_size,
    )
    train_cfg = get_training_config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        img_size=args.img_size,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Device: {device}")

    demo_mode = (args.mode == "demo")
    logger.info(f"🔧 Mode: {'DEMO (synthetic data)' if demo_mode else 'FULL (real datasets)'}")

    # ── Build Clients ────────────────────────────────────────────────────────
    clients = build_clients(fl_cfg, train_cfg, device, demo_mode=demo_mode)

    # ── Build Server ─────────────────────────────────────────────────────────
    # Proxy model: ResNet18 trên server
    proxy_model = ProxyModel(num_classes=10, feature_dim=fl_cfg.feature_dim)

    # Global model: cùng kiến trúc với client_0 (ResNet50) làm reference
    global_model = build_client_model("resnet50", num_classes=4, feature_dim=fl_cfg.feature_dim)

    # Proxy loader (dùng dataset của client_0 làm proxy)
    proxy_ds = build_demo_dataset("zerowaste-f-final", num_samples=100) if demo_mode else \
        build_dataset("zerowaste-f-final", train_cfg.data_root, "train", train_cfg.img_size)

    from torch.utils.data import DataLoader

    def simple_collate(batch):
        images = torch.stack([b["image"] for b in batch])
        labels_list = [b["labels"] for b in batch]
        targets = []
        for lbs in labels_list:
            targets.append(int(lbs[0].item()) if len(lbs) > 0 else 0)
        return {
            "images": images,
            "boxes": [b["boxes"] for b in batch],
            "labels": [b["labels"] for b in batch],
            "meta": {}
        }

    proxy_loader = DataLoader(
        proxy_ds,
        batch_size=fl_cfg.local_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=simple_collate,
    )

    server = FedKWAZServer(
        proxy_model=proxy_model,
        global_model=global_model,
        proxy_loader=proxy_loader,
        cfg=fl_cfg,
        device=device,
        output_dir=train_cfg.output_dir,
    )

    # Load checkpoint nếu resume
    if args.resume and Path(args.resume).exists():
        server.load_checkpoint(args.resume)
        logger.info(f"Resuming from round {server.current_round}")

    # ── FL Training Loop ─────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  FedKWAZ Industrial Waste Management")
    logger.info(f"  Clients: {len(clients)} factories")
    logger.info(f"  Rounds: {fl_cfg.num_rounds}")
    logger.info(f"  Local epochs: {fl_cfg.local_epochs}")
    logger.info(f"{'='*60}\n")

    for round_num in range(server.current_round, fl_cfg.num_rounds):
        round_summary = server.run_round(
            clients=clients,
            use_kwaz_aggregation=True,
        )

        # Print summary
        print_round_summary(round_summary)

        # Save periodically
        if (round_num + 1) % 10 == 0:
            server.save_checkpoint(f"checkpoint_round_{round_num+1}.pt")

    # Final save
    server.save_checkpoint("final_model.pt")
    server.save_history("training_history.json")

    # Plot curves
    plot_training_curves(server.history, output_dir=train_cfg.output_dir)
    logger.info(f"\n🏁 Training complete! Best accuracy: {server.best_global_acc:.4f}")
    plot_monitor_history( Path(train_cfg.output_dir) / "monitor")
    for client in clients:
        create_attention_progress_figure(
            monitor_dir=Path(train_cfg.output_dir) / "monitor",
            client_id=client.client_id,
            first_round=1,
            last_round=fl_cfg.num_rounds
        )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FedKWAZ: Federated Learning for Industrial Waste Management"
    )
    parser.add_argument("--mode", choices=["demo", "full", "eval"], default="demo",
                        help="demo=synthetic data, full=real datasets, eval=evaluate only")
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=160,
                        help="Image size (smaller for demo, 640 for full)")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()
    setup_logging(args.output_dir, args.log_level)
    train(args)


if __name__ == "__main__":
    main()
