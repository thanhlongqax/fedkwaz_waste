"""
Federated Learning Client
=========================
Mỗi FL Client đại diện cho một nhà máy tái chế với:
- Dataset riêng (rác thải đặc thù của nhà máy)
- Mô hình riêng (phụ thuộc phần cứng)
- Camera riêng (gây ra camera heterogeneity)
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from pathlib import Path
from PIL import Image
import json
import matplotlib.pyplot as plt
from fedkwaz.kwaz_core import (
    KWAZDetector, HierarchicalAdaptivePatchMixing,
    KnowledgeDiscrepancyPerceptron, FedKWAZLoss
)

logger = logging.getLogger(__name__)


# ─── Camera Metadata Registry ─────────────────────────────────────────────────

CAMERA_META_REGISTRY = {
    "zerowaste": {
        "modality_id": 0,           # RGB
        "resolution_ratio": 1.78,   # 1920x1080 vs 1080p baseline
        "noise_level": 0.05,        # Industrial grade - low noise
        "spectral_id": 0,           # Visible spectrum
        "camera_name": "Teledyne Dalsa Genie Nano",
    },
    "spectralwaste": {
        "modality_id": 2,           # RGB + HSI
        "resolution_ratio": 0.95,   # ~1024px
        "noise_level": 0.08,        # Line-scan camera noise
        "spectral_id": 2,           # SWIR (900-1700nm)
        "camera_name": "Specim FX17 + Teledyne DALSA Linea",
    },
    "taco": {
        "modality_id": 0,           # RGB
        "resolution_ratio": 0.6,    # Mobile camera variable
        "noise_level": 0.20,        # Mobile camera higher noise
        "spectral_id": 0,           # Visible
        "camera_name": "Various (crowdsourced mobile)",
    },
    "mjuwaste": {
        "modality_id": 1,           # RGBD
        "resolution_ratio": 0.44,   # 640x480
        "noise_level": 0.12,        # Kinect structured light noise
        "spectral_id": 0,           # Visible + IR depth
        "camera_name": "Microsoft Kinect v1",
    },
}


# ─── FL Client ────────────────────────────────────────────────────────────────

class FedKWAZClient:
    """
    FL Client cho một nhà máy tái chế
    Thực hiện local training với KWAZ-guided knowledge distillation
    """

    def __init__(
        self,
        client_id: str,
        dataset_name: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg,  # FedKWAZConfig
        device: torch.device,
        output_dir="./outputs",   # thêm
    ):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        # Global prototype được server tổng hợp từ tất cả clients.
        # Client sẽ dùng prototype này để regularize feature space
        # nhằm giảm Feature Heterogeneity giữa các dataset khác nhau.
        self.global_prototype = None
        # Camera metadata
        self.camera_meta = CAMERA_META_REGISTRY.get(dataset_name, {})

        # FedKWAZ components
        self.kwaz_detector = KWAZDetector(
            feature_dim=cfg.feature_dim,
            semantic_threshold=cfg.kwaz_semantic_threshold,
            decision_threshold=cfg.kwaz_decision_threshold,
            camera_threshold=cfg.kwaz_camera_threshold,
            top_k=cfg.kwaz_top_k,
        ).to(device)

        self.hapm = HierarchicalAdaptivePatchMixing(
            patch_sizes=cfg.hapm_patch_sizes,
            num_samples=cfg.hapm_num_samples,
            mix_alpha=cfg.hapm_mix_alpha,
        )

        self.kdp = KnowledgeDiscrepancyPerceptron(
            feature_dim=cfg.feature_dim,
            temperature=cfg.kdp_temperature,
            hidden_dim=cfg.kdp_hidden_dim,
        ).to(device)

        self.criterion = FedKWAZLoss(
            distill_temperature=cfg.distill_temperature,
            distill_alpha=cfg.distill_alpha,
            distill_beta=cfg.distill_beta,
        )

        # Training stats
        self.round_stats: List[Dict] = []
        self.best_metric = 0.0
        # Monitoring sample:
        # Lưu cố định 1 ảnh đại diện của client.
        # Ảnh này được predict sau mỗi FL round để theo dõi
        # sự thay đổi confidence và prediction theo thời gian.
        self.monitor_image = None
        self.monitor_history = []
        
        #lưu ảnh vào monitor
        self.output_dir = Path(output_dir)
        self.monitor_dir = self.output_dir / "monitor"

        self.monitor_dir.mkdir(
            parents=True,
            exist_ok=True
        )
    def _build_optimizer(self) -> optim.Optimizer:
        all_params = (
            list(self.model.parameters()) +
            list(self.kwaz_detector.parameters()) +
            list(self.kdp.parameters())
        )
        return optim.AdamW(
            all_params,
            lr=self.cfg.local_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
    def receive_global_prototype(self, prototype):
        """
            Nhận Global Prototype từ FL Server.

            Global Prototype được tổng hợp từ:
            - ZeroWaste
            - SpectralWaste
            - TACO
            - MJU-Waste

            Mục tiêu:
            Giảm feature drift giữa các domain khác nhau.
        """
        if prototype is None:
            return

        self.global_prototype = prototype.to(self.device)
    def _get_pseudo_targets(self, batch: Dict) -> torch.Tensor:
        """Tạo pseudo labels từ batch (sử dụng object count hoặc dominant class)"""
        labels_list = batch["labels"]
        pseudo = []
        for labels in labels_list:
            if len(labels) > 0:
                # Lấy class xuất hiện nhiều nhất
                mode = torch.mode(labels).values.item() if len(labels) > 0 else 0
                pseudo.append(int(mode))
            else:
                pseudo.append(0)  # background
        return torch.tensor(pseudo, dtype=torch.long, device=self.device)
    @torch.no_grad()
    def extract_feature_prototypes(self):
        """
        Trích xuất Feature Prototype của client.

        Prototype = trung bình Global Feature
        trên tập validation.

        Đây là Knowledge Packet được gửi lên server
        thay cho việc gửi toàn bộ model parameters.
        """
        self.model.eval()

        proto_list = []

        for batch in self.val_loader:

            images = batch["images"].to(self.device)

            if self.dataset_name == "spectralwaste-segmentation":
                out = self.model(
                    images,
                    hsi=batch["meta"]["hsi"].to(self.device)
                )

            elif self.dataset_name == "mju-waste":
                out = self.model(
                    images,
                    depth=batch["meta"]["depth"].to(self.device)
                )

            else:
                out = self.model(images)

            proto_list.append(
                out["global_feat"].mean(0)
            )

            if len(proto_list) >= 20:
                break

        return torch.stack(proto_list).mean(0).cpu()
    def init_monitor_image(self):

        sample = self.train_loader.dataset[0]

        self.monitor_image = {}

        self.monitor_image["image"] = sample["image"]

        if "hsi" in sample:
            self.monitor_image["hsi"] = sample["hsi"]

        if "depth" in sample:
            self.monitor_image["depth"] = sample["depth"]
    @torch.no_grad()
    def extract_class_centers(self):
        """
        Tính Class Centers.

        Mỗi class sẽ có:
        center_c = mean(feature_c)

        Giúp server hiểu cấu trúc semantic
        của từng client mà không cần truy cập dữ liệu gốc.
        """
        self.model.eval()

        centers = {}

        for batch in self.val_loader:

            images = batch["images"].to(self.device)

            targets = self._get_pseudo_targets(batch)

            if self.dataset_name == "spectralwaste-segmentation":
                out = self.model(
                    images,
                    hsi=batch["meta"]["hsi"].to(self.device)
                )

            elif self.dataset_name == "mju-waste":
                out = self.model(
                    images,
                    depth=batch["meta"]["depth"].to(self.device)
                )

            else:
                out = self.model(images)

            feats = out["global_feat"]

            for cls in targets.unique():

                cls = int(cls.item())

                mask = targets == cls

                if mask.sum() == 0:
                    continue

                centers[cls] = feats[mask].mean(0).cpu()

            break

        return centers
    def local_train(
        self,
        proxy_model: nn.Module,
        current_round: int,
    ) -> Tuple[Dict, Dict]:
        """
        Local Training Phase

        1. Nhận Proxy Model từ server
        2. Forward local model
        3. KWAZ discrepancy detection
        4. Knowledge Distillation
        5. Prototype Alignment
        6. Update local weights

        Cuối round:
        Trả về Knowledge Packet thay vì state_dict.
        """
        proxy_model = proxy_model.to(self.device)
        proxy_model.eval()  # Proxy model cố định trong local training

        optimizer = self._build_optimizer()
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.local_epochs
        )

        self.model.train()
        self.kwaz_detector.train()
        self.kdp.train()

        epoch_losses = []

        for epoch in range(self.cfg.local_epochs):
            batch_losses = []

            for batch_idx, batch in enumerate(self.train_loader):
                images = batch["images"].to(self.device)
                targets = self._get_pseudo_targets(batch)

                # ── Forward pass ──────────────────────────────────
                # Private model (local)
                if self.dataset_name == "spectralwaste-segmentation" and "hsi" in batch.get("meta", {}):
                    hsi = batch["meta"]["hsi"].to(self.device)
                    private_out = self.model(images, hsi=hsi)
                elif self.dataset_name == "mju-waste" and "depth" in batch.get("meta", {}):
                    depth = batch["meta"]["depth"].to(self.device)
                    private_out = self.model(images, depth=depth)
                else:
                    private_out = self.model(images)

                # Proxy model (server-side)
                with torch.no_grad():
                    proxy_out = proxy_model(images)
                # KWAZ phát hiện các mẫu dữ liệu có mức độ
                # Knowledge Discrepancy cao giữa:
                #
                # - Local Model
                # - Proxy Model
                #
                # Các mẫu này được xem là khó học hoặc
                # có Domain Shift mạnh.
                # ── KWAZ Detection ────────────────────────────────
                kwaz_result = self.kwaz_detector(
                    private_out, proxy_out,
                    camera_meta=self.camera_meta
                )

                # ── HAPM: Generate mixed samples ──────────────────
                if kwaz_result["kwaz_mask"].any():
                    all_mixed = self.hapm(images, images, kwaz_result["kwaz_score"])

                    # KDP: Select most discrepant samples
                    kdp_imgs_list = []
                    for scale_imgs in all_mixed[:2]:  # Chỉ 2 scales đầu để tiết kiệm memory
                        for s in range(min(2, scale_imgs.size(0))):
                            kdp_imgs_list.append(scale_imgs[s])

                    if kdp_imgs_list:
                        kdp_img = kdp_imgs_list[0]
                        kdp_priv_out = self.model(kdp_img)
                        with torch.no_grad():
                            kdp_proxy_out = proxy_model(kdp_img)

                        kdp_result = self.kdp(
                            kdp_priv_out["global_feat"],
                            kdp_proxy_out["global_feat"],
                        )
                    else:
                        kdp_result = {"kdp_loss": torch.tensor(0.0, device=self.device)}
                else:
                    kdp_result = {"kdp_loss": torch.tensor(0.0, device=self.device)}
                # DEBUG CLASS MISMATCH
                num_classes = private_out["logits"].shape[1]

                if targets.max() >= num_classes:
                    print("=" * 80)
                    print("CLIENT:", self.client_id)
                    print("DATASET:", self.dataset_name)
                    print("NUM_CLASSES:", num_classes)
                    print("TARGET_MAX:", targets.max().item())
                    print("TARGET_MIN:", targets.min().item())
                    print("UNIQUE_TARGETS:", torch.unique(targets))
                    raise ValueError("Target vượt quá số class của model")
                # ── Compute Loss ──────────────────────────────────
                losses = self.criterion(
                    private_out, proxy_out,
                    targets, kwaz_result, kdp_result
                )
                # Prototype Alignment Loss
                #
                # Ép feature của client tiến gần
                # Global Prototype của toàn hệ thống.
                #
                # Đây là cơ chế giảm Statistical Heterogeneity
                # giữa các dataset khác nhau.
                if self.global_prototype is not None:

                    # proto_loss = nn.functional.mse_loss(
                    #     private_out["global_feat"],
                    #     self.global_prototype.unsqueeze(0).expand(
                    #         private_out["global_feat"].shape[0],
                    #         -1
                    #     )
                    # )
                    global_proto = (
                        self.global_prototype
                        .detach()
                        .unsqueeze(0)
                        .expand(
                            private_out["global_feat"].shape[0],
                            -1
                        )
                    )

                    proto_loss = nn.functional.mse_loss(
                        private_out["global_feat"],
                        global_proto
                    )

                    losses["total"] += 0.1 * proto_loss
                # ── Backward ──────────────────────────────────────
                optimizer.zero_grad()
                losses["total"].backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_losses.append({k: v.item() for k, v in losses.items()})

                if batch_idx % 20 == 0:
                    logger.debug(
                        f"[{self.client_id}] Round {current_round} Epoch {epoch} "
                        f"Batch {batch_idx}/{len(self.train_loader)} "
                        f"Loss: {losses['total'].item():.4f} "
                        f"KWAZ active: {kwaz_result['kwaz_mask'].sum().item()}/{images.size(0)}"
                    )

            lr_scheduler.step()

            # Average batch losses cho epoch này
            avg_losses = {k: np.mean([bl[k] for bl in batch_losses]) for k in batch_losses[0]}
            epoch_losses.append(avg_losses)

        # Summary stats
        final_stats = {
            "client_id": self.client_id,
            "dataset": self.dataset_name,
            "round": current_round,
            "num_samples": len(self.train_loader.dataset),
            "avg_total_loss": np.mean([el["total"] for el in epoch_losses]),
            "avg_task_loss": np.mean([el["task"] for el in epoch_losses]),
            "avg_kwaz_loss": np.mean([el["kwaz"] for el in epoch_losses]),
            "avg_kdp_loss": np.mean([el["kdp"] for el in epoch_losses]),
            "camera_type": self.camera_meta.get("camera_name", "unknown"),
        }

        self.round_stats.append(final_stats)
        logger.info(
            f"✅ [{self.client_id}] Round {current_round} done | "
            f"Loss: {final_stats['avg_total_loss']:.4f} | "
            f"Camera: {self.camera_meta.get('camera_name', 'N/A')}"
        )
        # ------------------------------------------------------------------
        # Knowledge Packet
        #
        # Thay vì gửi model weights lên server,
        # mỗi client chỉ gửi knowledge-level information:
        #
        # - Feature Prototypes
        # - Class Centers
        # - KWAZ Statistics
        # - Camera Metadata
        #
        # Giảm communication cost
        # và hỗ trợ Heterogeneous Federated Learning.
        # ------------------------------------------------------------------
        knowledge_packet = {
            "feature_prototypes": self.extract_feature_prototypes(),
            "class_centers": self.extract_class_centers(),
            "kwaz_statistics": {
                "avg_kwaz_loss": final_stats["avg_kwaz_loss"],
                "avg_kdp_loss": final_stats["avg_kdp_loss"],
            },
            "camera_meta": self.camera_meta,
            "num_samples": final_stats["num_samples"],
        }
        # ==============================================================
        # FedKWAZ Knowledge Communication
        #
        # Traditional FL:
        #    Client -> state_dict -> Server
        #
        # FedKWAZ:
        #    Client -> Knowledge Packet -> Server
        #
        # Knowledge Packet gồm:
        #    Feature Prototypes
        #    Class Centers
        #    KWAZ Statistics
        #    Camera Metadata
        #
        # Điều này giúp hỗ trợ:
        #    - Model Heterogeneity
        #    - Data Heterogeneity
        #    - Camera Heterogeneity
        #
        # và giảm đáng kể communication overhead.
        # ==============================================================
        return knowledge_packet, final_stats
    #hàm lưu ảnh history 
    def save_monitor_image(self):
        """
        Lưu ảnh monitor của client.

        Mỗi client chỉ lưu duy nhất 1 ảnh đại diện.
        Ảnh này được sử dụng xuyên suốt quá trình training
        để đánh giá sự ổn định của model.
        """
        if self.monitor_image is None:
            return

        save_path = self.monitor_dir / f"{self.client_id}.jpg"

        img = self.monitor_image["image"].cpu()

        img = img.permute(1,2,0).numpy()

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(save_path)

        logger.info(f"📸 Saved monitor image -> {save_path}")
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Đánh giá model trên validation set"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets = self._get_pseudo_targets(batch)

            if self.dataset_name == "spectralwaste" and "hsi" in batch.get("meta", {}):
                out = self.model(images, hsi=batch["meta"]["hsi"].to(self.device))
            elif self.dataset_name == "mjuwaste" and "depth" in batch.get("meta", {}):
                out = self.model(images, depth=batch["meta"]["depth"].to(self.device))
            else:
                out = self.model(images)

            preds = out["logits"].argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        accuracy = total_correct / max(total_samples, 1)

        return {
            "client_id": self.client_id,
            "accuracy": accuracy,
            "num_eval_samples": total_samples,
        }
    @torch.no_grad()
    def monitor_prediction(self, round_num):
        """
        Predict trên ảnh monitor cố định.

        Mục đích:
        Theo dõi sự tiến hóa của model qua từng FL round.

        Output:
        - Prediction
        - Confidence
        - Loss

        Lưu vào:
        monitor/client_x_history.json
        """
        if self.monitor_image is None:
            return

        self.model.eval()

        image = self.monitor_image["image"].unsqueeze(0).to(self.device)

        if self.dataset_name == "spectralwaste-segmentation":
            hsi = self.monitor_image["hsi"].unsqueeze(0).to(self.device)
            out = self.model(image, hsi=hsi)

        elif self.dataset_name == "mju-waste":
            depth = self.monitor_image["depth"].unsqueeze(0).to(self.device)
            out = self.model(image, depth=depth)

        else:
            out = self.model(image)

        # ==================================================
        # Activation Map
        # ==================================================

        feat_map = out["raw_feat"]

        activation = feat_map.mean(dim=1)

        activation = torch.nn.functional.interpolate(
            activation.unsqueeze(1),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze()

        activation = activation.cpu().numpy()

        activation = (
            activation - activation.min()
        ) / (
            activation.max() - activation.min() + 1e-8
        )
        probs = torch.softmax(out["logits"], dim=1)

        pred = probs.argmax(dim=1).item()
        conf = probs.max().item()

        rgb = image[0].permute(1, 2, 0).cpu().numpy()

        rgb = (
            rgb - rgb.min()
        ) / (
            rgb.max() - rgb.min() + 1e-8
        )
        # Lưu ảnh gốc chỉ 1 lần
        if round_num == 1:
            plt.imsave(
                self.monitor_dir /
                f"{self.client_id}_input.png",
                rgb
            )
        plt.figure(figsize=(5,5))

        plt.imshow(rgb)

        plt.imshow(
            activation,
            cmap="jet",
            alpha=0.5
        )

        plt.axis("off")

        plt.savefig(
            self.monitor_dir /
            f"{self.client_id}_round_{round_num}.png",
            bbox_inches="tight"
        )

        plt.close()
        self.monitor_history.append({
            "round": round_num,
            "prediction": pred,
            "confidence": conf,
            "loss": float(
                self.round_stats[-1]["avg_total_loss"]
            )
        })
        history_file = self.monitor_dir / f"{self.client_id}_history.json"

        with open(history_file, "w") as f:
            json.dump(
                self.monitor_history,
                f,
                indent=2
            )
        logger.info(
            f"📸 [{self.client_id}] "
            f"Round={round_num} "
            f"Pred={pred} "
            f"Conf={conf:.4f}"
        )
    def compress_model_update(self, state_dict: Dict) -> Dict:
        """
        Model compression để giảm băng thông mạng
        Quantization 8-bit + Top-K gradient sparsification
        """
        if not self.cfg.enable_compression:
            return state_dict

        compressed = {}
        for k, v in state_dict.items():
            if v.dtype == torch.float32:
                # 8-bit quantization
                v_min, v_max = v.min(), v.max()
                scale = (v_max - v_min) / 255.0
                if scale > 1e-8:
                    quantized = ((v - v_min) / scale).round().byte()
                    compressed[k] = {
                        "data": quantized,
                        "min": v_min,
                        "scale": scale,
                        "dtype": "int8",
                    }
                else:
                    compressed[k] = v
            else:
                compressed[k] = v

        return compressed

    def decompress_model_update(self, compressed: Dict) -> Dict:
        """Giải nén model update"""
        decompressed = {}
        for k, v in compressed.items():
            if isinstance(v, dict) and v.get("dtype") == "int8":
                decompressed[k] = v["data"].float() * v["scale"] + v["min"]
            else:
                decompressed[k] = v
        return decompressed
