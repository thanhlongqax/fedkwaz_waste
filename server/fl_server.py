"""
FedKWAZ Prototype Server
========================

Server trung tâm cho Federated Learning dị thể.

Chức năng:
- Train Proxy Model trên Proxy Dataset
- Thu thập Knowledge Packets từ Clients
- KWAZ-aware Prototype Aggregation
- Broadcast Global Prototype
- Global Evaluation & Monitoring

Lưu ý:
Không thực hiện Weight Aggregation.
Tri thức được trao đổi thông qua Prototype Space
để hỗ trợ Heterogeneous Federated Learning.
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
import json
import time

logger = logging.getLogger(__name__)


class FedKWAZServer:

    """
    FedKWAZ Prototype Server

    Pipeline mỗi round:

    1. Train Proxy Model
    2. Phân phối Proxy Knowledge
    3. Client Local Training
    4. Thu thập Knowledge Packets
    5. Aggregate Global Prototype
    6. Broadcast Prototype Bank
    7. Evaluate & Monitor
    """
    def __init__(
        self,
        proxy_model: nn.Module,
        global_model: nn.Module,
        proxy_loader: DataLoader,
        cfg,             # FedKWAZConfig
        device: torch.device,
        output_dir: str = "./outputs",
    ):
        self.proxy_model = proxy_model.to(device)
        # self.global_model = global_model.to(device)
        self.proxy_loader = proxy_loader
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_round = 0
        self.best_global_acc = 0.0
        self.history: List[Dict] = []
        self.prototype_bank = None
        # Proxy optimizer
        self.proxy_optimizer = optim.AdamW(
            self.proxy_model.parameters(),
            lr=cfg.proxy_lr,
            weight_decay=1e-4,
        )

    # ── Proxy Model Training ──────────────────────────────────────────────────

    def train_proxy(self, num_epochs: int = 1) -> float:
        """Train proxy model trên server dataset"""
        self.proxy_model.train()
        total_loss = 0.0
        n_batches = 0

        for _ in range(num_epochs):
            for batch in self.proxy_loader:
                images = batch["images"].to(self.device)
                targets = self._batch_to_targets(batch)

                self.proxy_optimizer.zero_grad()
                out = self.proxy_model(images)
                loss = nn.functional.cross_entropy(out["logits"], targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.proxy_model.parameters(), 1.0)
                self.proxy_optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"📡 Proxy model trained | Loss: {avg_loss:.4f}")
        return avg_loss

    def _batch_to_targets(self, batch: Dict) -> torch.Tensor:
        labels_list = batch["labels"]
        targets = []
        for labels in labels_list:
            if len(labels) > 0:
                targets.append(int(torch.mode(labels).values.item()))
            else:
                targets.append(0)
        return torch.tensor(targets, dtype=torch.long, device=self.device)

    # ── Model Aggregation ─────────────────────────────────────────────────────

    # def fedavg_aggregate(
    #     self,
    #     client_updates: List[Tuple[Dict, Dict]],  # [(state_dict, stats), ...]
    # ) -> Dict:
    #     """
    #     FedAvg aggregation với sample-count weighting
    #     """
    #     if not client_updates:
    #         return self.global_model.state_dict()

    #     total_samples = sum(stats["num_samples"] for _, stats in client_updates)
    #     weights = [stats["num_samples"] / total_samples for _, stats in client_updates]

    #     # Initialize với zeros
    #     aggregated = {}
    #     ref_state = client_updates[0][0]

    #     for key in ref_state.keys():
    #         if ref_state[key].dtype in [torch.float32, torch.float16, torch.float64]:
    #             aggregated[key] = torch.zeros_like(ref_state[key])
    #             for (state_dict, _), w in zip(client_updates, weights):
    #                 if key in state_dict:
    #                     # Handle potential shape mismatch (model heterogeneity)
    #                     if state_dict[key].shape == aggregated[key].shape:
    #                         aggregated[key] += w * state_dict[key].float()
    #         else:
    #             aggregated[key] = ref_state[key]

    #     return aggregated
    # # def aggregate_prototypes(self, client_updates):

    # #     all_proto = []
    # #     weights = []

    # #     for packet, stats in client_updates:

    # #         all_proto.append(
    # #             packet["feature_prototypes"].to(self.device)
    # #         )

    # #         weights.append(
    # #             packet["num_samples"]
    # #         )

    # #     weights = np.array(weights)
    # #     weights = weights / weights.sum()

    # #     global_proto = torch.zeros_like(all_proto[0])

    # #     for w, p in zip(weights, all_proto):
    # #         global_proto += float(w) * p

    # #     return global_proto
    def aggregate_prototypes(self, client_updates):
        """
        Aggregate prototype vectors từ các client.

        client_updates:
        [
            (knowledge_packet, stats),
            ...
        ]
        """

        if len(client_updates) == 0:
            return None

        # ==================================================
        # Sample Importance
        # Client có nhiều dữ liệu hơn đóng góp nhiều hơn
        # ==================================================

        total_samples = sum(
            stats["num_samples"]
            for _, stats in client_updates
        )

        sample_weights = np.array([
            stats["num_samples"] / total_samples
            for _, stats in client_updates
        ])

        # ==================================================
        # Learning Quality Weight
        # KWAZ loss càng thấp => Prototype càng đáng tin cậy
        # ==================================================

        kwaz_losses = np.array([
            stats.get("avg_kwaz_loss", 1.0)
            for _, stats in client_updates
        ])

        inv_kwaz = 1.0 / (kwaz_losses + 1e-8)

        quality_weights = inv_kwaz / inv_kwaz.sum()

        # ==================================================
        # Sensor Diversity Weight
        # Khuyến khích các nguồn camera hiếm
        # đóng góp nhiều hơn vào Prototype Bank
        # ==================================================

        camera_types = [
            stats.get("camera_type", "unknown")
            for _, stats in client_updates
        ]

        camera_bonus = np.ones(len(client_updates))

        if len(set(camera_types)) > 1:

            for i, cam in enumerate(camera_types):

                count = camera_types.count(cam)

                camera_bonus[i] = 1.0 / count

            camera_bonus = camera_bonus / camera_bonus.sum()

        # ==================================================
        # Hybrid Prototype Weight
        # 50% Sample
        # 30% Learning Quality
        # 20% Sensor Diversity
        # ==================================================

        final_weights = (
            0.5 * sample_weights +
            0.3 * quality_weights +
            0.2 * camera_bonus
        )

        final_weights = final_weights / final_weights.sum()

        # ==================================================
        # Aggregate Prototype
        # ==================================================

        proto_dim = client_updates[0][0]["feature_prototypes"].shape[0]

        global_proto = torch.zeros(
            proto_dim,
            device=self.device
        )

        for i, (packet, stats) in enumerate(client_updates):

            proto = packet["feature_prototypes"].to(self.device)

            global_proto += final_weights[i] * proto

        logger.info(
            "🧠 Prototype aggregation | "
            + " | ".join([
                f"{client_updates[i][1]['client_id']}: {final_weights[i]:.3f}"
                for i in range(len(client_updates))
            ])
        )

        return global_proto
    # def kwaz_aware_aggregate(
    #     self,
    #     client_updates: List[Tuple[Dict, Dict]],
    # ) -> Dict:
    #     """
    #     KWAZ-aware aggregation: Clients với KWAZ loss thấp hơn
    #     (đã học tốt hơn) nhận weight cao hơn trong aggregation
    #     Novel contribution: kết hợp sample count + learning quality
    #     """
    #     if not client_updates:
    #         return self.global_model.state_dict()

    #     total_samples = sum(s["num_samples"] for _, s in client_updates)
    #     sample_weights = np.array([s["num_samples"] / total_samples for _, s in client_updates])

    #     # Quality weight: inverse KWAZ loss (lower KWAZ = better learned)
    #     kwaz_losses = np.array([
    #         s.get("avg_kwaz_loss", 1.0) for _, s in client_updates
    #     ])
    #     # Softmax inverse
    #     inv_kwaz = 1.0 / (kwaz_losses + 1e-8)
    #     quality_weights = inv_kwaz / inv_kwaz.sum()

    #     # Camera diversity bonus: clients với camera khác biệt nhận bonus
    #     camera_types = [s.get("camera_type", "unknown") for _, s in client_updates]
    #     unique_cameras = len(set(camera_types))
    #     camera_bonus = np.ones(len(client_updates))
    #     if unique_cameras > 1:
    #         for i, cam in enumerate(camera_types):
    #             # Rare camera type nhận bonus cao hơn
    #             count = camera_types.count(cam)
    #             camera_bonus[i] = 1.0 / count
    #         camera_bonus = camera_bonus / camera_bonus.sum()

    #     # Final weights: 50% sample + 30% quality + 20% camera diversity
    #     final_weights = (
    #         0.5 * sample_weights +
    #         0.3 * quality_weights +
    #         0.2 * camera_bonus
    #     )
    #     final_weights = final_weights / final_weights.sum()

    #     # Aggregate
    #     aggregated = {}
    #     ref_state = client_updates[0][0]

    #     for key in ref_state.keys():
    #         if ref_state[key].dtype in [torch.float32, torch.float16, torch.float64]:
    #             aggregated[key] = torch.zeros_like(ref_state[key])
    #             for i, (state_dict, _) in enumerate(client_updates):
    #                 if key in state_dict and state_dict[key].shape == aggregated[key].shape:
    #                     aggregated[key] += final_weights[i] * state_dict[key].float()
    #         else:
    #             aggregated[key] = ref_state[key]

    #     logger.info(
    #         f"🔀 KWAZ-aware aggregation | Weights: "
    #         + " | ".join([
    #             f"{client_updates[i][1]['client_id']}: {final_weights[i]:.3f}"
    #             for i in range(len(client_updates))
    #         ])
    #     )
    #     return aggregated

    # def broadcast_global_model(self, client_model: nn.Module) -> nn.Module:
    #     """
    #     Broadcast global knowledge sang client model
    #     Xử lý heterogeneous architectures bằng cách chỉ copy các layers có shape matching
    #     """
    #     global_state = self.global_model.state_dict()
    #     client_state = client_model.state_dict()

    #     updated_state = copy.deepcopy(client_state)
    #     copied_count = 0

    #     for key in client_state.keys():
    #         # Tìm matching key trong global model (có thể khác tên)
    #         if key in global_state and global_state[key].shape == client_state[key].shape:
    #             updated_state[key] = global_state[key]
    #             copied_count += 1

    #     logger.debug(f"Broadcast: {copied_count}/{len(client_state)} layers synced")
    #     client_model.load_state_dict(updated_state, strict=False)
    #     return client_model

    # ── Main FL Round ─────────────────────────────────────────────────────────

    def run_round(
        self,
        clients: List,
        use_kwaz_aggregation: bool = True,
    ) -> Dict:
        """
        Thực hiện một FL round hoàn chỉnh:
        1. Train proxy model
        2. Distribute proxy to clients
        3. Clients perform local training
        4. Aggregate updates
        5. Evaluate
        """
        self.current_round += 1
        round_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 FL ROUND {self.current_round}/{self.cfg.num_rounds}")
        logger.info(f"{'='*60}")

        # Step 1: Update proxy model
        proxy_loss = self.train_proxy(num_epochs=1)

        # Step 2: Collect Knowledge Packets from Clients
        client_updates = []
        for client in clients:
            logger.info(f"  → Training {client.client_id} ({client.dataset_name})...")

            # # Distribute current proxy
            # proxy_copy = copy.deepcopy(self.proxy_model)
            proxy_copy = self.proxy_model
            # Local training
            state_dict, stats = client.local_train(
                proxy_model=proxy_copy,
                current_round=self.current_round,
            )

            # # Optional compression
            # if self.cfg.enable_compression:
            #     state_dict = client.compress_model_update(state_dict)
            #     state_dict = client.decompress_model_update(state_dict)

            client_updates.append((state_dict, stats))
            

        # # Step 3: Aggregation
        # if use_kwaz_aggregation:
        #     aggregated_state = self.kwaz_aware_aggregate(client_updates)
        # else:
        #     aggregated_state = self.fedavg_aggregate(client_updates)

        # # Update global model (với state dict từ first client làm reference shape)
        # try:
        #     self.global_model.load_state_dict(aggregated_state, strict=False)
        # except Exception as e:
        #     logger.warning(f"Global model update warning: {e}")

        # Step 3: Build Global Prototype Bank
        self.prototype_bank = self.aggregate_prototypes(
            client_updates
        )

        logger.info(
            f"🧠 Prototype bank updated "
            f"| Shape: {self.prototype_bank.shape}"
        )
        # Step 4: Broadcast Global Prototype
        # Client sử dụng Prototype Distillation
        # trong vòng train tiếp theo
        for client in clients:
            # client.model = self.broadcast_global_model(client.model)
            client.receive_global_prototype(
                self.prototype_bank
            )
            client.monitor_prediction(
                    round_num=self.current_round
                )
        # Step 5: Evaluate
        round_metrics = {}
        if self.current_round % self.cfg.eval_every == 0:
            for client in clients:
                metrics = client.evaluate()
                round_metrics[client.client_id] = metrics
                logger.info(
                    f"  📊 {client.client_id}: "
                    f"Acc = {metrics['accuracy']:.4f}"
                )

        # Round summary
        round_time = time.time() - round_start
        round_summary = {
            "round": self.current_round,
            "proxy_loss": proxy_loss,
            "client_stats": [stats for _, stats in client_updates],
            "eval_metrics": round_metrics,
            "round_time_sec": round_time,
            "aggregation": "kwaz_aware" if use_kwaz_aggregation else "fedavg",
        }

        if round_metrics:
            avg_acc = np.mean([m["accuracy"] for m in round_metrics.values()])
            round_summary["avg_accuracy"] = avg_acc
            if avg_acc > self.best_global_acc:
                self.best_global_acc = avg_acc
                self.save_checkpoint("best_model.pt")

        self.history.append(round_summary)
        logger.info(
            f"⏱ Round {self.current_round} completed in {round_time:.1f}s"
        )
        return round_summary
    # Prototype Checkpointing
    # Lưu trạng thái Federated Knowledge
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        checkpoint = {
            "round": self.current_round,

            "prototype_bank":
                self.prototype_bank,

            "best_acc":
                self.best_global_acc,

            "history":
                self.history
        }
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"💾 Checkpoint saved: {path}")

    # def load_checkpoint(self, path: str):
    #     checkpoint = torch.load(path, map_location=self.device)
    #     self.current_round = checkpoint["round"]
    #     self.global_model.load_state_dict(checkpoint["global_model_state"], strict=False)
    #     self.proxy_model.load_state_dict(checkpoint["proxy_model_state"], strict=False)
    #     self.best_global_acc = checkpoint["best_acc"]
    #     self.history = checkpoint.get("history", [])
    #     logger.info(f"📂 Checkpoint loaded from round {self.current_round}")
    def load_checkpoint(self, path: str):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.current_round = checkpoint["round"]

        self.prototype_bank = checkpoint.get(
            "prototype_bank",
            None
        )

        self.best_global_acc = checkpoint["best_acc"]

        self.history = checkpoint.get(
            "history",
            []
        )

        logger.info(
            f"📂 Prototype checkpoint loaded "
            f"from round {self.current_round}"
        )
    def save_history(self, filename: str = "training_history.json"):
        path = self.output_dir / filename
        # Convert tensors to Python types for JSON serialization
        def to_serializable(obj):
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                return float(obj) if obj.ndim == 0 else obj.tolist()
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return float(obj)
            return obj

        with open(path, "w") as f:
            json.dump(to_serializable(self.history), f, indent=2)
        logger.info(f"📄 History saved: {path}")
