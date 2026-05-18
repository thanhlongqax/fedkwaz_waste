"""
FedKWAZ Configuration for Industrial Waste Management
=====================================================
Hệ thống cấu hình tập trung cho toàn bộ pipeline FedKWAZ
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os

# ─── Dataset Registry ───────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "zerowaste": {
        "name": "ZeroWaste-f",
        "url": "https://zenodo.org/record/6412647",
        "num_classes": 4,
        "classes": ["rigid_plastic", "cardboard", "metal", "soft_plastic"],
        "camera_type": "industrial_rgb",
        "resolution": (1920, 1080),
        "annotation_format": "COCO",
        "description": "Conveyor belt waste detection - CVPR 2022",
    },
    "spectralwaste": {
        "name": "SpectralWaste",
        "url": "https://zenodo.org/records/10880544",
        "num_classes": 6,
        "classes": ["film", "basket", "video_tape", "filaments", "trash_bag", "cardboard"],
        "camera_type": "rgb_hyperspectral",
        "resolution": (1024, 1024),
        "annotation_format": "COCO",
        "description": "RGB + Hyperspectral industrial sorting - IROS 2024",
    },
    "taco": {
        "name": "TACO",
        "url": "http://tacodataset.org",
        "num_classes": 60,
        "classes": None,  # Loaded dynamically
        "camera_type": "crowdsourced_mobile",
        "resolution": None,  # Variable
        "annotation_format": "COCO",
        "description": "Crowdsourced multi-camera waste dataset",
    },
    "mjuwaste": {
        "name": "MJU-Waste",
        "url": "https://github.com/realwecan/mju-waste",
        "num_classes": 1,
        "classes": ["waste"],
        "camera_type": "rgbd_kinect",
        "resolution": (640, 480),
        "annotation_format": "COCO",
        "description": "RGBD waste segmentation - Sensors 2020",
    },
}

# ─── FL Client Assignment ────────────────────────────────────────────────────

CLIENT_DATASET_MAP = {
    "client_0": "zerowaste",       # Nhà máy tái chế nhựa - Industrial RGB
    "client_1": "spectralwaste",   # Nhà máy phân loại quang học - RGB+HSI
    "client_2": "taco",            # Nhà máy đô thị - Mobile cameras
    "client_3": "mjuwaste",        # Nhà máy thí nghiệm - RGBD
}

CLIENT_MODEL_MAP = {
    "client_0": "resnet50",        # Server GPU - mô hình lớn
    "client_1": "efficientnet_b3", # Workstation - mô hình trung bình
    "client_2": "yolov8n",         # Edge NVIDIA Jetson - mô hình nhẹ
    "client_3": "mobilenetv3",     # Raspberry Pi - mô hình tối giản
}

# ─── FedKWAZ Hyperparameters ─────────────────────────────────────────────────

@dataclass
class FedKWAZConfig:
    # === FL Training ===
    num_rounds: int = 50
    num_clients: int = 4
    clients_per_round: int = 4           # Tất cả clients tham gia mỗi round
    local_epochs: int = 3
    local_batch_size: int = 16
    local_lr: float = 1e-3
    local_lr_decay: float = 0.99

    # === KWAZ Core Parameters ===
    kwaz_top_k: int = 10                 # Số lượng top-k weak-aware patches
    kwaz_semantic_threshold: float = 0.3 # Ngưỡng phát hiện Semantic WAZ
    kwaz_decision_threshold: float = 0.4 # Ngưỡng phát hiện Decision WAZ
    kwaz_camera_threshold: float = 0.35  # Ngưỡng Camera WAZ (mở rộng mới)
    feature_dim: int = 256               # Chiều không gian biểu diễn

    # === HAPM Parameters (Hierarchical Adaptive Patch Mixing) ===
    hapm_num_scales: int = 3             # Số cấp độ phân cấp (patch scales)
    hapm_patch_sizes: List[int] = field(default_factory=lambda: [32, 64, 128])
    hapm_mix_alpha: float = 0.5          # Tham số CutMix alpha
    hapm_num_samples: int = 5            # Số mẫu mixing mỗi batch

    # === KDP Parameters (Knowledge Discrepancy Perceptron) ===
    kdp_hidden_dim: int = 128
    kdp_temperature: float = 0.07        # Temperature cho contrastive loss

    # === Distillation Parameters ===
    distill_temperature: float = 4.0
    distill_alpha: float = 0.5           # Cân bằng CE loss vs KD loss
    distill_beta: float = 0.3            # Trọng số KWAZ-guided refinement

    # === Proxy Model ===
    proxy_model: str = "resnet18"        # Proxy model nhỏ trên server
    proxy_dataset: str = "zerowaste"     # Dataset proxy (có thể public)
    proxy_lr: float = 5e-4

    # === Aggregation ===
    aggregation: str = "fedavg"          # fedavg | fedprox | scaffold
    fedprox_mu: float = 0.01

    # === Model Compression (Bandwidth constraint) ===
    enable_compression: bool = True
    compression_ratio: float = 0.5      # Nén 50% gradient/KWAZ representation
    quantization_bits: int = 8

    # === Privacy ===
    enable_dp: bool = False              # Differential Privacy
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0

    # === Evaluation ===
    eval_every: int = 5                  # Đánh giá sau mỗi N rounds
    save_best: bool = True
    metrics: List[str] = field(
        default_factory=lambda: ["mAP50", "mAP75", "mAP50_95", "per_class_AP"]
    )


@dataclass
class TrainingConfig:
    # === Paths ===
    data_root: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # === Model Training ===
    img_size: int = 640
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"                 # cuda | cpu | mps

    # === Augmentation (per client) ===
    augment_rgb: bool = True
    augment_hsi: bool = True             # Chỉ áp dụng cho SpectralWaste
    color_jitter: float = 0.4
    random_flip: float = 0.5
    random_scale: List[float] = field(default_factory=lambda: [0.5, 1.5])

    # === Logging ===
    use_wandb: bool = False
    project_name: str = "FedKWAZ-WasteManagement"
    experiment_name: str = "baseline"


# ─── Singleton Configs ───────────────────────────────────────────────────────

def get_fedkwaz_config(**kwargs) -> FedKWAZConfig:
    cfg = FedKWAZConfig()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def get_training_config(**kwargs) -> TrainingConfig:
    cfg = TrainingConfig()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
