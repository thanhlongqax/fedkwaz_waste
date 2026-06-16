"""
Kiến trúc Mô hình Dị thể cho FL Clients
=========================================
ResNet50 | EfficientNet-B3 | MobileNetV3 | Lightweight Detector
Mỗi model đại diện cho phần cứng khác nhau tại mỗi nhà máy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


# ─── Feature Projector (Dùng chung cho KWAZ alignment) ────────────────────────

class FeatureProjector(nn.Module):
    """
    Project feature maps từ các backbone khác nhau về cùng feature_dim
    Đây là 'cầu nối' cho KWAZ alignment giữa các mô hình dị thể
    """
    def __init__(self, in_dim: int, feature_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class SpatialFeatureProjector(nn.Module):
    """
    Spatial projector giữ nguyên spatial dimension (cho segmentation)
    Dùng để tính Spatial KWAZ (pixel/patch level)
    """
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─── Client Model Base ────────────────────────────────────────────────────────

class WasteClientModel(nn.Module):
    """Base class cho tất cả client models trong FL"""

    def __init__(self, num_classes: int, feature_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.backbone = None
        self.neck = None
        self.head = None
        self.projector = None         # Global feature projector
        self.spatial_projector = None # Spatial feature projector

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Trích xuất features ở nhiều cấp độ cho KWAZ computation"""
        raise NotImplementedError

    @property
    def model_size(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Client 0: ResNet50 (Server GPU - ZeroWaste) ──────────────────────────────

class ResNet50WasteDetector(WasteClientModel):
    """
    Mô hình lớn cho Client 0 (Nhà máy tái chế nhựa)
    Hardware: NVIDIA RTX 3080, ZeroWaste dataset (4 classes)
    """

    def __init__(self, num_classes: int = 4, feature_dim: int = 256, pretrained: bool = True):
        super().__init__(num_classes, feature_dim)

        # Import torchvision backbone
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        except ImportError:
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=pretrained)

        # Lấy feature layers
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256 channels
        self.layer2 = backbone.layer2   # 512 channels
        self.layer3 = backbone.layer3   # 1024 channels
        self.layer4 = backbone.layer4   # 2048 channels

        # FPN neck (Feature Pyramid Network lite)
        self.fpn_lat3 = nn.Conv2d(1024, 256, 1)
        self.fpn_lat4 = nn.Conv2d(2048, 256, 1)
        self.fpn_out = nn.Conv2d(256, 256, 3, padding=1)

        # Detection head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes + 1),
        )

        # KWAZ projectors
        self.projector = FeatureProjector(2048, feature_dim)
        self.spatial_projector = SpatialFeatureProjector(256, feature_dim)

    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.layer0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {"low": f2, "mid": f3, "high": f4}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.get_features(x)
        f3, f4 = feats["mid"], feats["high"]

        # FPN
        p4 = self.fpn_lat4(f4)
        p3 = self.fpn_lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="nearest")
        p_out = self.fpn_out(p3)

        # Predictions
        logits = self.cls_head(p_out)
        global_feat = self.projector(feats["high"])
        spatial_feat = self.spatial_projector(p_out)

        return {
            "logits": logits,
            "global_feat": global_feat,
            "spatial_feat": spatial_feat,
            "raw_feat": feats["high"],
        }


# ─── Client 1: EfficientNet-B3 (Workstation - SpectralWaste) ─────────────────

class EfficientNetWasteDetector(WasteClientModel):
    """
    Mô hình trung bình cho Client 1 (Nhà máy phân loại quang học)
    Hardware: Workstation RTX 3060, SpectralWaste (6 classes, RGB+HSI)
    Extension: Thêm HSI fusion module
    """

    def __init__(self, num_classes: int = 6, feature_dim: int = 256,
                 use_hsi: bool = True, hsi_channels: int = 224, pretrained: bool = True):
        super().__init__(num_classes, feature_dim)
        self.use_hsi = use_hsi

        # RGB backbone
        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            eff = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            from torchvision.models import efficientnet_b3
            eff = efficientnet_b3(pretrained=pretrained)

        self.rgb_features = eff.features     # Output: 1536 channels
        self.rgb_avgpool = eff.avgpool

        # HSI branch: dimensionality reduction trước khi fusion
        if use_hsi:
            self.hsi_encoder = nn.Sequential(
                nn.Conv2d(hsi_channels, 64, 1, bias=False),   # Band reduction
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            fusion_dim = 1536 + 256
        else:
            fusion_dim = 1536

        # Fusion & head
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.cls_head = nn.Linear(1024, num_classes + 1)

        # KWAZ projectors
        self.projector = FeatureProjector(1536, feature_dim)
        self.spatial_projector = nn.Sequential(
            nn.Conv2d(1536, 256, 1),
            nn.BatchNorm2d(256), nn.ReLU(),
            SpatialFeatureProjector(256, feature_dim),
        )

    def get_features(self, x: torch.Tensor, hsi: Optional[torch.Tensor] = None) -> Dict:
        rgb_feat_map = self.rgb_features(x)
        rgb_feat = self.rgb_avgpool(rgb_feat_map).flatten(1)
        result = {"rgb_feat_map": rgb_feat_map, "rgb_feat": rgb_feat}

        if self.use_hsi and hsi is not None:
            hsi_feat = self.hsi_encoder(hsi)
            result["hsi_feat"] = hsi_feat
        return result

    def forward(self, x: torch.Tensor, hsi: Optional[torch.Tensor] = None) -> Dict:
        feats = self.get_features(x, hsi)
        rgb_feat = feats["rgb_feat"]

        if self.use_hsi and "hsi_feat" in feats:
            fused = torch.cat([rgb_feat, feats["hsi_feat"]], dim=1)
        else:
            fused = rgb_feat
            if self.use_hsi:  # padding nếu không có HSI
                pad = torch.zeros(rgb_feat.size(0), 256, device=rgb_feat.device)
                fused = torch.cat([rgb_feat, pad], dim=1)

        fused = self.fusion(fused)
        logits = self.cls_head(fused)

        global_feat = self.projector(feats["rgb_feat_map"])
        spatial_feat_map = feats["rgb_feat_map"]
        spatial_feat = nn.functional.adaptive_avg_pool2d(spatial_feat_map, 16)
        B, C, H, W = spatial_feat.shape
        spatial_feat = spatial_feat.view(B, -1)

        return {
            "logits": logits,
            "global_feat": global_feat,
            "spatial_feat": spatial_feat,
            "raw_feat": feats["rgb_feat_map"],
        }


# ─── Client 2: YOLOv8-inspired Nano (Jetson Edge - TACO) ─────────────────────

class NanoWasteDetector(WasteClientModel):
    """
    Mô hình nhẹ cho Client 2 (Edge NVIDIA Jetson)
    Hardware: Jetson Orin Nano, TACO dataset (nhiều classes)
    Design: Inspired by YOLOv8n - tối ưu cho real-time inference
    """

    def __init__(self, num_classes: int = 28, feature_dim: int = 256):
        super().__init__(num_classes, feature_dim)

        def conv_bn(in_c, out_c, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, s, p, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(inplace=True),
            )

        def c2f_block(in_c, out_c, n=1):
            """C2f block (simplified Cross Stage Partial with 2 convs)"""
            hidden = out_c // 2
            layers = [conv_bn(in_c, out_c, 1, 1, 0)]
            for _ in range(n):
                layers.append(nn.Sequential(
                    conv_bn(out_c, hidden, 3),
                    conv_bn(hidden, out_c, 3),
                ))
            return nn.ModuleList(layers)

        # Backbone (CSP-inspired)
        self.stem = conv_bn(3, 32, 3, 2)
        self.stage1 = nn.Sequential(conv_bn(32, 64, 3, 2), conv_bn(64, 64, 3))
        self.stage2 = nn.Sequential(conv_bn(64, 128, 3, 2), conv_bn(128, 128, 3), conv_bn(128, 128, 3))
        self.stage3 = nn.Sequential(conv_bn(128, 256, 3, 2), conv_bn(256, 256, 3), conv_bn(256, 256, 3))
        self.stage4 = nn.Sequential(conv_bn(256, 256, 3, 2), conv_bn(256, 256, 3))

        # Neck (lightweight FPN)
        self.neck_up = nn.Sequential(conv_bn(256, 128, 1, 1, 0), nn.Upsample(scale_factor=2))
        self.neck_fuse = conv_bn(384, 256, 3)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, num_classes + 1),
        )

        # KWAZ projectors
        self.projector = FeatureProjector(256, feature_dim)
        self.spatial_projector = SpatialFeatureProjector(256, feature_dim)

    def get_features(self, x: torch.Tensor) -> Dict:
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return {"low": s2, "mid": s3, "high": s4}

    def forward(self, x: torch.Tensor) -> Dict:
        feats = self.get_features(x)
        s3, s4 = feats["mid"], feats["high"]

        neck_out = self.neck_up(s4)
        if neck_out.shape[-2:] != s3.shape[-2:]:
            neck_out = F.interpolate(neck_out, size=s3.shape[-2:], mode="nearest")
        fused = self.neck_fuse(torch.cat([neck_out, s3], dim=1))

        logits = self.head(fused)
        global_feat = self.projector(feats["high"])
        spatial_feat = self.spatial_projector(fused)

        return {
            "logits": logits,
            "global_feat": global_feat,
            "spatial_feat": F.adaptive_avg_pool2d(spatial_feat, 1).flatten(1),
            "raw_feat": feats["high"],
        }


# ─── Client 3: MobileNetV3 (Raspberry Pi - MJU-Waste) ────────────────────────

class MobileNetV3WasteDetector(WasteClientModel):
    """
    Mô hình tối giản cho Client 3 (Raspberry Pi IoT)
    Hardware: Raspberry Pi 4, MJU-Waste RGBD
    Extension: Depth fusion cho RGBD camera
    """

    def __init__(self, num_classes: int = 1, feature_dim: int = 256,
                 use_depth: bool = True, pretrained: bool = True):
        super().__init__(num_classes, feature_dim)
        self.use_depth = use_depth

        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            mob = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            from torchvision.models import mobilenet_v3_small
            mob = mobilenet_v3_small(pretrained=pretrained)

        self.features = mob.features        # Output: 576 channels
        self.avgpool = mob.avgpool

        # Depth encoder
        if use_depth:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Hardswish(),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.Hardswish(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            fusion_input = 576 + 32
        else:
            fusion_input = 576

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes + 1),
        )

        self.projector = FeatureProjector(576, feature_dim)
        self.spatial_projector = SpatialFeatureProjector(576, feature_dim)

    def get_features(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None) -> Dict:
        feat_map = self.features(x)
        feat = self.avgpool(feat_map).flatten(1)
        result = {"feat_map": feat_map, "feat": feat}
        if self.use_depth and depth is not None:
            result["depth_feat"] = self.depth_encoder(depth)
        return result

    def forward(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None) -> Dict:
        feats = self.get_features(x, depth)
        feat = feats["feat"]

        if self.use_depth and "depth_feat" in feats:
            fused = torch.cat([feat, feats["depth_feat"]], dim=1)
        elif self.use_depth:
            pad = torch.zeros(feat.size(0), 32, device=feat.device)
            fused = torch.cat([feat, pad], dim=1)
        else:
            fused = feat

        logits = self.classifier(fused)
        global_feat = self.projector(feats["feat_map"])
        spatial_feat = self.spatial_projector(feats["feat_map"])

        return {
            "logits": logits,
            "global_feat": global_feat,
            "spatial_feat": F.adaptive_avg_pool2d(spatial_feat, 1).flatten(1),
            "raw_feat": feats["feat_map"],
        }


# ─── Proxy Model (Server-side small model) ───────────────────────────────────

class ProxyModel(WasteClientModel):
    """
    Mô hình proxy nhỏ trên FL Server
    Dùng để bridge knowledge exchange giữa các client dị thể
    Được train trên public proxy dataset
    """

    def __init__(self, num_classes: int = 10, feature_dim: int = 256, pretrained: bool = True):
        super().__init__(num_classes, feature_dim)

        try:
            from torchvision.models import resnet18, ResNet18_Weights
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            from torchvision.models import resnet18
            resnet = resnet18(pretrained=pretrained)

        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.avgpool = resnet.avgpool

        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes + 1),
        )
        self.projector = FeatureProjector(512, feature_dim)
        self.spatial_projector = SpatialFeatureProjector(512, feature_dim)

    def get_features(self, x: torch.Tensor) -> Dict:
        feat_map = self.encoder(x)
        return {"feat_map": feat_map, "feat": self.avgpool(feat_map).flatten(1)}

    def forward(self, x: torch.Tensor) -> Dict:
        feats = self.get_features(x)
        logits = self.cls_head(feats["feat"])
        global_feat = self.projector(feats["feat_map"])
        spatial_feat = self.spatial_projector(feats["feat_map"])

        return {
            "logits": logits,
            "global_feat": global_feat,
            "spatial_feat": F.adaptive_avg_pool2d(spatial_feat, 1).flatten(1),
            "raw_feat": feats["feat_map"],
        }


# ─── Model Factory ───────────────────────────────────────────────────────────

def build_client_model(
    model_name: str,
    num_classes: int,
    feature_dim: int = 256,
    **kwargs
) -> WasteClientModel:
    model_map = {
        "resnet50": ResNet50WasteDetector,
        "efficientnet_b3": EfficientNetWasteDetector,
        "yolov8n": NanoWasteDetector,
        "mobilenetv3": MobileNetV3WasteDetector,
        "proxy": ProxyModel,
    }
    if model_name not in model_map:
        raise ValueError(f"Model '{model_name}' không hợp lệ. Chọn: {list(model_map.keys())}")
    return model_map[model_name](num_classes=num_classes, feature_dim=feature_dim, **kwargs)


def count_parameters(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M"
