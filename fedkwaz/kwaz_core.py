"""
FedKWAZ Core Algorithm
======================
Knowledge Weak-Aware Zones (KWAZ) + Hierarchical Adaptive Patch Mixing (HAPM)
+ Knowledge Discrepancy Perceptron (KDP)
Dựa trên paper NeurIPS 2025 với mở rộng Camera WAZ cho domain rác thải công nghiệp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import math


# ─── KWAZ Detector ─────────────────────────────────────────────────────────

class KWAZDetector(nn.Module):
    """
    Knowledge Weak-Aware Zone Detector
    Xác định 3 loại vùng sai lệch tri thức:
    1. Semantic WAZ: Sai lệch trong representation space (feature embedding)
    2. Decision WAZ: Sai lệch trong decision space (class probability)
    3. Camera WAZ: Sai lệch do camera heterogeneity (NOVEL - đóng góp của đề tài)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        semantic_threshold: float = 0.3,
        decision_threshold: float = 0.4,
        camera_threshold: float = 0.35,
        top_k: int = 10,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.semantic_threshold = semantic_threshold
        self.decision_threshold = decision_threshold
        self.camera_threshold = camera_threshold
        self.top_k = top_k

        # Semantic discrepancy estimator
        self.semantic_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

        # Decision discrepancy estimator
        self.decision_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

        # Camera discrepancy estimator (NOVEL: sensor-level heterogeneity)
        self.camera_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2 + 4, feature_dim),  # +4 for camera metadata
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

    def compute_semantic_kwaz(
        self,
        private_feat: torch.Tensor,   # (B, D)
        proxy_feat: torch.Tensor,     # (B, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính Semantic WAZ: Vùng sai lệch trong không gian biểu diễn
        Returns: (scores, mask) - scores là mức độ sai lệch, mask là vùng weak
        """
        # Cosine similarity trong representation space
        private_norm = F.normalize(private_feat, dim=-1)
        proxy_norm = F.normalize(proxy_feat, dim=-1)
        cos_sim = (private_norm * proxy_norm).sum(dim=-1)  # (B,)

        # L2 discrepancy
        l2_disc = F.mse_loss(private_feat, proxy_feat, reduction="none").mean(dim=-1)  # (B,)

        # Combined score qua estimator
        concat = torch.cat([private_feat, proxy_feat], dim=-1)
        estimator_score = self.semantic_estimator(concat).squeeze(-1)  # (B,)

        # Semantic WAZ score = weighted combination
        semantic_score = 0.4 * (1 - cos_sim) + 0.3 * l2_disc.tanh() + 0.3 * estimator_score

        mask = semantic_score > self.semantic_threshold
        return semantic_score, mask

    def compute_decision_kwaz(
        self,
        private_logits: torch.Tensor,   # (B, C)
        proxy_logits: torch.Tensor,     # (B, C)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính Decision WAZ: Vùng sai lệch trong không gian quyết định
        Đặc biệt quan trọng khi các nhà máy có class distribution khác nhau (Non-IID)
        """
        private_prob = F.softmax(private_logits, dim=-1)
        proxy_prob = F.softmax(proxy_logits, dim=-1)

        # KL Divergence P || Q
        kl_div = F.kl_div(
            F.log_softmax(private_logits, dim=-1),
            proxy_prob,
            reduction="none"
        ).sum(dim=-1)  # (B,)

        # JS Divergence (symmetric)
        m = 0.5 * (private_prob + proxy_prob)
        js_div = 0.5 * (
            F.kl_div(F.log_softmax(private_logits, dim=-1), m, reduction="none").sum(-1) +
            F.kl_div(torch.log(proxy_prob + 1e-8), m, reduction="none").sum(-1)
        )

        # Prediction agreement
        private_pred = private_logits.argmax(dim=-1)
        proxy_pred = proxy_logits.argmax(dim=-1)
        disagree = (private_pred != proxy_pred).float()

        decision_score = 0.4 * kl_div.tanh() + 0.4 * js_div + 0.2 * disagree
        mask = decision_score > self.decision_threshold
        return decision_score, mask

    def compute_camera_kwaz(
        self,
        private_feat: torch.Tensor,    # (B, D)
        proxy_feat: torch.Tensor,      # (B, D)
        camera_meta: Dict,             # Camera metadata {resolution, modality, noise_level, ...}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính Camera WAZ: Sai lệch do camera heterogeneity
        NOVEL CONTRIBUTION: Mở rộng FedKWAZ gốc cho sensor-level heterogeneity

        Camera metadata encoding:
        - modality: 0=RGB, 1=RGBD, 2=RGB+HSI, 3=Thermal
        - resolution_ratio: tỉ lệ độ phân giải so với chuẩn 1080p
        - noise_level: ước tính noise (từ camera specs)
        - spectral_range: phạm vi quang phổ (0=visible, 1=NIR, 2=SWIR)
        """
        # Encode camera metadata thành vector
        B = private_feat.size(0)
        device = private_feat.device

        modality = torch.full((B, 1), camera_meta.get("modality_id", 0), device=device).float()
        res_ratio = torch.full((B, 1), camera_meta.get("resolution_ratio", 1.0), device=device).float()
        noise = torch.full((B, 1), camera_meta.get("noise_level", 0.1), device=device).float()
        spectral = torch.full((B, 1), camera_meta.get("spectral_id", 0), device=device).float()

        camera_vec = torch.cat([modality, res_ratio, noise, spectral], dim=-1)  # (B, 4)

        concat = torch.cat([private_feat, proxy_feat, camera_vec], dim=-1)
        camera_score = self.camera_estimator(concat).squeeze(-1)  # (B,)

        mask = camera_score > self.camera_threshold
        return camera_score, mask

    def forward(
        self,
        private_output: Dict,
        proxy_output: Dict,
        camera_meta: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tổng hợp tất cả KWAZ scores
        Returns: dictionary với scores và masks cho từng loại KWAZ
        """
        private_feat = private_output["global_feat"]
        proxy_feat = proxy_output["global_feat"]
        private_logits = private_output["logits"]
        proxy_logits = proxy_output["logits"]

        # Align dimensions nếu cần (different num_classes)
        min_classes = min(private_logits.size(-1), proxy_logits.size(-1))
        private_logits_aligned = private_logits[:, :min_classes]
        proxy_logits_aligned = proxy_logits[:, :min_classes]

        # Align feature dimensions
        if private_feat.size(-1) != proxy_feat.size(-1):
            min_dim = min(private_feat.size(-1), proxy_feat.size(-1))
            private_feat = private_feat[:, :min_dim]
            proxy_feat = proxy_feat[:, :min_dim]

        sem_score, sem_mask = self.compute_semantic_kwaz(private_feat, proxy_feat)
        dec_score, dec_mask = self.compute_decision_kwaz(
            private_logits_aligned, proxy_logits_aligned
        )

        result = {
            "semantic_score": sem_score,
            "semantic_mask": sem_mask,
            "decision_score": dec_score,
            "decision_mask": dec_mask,
        }

        if camera_meta is not None:
            cam_score, cam_mask = self.compute_camera_kwaz(private_feat, proxy_feat, camera_meta)
            result["camera_score"] = cam_score
            result["camera_mask"] = cam_mask
            # Combined KWAZ mask
            result["kwaz_mask"] = sem_mask | dec_mask | cam_mask
            result["kwaz_score"] = (sem_score + dec_score + cam_score) / 3
        else:
            result["kwaz_mask"] = sem_mask | dec_mask
            result["kwaz_score"] = (sem_score + dec_score) / 2

        return result


# ─── HAPM: Hierarchical Adaptive Patch Mixing ─────────────────────────────────

class HierarchicalAdaptivePatchMixing(nn.Module):
    """
    HAPM: Tạo các mẫu trộn phân cấp để khai thác KWAZ
    3 cấp độ: fine (32x32), medium (64x64), coarse (128x128)
    Áp dụng CutMix với adaptive ratio dựa trên KWAZ scores
    """

    def __init__(
        self,
        patch_sizes: List[int] = [32, 64, 128],
        num_samples: int = 5,
        mix_alpha: float = 0.5,
    ):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.num_samples = num_samples
        self.mix_alpha = mix_alpha

    def _cutmix_patches(
        self,
        img_a: torch.Tensor,   # (B, C, H, W)
        img_b: torch.Tensor,   # (B, C, H, W)
        patch_size: int,
        kwaz_scores: Optional[torch.Tensor] = None,  # (B,) - high score = more mixing
    ) -> torch.Tensor:
        """CutMix at specific patch size với adaptive mixing ratio"""
        B, C, H, W = img_a.shape
        mixed = img_a.clone()

        for b in range(B):
            # Adaptive lambda based on KWAZ score
            if kwaz_scores is not None:
                score = kwaz_scores[b].item()
                lam = np.random.beta(
                    self.mix_alpha * (1 + score),
                    self.mix_alpha
                )
            else:
                lam = np.random.beta(self.mix_alpha, self.mix_alpha)

            # Random patch location
            n_patches_h = H // patch_size
            n_patches_w = W // patch_size
            if n_patches_h < 1 or n_patches_w < 1:
                continue

            # Select patch region for mixing
            ph = np.random.randint(0, n_patches_h)
            pw = np.random.randint(0, n_patches_w)
            n_mix_h = max(1, int(n_patches_h * lam))
            n_mix_w = max(1, int(n_patches_w * lam))

            y1 = ph * patch_size
            y2 = min(H, (ph + n_mix_h) * patch_size)
            x1 = pw * patch_size
            x2 = min(W, (pw + n_mix_w) * patch_size)

            mixed[b, :, y1:y2, x1:x2] = img_b[b, :, y1:y2, x1:x2]

        return mixed

    def forward(
        self,
        private_imgs: torch.Tensor,    # (B, C, H, W)
        proxy_imgs: torch.Tensor,      # (B, C, H, W)
        kwaz_scores: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Tạo num_samples * num_scales mixed samples
        Returns: list of mixed tensors, mỗi cấp độ num_samples samples
        """
        all_mixed = []

        for patch_size in self.patch_sizes:
            scale_mixed = []
            for _ in range(self.num_samples):
                mixed = self._cutmix_patches(
                    private_imgs, proxy_imgs, patch_size, kwaz_scores
                )
                scale_mixed.append(mixed)
            all_mixed.append(torch.stack(scale_mixed))  # (num_samples, B, C, H, W)

        return all_mixed  # list of [num_samples, B, C, H, W] for each scale


# ─── KDP: Knowledge Discrepancy Perceptron ────────────────────────────────────

class KnowledgeDiscrepancyPerceptron(nn.Module):
    """
    KDP: Chọn các mẫu có sự sai lệch tri thức lớn nhất để tập trung học
    Dùng contrastive learning để phân biệt samples với high/low discrepancy
    """

    def __init__(self, feature_dim: int = 256, temperature: float = 0.07, hidden_dim: int = 128):
        super().__init__()
        self.temperature = temperature

        # Score network: đánh giá mức độ discrepancy của một sample
        self.score_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Feature alignment network
        self.align_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def compute_discrepancy_scores(
        self,
        private_feats: List[torch.Tensor],  # list of (B, D) for each mixed sample
        proxy_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """Tính discrepancy score cho mỗi mixed sample"""
        scores = []
        for pf, qf in zip(private_feats, proxy_feats):
            concat = torch.cat([pf, qf], dim=-1)  # (B, 2D)
            score = self.score_net(concat)          # (B, 1)
            scores.append(score)
        scores = torch.cat(scores, dim=-1)  # (B, num_samples)
        return scores  # (B, num_samples)

    def select_top_k_samples(
        self,
        all_mixed_imgs: List[torch.Tensor],   # list of (num_samples, B, C, H, W)
        private_model_fn,                      # callable: img -> output
        proxy_model_fn,                        # callable: img -> output
        k: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chọn k samples có discrepancy lớn nhất từ tất cả các scales
        Returns: (selected_imgs (k*B, C, H, W), selected_scores (k*B,))
        """
        all_scores = []
        all_imgs_flat = []

        for scale_imgs in all_mixed_imgs:   # (num_samples, B, C, H, W)
            for s_idx in range(scale_imgs.size(0)):
                imgs = scale_imgs[s_idx]    # (B, C, H, W)
                with torch.no_grad():
                    p_out = private_model_fn(imgs)
                    q_out = proxy_model_fn(imgs)

                pf = p_out["global_feat"]
                qf = q_out["global_feat"]
                if pf.size(-1) != qf.size(-1):
                    min_d = min(pf.size(-1), qf.size(-1))
                    pf, qf = pf[:, :min_d], qf[:, :min_d]

                concat = torch.cat([pf, qf], dim=-1)
                score = self.score_net(concat).squeeze(-1)  # (B,)
                all_scores.append(score)
                all_imgs_flat.append(imgs)

        # Stack and select top-k
        all_scores_tensor = torch.stack(all_scores, dim=1)   # (B, total_samples)
        _, top_k_idx = all_scores_tensor.topk(k, dim=1)      # (B, k)

        B, C, H, W = all_imgs_flat[0].shape
        selected_imgs = []
        selected_scores = []
        for b in range(B):
            for ki in range(k):
                sample_idx = top_k_idx[b, ki].item()
                selected_imgs.append(all_imgs_flat[sample_idx][b])
                selected_scores.append(all_scores[sample_idx][b])

        return (
            torch.stack(selected_imgs),    # (k*B, C, H, W)
            torch.stack(selected_scores),  # (k*B,)
        )

    def forward(
        self,
        private_feat: torch.Tensor,
        proxy_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Tính KDP alignment loss"""
        if private_feat.size(-1) != proxy_feat.size(-1):
            min_d = min(private_feat.size(-1), proxy_feat.size(-1))
            private_feat = private_feat[:, :min_d]
            proxy_feat = proxy_feat[:, :min_d]

        p_aligned = self.align_net(private_feat)
        q_aligned = self.align_net(proxy_feat)

        # NT-Xent contrastive loss
        p_norm = F.normalize(p_aligned, dim=-1)
        q_norm = F.normalize(q_aligned, dim=-1)

        logits = torch.mm(p_norm, q_norm.T) / self.temperature  # (B, B)
        labels = torch.arange(p_norm.size(0), device=p_norm.device)
        loss = F.cross_entropy(logits, labels)

        # Discrepancy score
        concat = torch.cat([private_feat, proxy_feat], dim=-1)
        disc_score = self.score_net(concat).squeeze(-1)

        return {
            "kdp_loss": loss,
            "disc_score": disc_score,
            "p_aligned": p_aligned,
            "q_aligned": q_aligned,
        }


# ─── FedKWAZ Loss ─────────────────────────────────────────────────────────────

class FedKWAZLoss(nn.Module):
    """
    Tổng hợp tất cả losses trong FedKWAZ:
    1. Task loss (CE/Focal)
    2. Global alignment loss (Stage 1)
    3. KWAZ-guided refinement loss (Stage 2)
    4. KDP contrastive loss
    5. Camera WAZ loss (Novel)
    """

    def __init__(
        self,
        distill_temperature: float = 4.0,
        distill_alpha: float = 0.5,
        distill_beta: float = 0.3,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.T = distill_temperature
        self.alpha = distill_alpha
        self.beta = distill_beta
        self.use_focal = use_focal
        self.gamma = focal_gamma

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss cho class imbalance trong waste detection"""
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()

    def kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Knowledge Distillation loss với temperature scaling"""
        min_classes = min(student_logits.size(-1), teacher_logits.size(-1))
        student_logits = student_logits[:, :min_classes]
        teacher_logits = teacher_logits[:, :min_classes]

        return F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction="batchmean",
        ) * (self.T ** 2)

    def kwaz_guided_loss(
        self,
        private_output: Dict,
        proxy_output: Dict,
        kwaz_scores: torch.Tensor,    # (B,) - sử dụng như attention weight
    ) -> torch.Tensor:
        """
        KWAZ-guided refinement: Tập trung distillation vào vùng weak
        Sample với KWAZ score cao sẽ nhận loss weight lớn hơn
        """
        private_logits = private_output["logits"]
        proxy_logits = proxy_output["logits"]
        min_classes = min(private_logits.size(-1), proxy_logits.size(-1))

        kl_per_sample = F.kl_div(
            F.log_softmax(private_logits[:, :min_classes] / self.T, dim=-1),
            F.softmax(proxy_logits[:, :min_classes] / self.T, dim=-1),
            reduction="none",
        ).sum(dim=-1) * (self.T ** 2)

        # Weight by KWAZ scores (high KWAZ = focus more)
        weights = F.softmax(kwaz_scores.detach(), dim=0) * kwaz_scores.size(0)
        return (weights * kl_per_sample).mean()

    def representation_alignment_loss(
        self,
        private_feat: torch.Tensor,
        proxy_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Stage 1: Global class-level representation alignment"""
        if private_feat.size(-1) != proxy_feat.size(-1):
            min_d = min(private_feat.size(-1), proxy_feat.size(-1))
            private_feat = private_feat[:, :min_d]
            proxy_feat = proxy_feat[:, :min_d]

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(private_feat, proxy_feat, dim=-1)
        return (1 - cos_sim).mean()

    def forward(
        self,
        private_output: Dict,
        proxy_output: Dict,
        targets: torch.Tensor,
        kwaz_result: Dict,
        kdp_result: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Tính tổng hợp tất cả losses"""
        # 1. Task loss
        if self.use_focal:
            task_loss = self.focal_loss(private_output["logits"], targets)
        else:
            task_loss = F.cross_entropy(private_output["logits"], targets)

        # 2. Global alignment (Stage 1)
        repr_loss = self.representation_alignment_loss(
            private_output["global_feat"], proxy_output["global_feat"]
        )
        global_kd_loss = self.kd_loss(private_output["logits"], proxy_output["logits"])

        # 3. KWAZ-guided refinement (Stage 2)
        kwaz_score = kwaz_result.get("kwaz_score", torch.zeros(targets.size(0), device=targets.device))
        kwaz_loss = self.kwaz_guided_loss(private_output, proxy_output, kwaz_score)

        # 4. KDP loss
        kdp_loss = kdp_result.get("kdp_loss", torch.tensor(0.0, device=targets.device))

        # 5. Camera WAZ loss (Novel)
        camera_loss = torch.tensor(0.0, device=targets.device)
        if "camera_score" in kwaz_result:
            camera_loss = kwaz_result["camera_score"].mean() * 0.1

        # Total loss
        total_loss = (
            task_loss
            + self.alpha * (repr_loss + global_kd_loss)
            + self.beta * (kwaz_loss + kdp_loss)
            + 0.1 * camera_loss
        )

        return {
            "total": total_loss,
            "task": task_loss.detach(),
            "repr": repr_loss.detach(),
            "global_kd": global_kd_loss.detach(),
            "kwaz": kwaz_loss.detach(),
            "kdp": kdp_loss.detach(),
            "camera": camera_loss.detach(),
        }
