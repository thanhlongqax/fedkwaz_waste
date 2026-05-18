"""
Dataset Loaders cho 4 nguồn dữ liệu rác thải công nghiệp
=========================================================
ZeroWaste | SpectralWaste | TACO | MJU-Waste
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─── Base Dataset ────────────────────────────────────────────────────────────

class WasteBaseDataset(Dataset):
    """Abstract base class cho tất cả waste datasets"""

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        transform: Optional[Callable] = None,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.augment = augment and split == "train"

        self.images: List[Path] = []
        self.annotations: List[Dict] = []
        self.class_names: List[str] = []

        self._load_data()

    def _load_data(self):
        raise NotImplementedError

    def _default_transforms(self) -> Callable:
        transforms = [T.Resize((self.img_size, self.img_size))]
        if self.augment:
            transforms += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                T.RandomRotation(degrees=15),
            ]
        transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        return T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError

    def get_client_info(self) -> Dict:
        """Trả về metadata về camera & dataset cho FL client"""
        raise NotImplementedError


# ─── ZeroWaste Dataset ───────────────────────────────────────────────────────

class ZeroWasteDataset(WasteBaseDataset):
    """
    ZeroWaste-f Dataset - CVPR 2022
    Camera: Industrial fixed RGB (conveyor belt)
    Classes: rigid_plastic, cardboard, metal, soft_plastic
    Download: https://zenodo.org/record/6412647
    """

    CLASS_NAMES = ["background", "rigid_plastic", "cardboard", "metal", "soft_plastic"]
    CAMERA_TYPE = "industrial_rgb_fixed"

    def __init__(self, root: str, split: str = "train", img_size: int = 640, **kwargs):
        self.class_names = self.CLASS_NAMES
        super().__init__(root, split, img_size, **kwargs)

    def _load_data(self):
        split_dir = self.root / self.split
        ann_file = self.root / f"{self.split}_annotations.json"

        if not split_dir.exists():
            raise FileNotFoundError(
                f"ZeroWaste dataset không tìm thấy tại {self.root}\n"
                f"Tải về tại: https://zenodo.org/record/6412647"
            )

        # Load COCO annotations
        if ann_file.exists():
            with open(ann_file, "r") as f:
                coco_data = json.load(f)
            self._parse_coco(coco_data, split_dir)
        else:
            # Fallback: scan images without annotations
            self.images = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.png"))
            self.annotations = [{"boxes": [], "labels": [], "masks": []}] * len(self.images)

    def _parse_coco(self, coco_data: Dict, img_dir: Path):
        img_id_to_file = {img["id"]: img_dir / img["file_name"] for img in coco_data["images"]}
        img_id_to_anns = {img["id"]: [] for img in coco_data["images"]}

        for ann in coco_data.get("annotations", []):
            img_id_to_anns[ann["image_id"]].append(ann)

        for img_id, img_path in img_id_to_file.items():
            if img_path.exists():
                self.images.append(img_path)
                anns = img_id_to_anns[img_id]
                self.annotations.append({
                    "boxes": [a["bbox"] for a in anns],        # [x, y, w, h]
                    "labels": [a["category_id"] for a in anns],
                    "masks": [a.get("segmentation", []) for a in anns],
                    "areas": [a.get("area", 0) for a in anns],
                })

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        ann = self.annotations[idx]

        # Resize
        w_orig, h_orig = image.size
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Scale bounding boxes
        scale_x = self.img_size / w_orig
        scale_y = self.img_size / h_orig
        boxes = []
        for box in ann["boxes"]:
            x, y, bw, bh = box
            boxes.append([x * scale_x, y * scale_y, (x + bw) * scale_x, (y + bh) * scale_y])

        if self.transform:
            image = self.transform(image)
        else:
            image = self._default_transforms()(image)

        return {
            "image": image,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(ann["labels"], dtype=torch.long),
            "image_path": str(img_path),
            "dataset": "zerowaste",
            "camera_type": self.CAMERA_TYPE,
        }

    def get_client_info(self) -> Dict:
        return {
            "dataset_name": "ZeroWaste-f",
            "camera_type": self.CAMERA_TYPE,
            "num_classes": len(self.CLASS_NAMES) - 1,
            "num_samples": len(self),
            "modality": "RGB",
            "resolution": "1920x1080",
            "hardware": "NVIDIA RTX 3080 (Server GPU)",
        }


# ─── SpectralWaste Dataset ────────────────────────────────────────────────────

class SpectralWasteDataset(WasteBaseDataset):
    """
    SpectralWaste Dataset - IROS 2024
    Camera: Line-scan RGB (Teledyne DALSA) + Hyperspectral NIR (Specim FX17, 224 bands)
    Classes: film, basket, video_tape, filaments, trash_bag, cardboard
    Download: https://zenodo.org/records/10880544 | OneDrive của tác giả
    """

    CLASS_NAMES = ["background", "film", "basket", "video_tape", "filaments", "trash_bag", "cardboard"]
    CAMERA_TYPE = "rgb_hyperspectral_linescan"
    HSI_CHANNELS = 224  # 900-1700nm, 224 bands

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        use_hsi: bool = True,       # True: RGB+HSI | False: RGB only
        hsi_bands: Optional[List[int]] = None,  # None = dùng tất cả bands
        **kwargs
    ):
        self.use_hsi = use_hsi
        self.hsi_bands = hsi_bands or list(range(self.HSI_CHANNELS))
        self.class_names = self.CLASS_NAMES
        super().__init__(root, split, img_size, **kwargs)

    def _load_data(self):
        split_map = {"train": "train", "val": "val", "test": "test"}
        split_dir = self.root / split_map.get(self.split, self.split)

        if not split_dir.exists():
            raise FileNotFoundError(
                f"SpectralWaste không tìm thấy tại {self.root}\n"
                f"Tải về tại: https://zenodo.org/records/10880544"
            )

        ann_file = split_dir / "annotations.json"
        rgb_dir = split_dir / "rgb"
        hsi_dir = split_dir / "hsi"

        if not rgb_dir.exists():
            rgb_dir = split_dir

        img_files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))

        if ann_file.exists():
            with open(ann_file) as f:
                coco_data = json.load(f)
            self._parse_coco_spectral(coco_data, rgb_dir, hsi_dir)
        else:
            for img_path in img_files:
                self.images.append(img_path)
                hsi_path = hsi_dir / (img_path.stem + ".npy") if hsi_dir.exists() else None
                self.annotations.append({
                    "boxes": [], "labels": [], "masks": [],
                    "hsi_path": str(hsi_path) if hsi_path and hsi_path.exists() else None,
                })

    def _parse_coco_spectral(self, coco_data: Dict, rgb_dir: Path, hsi_dir: Path):
        for img_info in coco_data["images"]:
            rgb_path = rgb_dir / img_info["file_name"]
            if not rgb_path.exists():
                continue

            hsi_path = hsi_dir / (Path(img_info["file_name"]).stem + ".npy")
            anns = [a for a in coco_data.get("annotations", [])
                    if a["image_id"] == img_info["id"]]

            self.images.append(rgb_path)
            self.annotations.append({
                "boxes": [a["bbox"] for a in anns],
                "labels": [a["category_id"] for a in anns],
                "masks": [a.get("segmentation", []) for a in anns],
                "hsi_path": str(hsi_path) if hsi_path.exists() else None,
            })

    def _load_hsi(self, hsi_path: Optional[str]) -> Optional[torch.Tensor]:
        """Load hyperspectral image (H, W, C) → (C, H, W) tensor"""
        if hsi_path is None or not Path(hsi_path).exists():
            return None
        try:
            hsi = np.load(hsi_path).astype(np.float32)  # (H, W, 224)
            hsi = hsi[:, :, self.hsi_bands]               # Band selection
            hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min() + 1e-8)
            hsi_tensor = torch.from_numpy(hsi).permute(2, 0, 1)  # (C, H, W)
            return TF.resize(hsi_tensor, [self.img_size, self.img_size])
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.images[idx]
        ann = self.annotations[idx]

        image = Image.open(img_path).convert("RGB")
        w_orig, h_orig = image.size
        image = image.resize((self.img_size, self.img_size))

        scale_x, scale_y = self.img_size / w_orig, self.img_size / h_orig
        boxes = []
        for box in ann["boxes"]:
            x, y, bw, bh = box
            boxes.append([x * scale_x, y * scale_y, (x + bw) * scale_x, (y + bh) * scale_y])

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = self._default_transforms()(image)

        # Load HSI nếu có
        hsi_tensor = None
        if self.use_hsi:
            hsi_tensor = self._load_hsi(ann.get("hsi_path"))

        # Nếu không có HSI, tạo tensor rỗng với số kênh tương ứng
        if hsi_tensor is None:
            hsi_tensor = torch.zeros(len(self.hsi_bands), self.img_size, self.img_size)

        return {
            "image": image_tensor,           # (3, H, W) - RGB
            "hsi": hsi_tensor,               # (224, H, W) - Hyperspectral
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(ann["labels"], dtype=torch.long),
            "image_path": str(img_path),
            "dataset": "spectralwaste",
            "camera_type": self.CAMERA_TYPE,
            "has_hsi": hsi_tensor is not None,
        }

    def get_client_info(self) -> Dict:
        return {
            "dataset_name": "SpectralWaste",
            "camera_type": self.CAMERA_TYPE,
            "num_classes": len(self.CLASS_NAMES) - 1,
            "num_samples": len(self),
            "modality": "RGB+HSI",
            "hsi_bands": f"{len(self.hsi_bands)} bands (900-1700nm)",
            "hardware": "NVIDIA RTX 3060 (Workstation)",
        }


# ─── TACO Dataset ─────────────────────────────────────────────────────────────

class TACODataset(WasteBaseDataset):
    """
    TACO Dataset - crowdsourced từ nhiều loại camera
    Classes: 60 loại rác (subset: 28 supercategories)
    Download: http://tacodataset.org / https://github.com/pedropro/TACO
    """

    CAMERA_TYPE = "crowdsourced_mobile_varied"
    SUPERCATEGORY_FILTER = [
        "Plastic bag & wrapper", "Bottle", "Can", "Cup", "Carton",
        "Paper", "Cigarette", "Glass bottle", "Metal bottle cap", "Other plastic"
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        use_supercategories: bool = True,  # Gom nhóm 60→28 class
        **kwargs
    ):
        self.use_supercategories = use_supercategories
        super().__init__(root, split, img_size, **kwargs)

    def _load_data(self):
        ann_file = self.root / "annotations.json"
        img_dir = self.root / "data"

        if not ann_file.exists():
            raise FileNotFoundError(
                f"TACO dataset không tìm thấy tại {self.root}\n"
                f"Tải về bằng: git clone https://github.com/pedropro/TACO\n"
                f"Sau đó chạy: python download.py"
            )

        with open(ann_file) as f:
            coco_data = json.load(f)

        # Build category mapping
        cat_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}
        cat_id_to_super = {c["id"]: c.get("supercategory", c["name"])
                           for c in coco_data["categories"]}

        if self.use_supercategories:
            supers = sorted(set(cat_id_to_super.values()))
            super_to_idx = {s: i + 1 for i, s in enumerate(supers)}
            self.class_names = ["background"] + supers
            self._cat_id_to_label = {cid: super_to_idx[cat_id_to_super[cid]]
                                      for cid in cat_id_to_name}
        else:
            cats = sorted(cat_id_to_name.items())
            self.class_names = ["background"] + [c[1] for c in cats]
            self._cat_id_to_label = {cid: i + 1 for i, (cid, _) in enumerate(cats)}

        # Split dataset
        all_images = coco_data["images"]
        n = len(all_images)
        split_idx = {"train": (0, int(0.7 * n)),
                     "val": (int(0.7 * n), int(0.85 * n)),
                     "test": (int(0.85 * n), n)}
        lo, hi = split_idx.get(self.split, (0, n))
        selected_images = all_images[lo:hi]

        img_id_to_anns = {img["id"]: [] for img in selected_images}
        for ann in coco_data.get("annotations", []):
            if ann["image_id"] in img_id_to_anns:
                img_id_to_anns[ann["image_id"]].append(ann)

        for img_info in selected_images:
            img_path = img_dir / img_info["file_name"]
            if not img_path.exists():
                img_path = self.root / img_info["file_name"]
            if img_path.exists():
                anns = img_id_to_anns[img_info["id"]]
                self.images.append(img_path)
                self.annotations.append({
                    "boxes": [a["bbox"] for a in anns],
                    "labels": [self._cat_id_to_label.get(a["category_id"], 0) for a in anns],
                    "masks": [a.get("segmentation", []) for a in anns],
                })

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.images[idx]
        ann = self.annotations[idx]

        image = Image.open(img_path).convert("RGB")
        w_orig, h_orig = image.size
        image = image.resize((self.img_size, self.img_size))

        scale_x, scale_y = self.img_size / w_orig, self.img_size / h_orig
        boxes = []
        for box in ann["boxes"]:
            x, y, bw, bh = box
            boxes.append([x * scale_x, y * scale_y, (x + bw) * scale_x, (y + bh) * scale_y])

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = self._default_transforms()(image)

        return {
            "image": image_tensor,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(ann["labels"], dtype=torch.long),
            "image_path": str(img_path),
            "dataset": "taco",
            "camera_type": self.CAMERA_TYPE,
        }

    def get_client_info(self) -> Dict:
        return {
            "dataset_name": "TACO",
            "camera_type": self.CAMERA_TYPE,
            "num_classes": len(self.class_names) - 1,
            "num_samples": len(self),
            "modality": "RGB (variable)",
            "resolution": "Variable (crowdsourced)",
            "hardware": "NVIDIA Jetson Orin (Edge)",
        }


# ─── MJU-Waste Dataset ────────────────────────────────────────────────────────

class MJUWasteDataset(WasteBaseDataset):
    """
    MJU-Waste Dataset - Sensors 2020
    Camera: Microsoft Kinect (RGB + Depth)
    Classes: waste (binary segmentation)
    Download: https://drive.google.com/file/d/1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny
    """

    CLASS_NAMES = ["background", "waste"]
    CAMERA_TYPE = "rgbd_kinect"

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        use_depth: bool = True,
        **kwargs
    ):
        self.use_depth = use_depth
        self.class_names = self.CLASS_NAMES
        super().__init__(root, split, img_size, **kwargs)

    def _load_data(self):
        split_file = self.root / f"{self.split}.json"

        if not split_file.exists():
            raise FileNotFoundError(
                f"MJU-Waste không tìm thấy tại {self.root}\n"
                f"Tải về tại: https://drive.google.com/file/d/1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny"
            )

        with open(split_file) as f:
            data = json.load(f)

        img_dir = self.root / "JPEGImages"
        depth_dir = self.root / "Depth"
        ann_dir = self.root / "SegmentationClass"

        for img_info in data.get("images", []):
            file_name = img_info["file_name"]
            img_path = img_dir / file_name
            if not img_path.exists():
                continue

            stem = Path(file_name).stem
            depth_path = depth_dir / f"{stem}.png"
            mask_path = ann_dir / f"{stem}.png"

            anns = [a for a in data.get("annotations", [])
                    if a["image_id"] == img_info["id"]]

            self.images.append(img_path)
            self.annotations.append({
                "boxes": [a["bbox"] for a in anns],
                "labels": [1] * len(anns),   # Tất cả đều là "waste"
                "masks": [a.get("segmentation", []) for a in anns],
                "depth_path": str(depth_path) if depth_path.exists() else None,
                "mask_path": str(mask_path) if mask_path.exists() else None,
            })

    def _load_depth(self, depth_path: Optional[str]) -> Optional[torch.Tensor]:
        """Load depth image → (1, H, W) normalized tensor"""
        if depth_path is None or not Path(depth_path).exists():
            return None
        try:
            depth = Image.open(depth_path)
            depth = depth.resize((self.img_size, self.img_size), Image.NEAREST)
            depth_arr = np.array(depth, dtype=np.float32)
            depth_arr = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8)
            return torch.from_numpy(depth_arr).unsqueeze(0)  # (1, H, W)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.images[idx]
        ann = self.annotations[idx]

        image = Image.open(img_path).convert("RGB")
        w_orig, h_orig = image.size
        image = image.resize((self.img_size, self.img_size))

        scale_x, scale_y = self.img_size / w_orig, self.img_size / h_orig
        boxes = []
        for box in ann["boxes"]:
            x, y, bw, bh = box
            boxes.append([x * scale_x, y * scale_y, (x + bw) * scale_x, (y + bh) * scale_y])

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = self._default_transforms()(image)

        depth_tensor = None
        if self.use_depth:
            depth_tensor = self._load_depth(ann.get("depth_path"))
        if depth_tensor is None:
            depth_tensor = torch.zeros(1, self.img_size, self.img_size)

        return {
            "image": image_tensor,
            "depth": depth_tensor,                 # (1, H, W)
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(ann["labels"], dtype=torch.long),
            "image_path": str(img_path),
            "dataset": "mjuwaste",
            "camera_type": self.CAMERA_TYPE,
        }

    def get_client_info(self) -> Dict:
        return {
            "dataset_name": "MJU-Waste",
            "camera_type": self.CAMERA_TYPE,
            "num_classes": 1,
            "num_samples": len(self),
            "modality": "RGBD",
            "resolution": "640x480",
            "hardware": "Raspberry Pi 4 (IoT Edge)",
        }


# ─── Dataset Factory ─────────────────────────────────────────────────────────

DATASET_CLASSES = {
    "zerowaste": ZeroWasteDataset,
    "spectralwaste": SpectralWasteDataset,
    "taco": TACODataset,
    "mjuwaste": MJUWasteDataset,
}


def build_dataset(
    dataset_name: str,
    data_root: str,
    split: str = "train",
    img_size: int = 640,
    **kwargs,
) -> WasteBaseDataset:
    if dataset_name not in DATASET_CLASSES:
        raise ValueError(f"Dataset '{dataset_name}' không hợp lệ. Chọn: {list(DATASET_CLASSES.keys())}")
    cls = DATASET_CLASSES[dataset_name]
    root = os.path.join(data_root, dataset_name)
    return cls(root=root, split=split, img_size=img_size, **kwargs)


def build_dataloader(
    dataset: WasteBaseDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    def collate_fn(batch):
        """Custom collate để xử lý bounding boxes có số lượng khác nhau"""
        images = torch.stack([b["image"] for b in batch])
        boxes = [b["boxes"] for b in batch]
        labels = [b["labels"] for b in batch]
        meta = {
            "image_paths": [b["image_path"] for b in batch],
            "dataset": batch[0]["dataset"],
            "camera_type": batch[0]["camera_type"],
        }

        # Optional modalities
        if "hsi" in batch[0]:
            meta["hsi"] = torch.stack([b["hsi"] for b in batch])
        if "depth" in batch[0]:
            meta["depth"] = torch.stack([b["depth"] for b in batch])

        return {"images": images, "boxes": boxes, "labels": labels, "meta": meta}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=split == "train" if hasattr(dataset, "split") else True,
    )


def build_demo_dataset(dataset_name: str, num_samples: int = 100) -> WasteBaseDataset:
    """Tạo demo dataset synthetic khi chưa có data thực để test pipeline"""
    from torch.utils.data import TensorDataset

    class DemoDataset(Dataset):
        def __init__(self, name, num_samples, img_size=640):
            self.dataset_name = name
            self.num_samples = num_samples
            self.img_size = img_size
            self.class_names = DATASET_CLASSES[name].CLASS_NAMES if hasattr(
                DATASET_CLASSES[name], "CLASS_NAMES") else ["background", "waste"]
            # Camera-specific info
            self.camera_type = getattr(DATASET_CLASSES[name], "CAMERA_TYPE", "unknown")

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Tạo ảnh synthetic với noise theo camera type
            if "spectral" in self.camera_type:
                noise_std = 0.15
            elif "rgbd" in self.camera_type:
                noise_std = 0.10
            elif "crowdsourced" in self.camera_type:
                noise_std = 0.25
            else:
                noise_std = 0.05

            image = torch.randn(3, self.img_size // 4, self.img_size // 4) * noise_std + 0.5
            image = torch.clamp(image, 0, 1)

            # Random boxes
            num_objects = np.random.randint(0, 5)
            boxes = torch.rand(num_objects, 4) * (self.img_size // 4)
            if num_objects > 0:
                boxes[:, 2] = boxes[:, 0] + torch.rand(num_objects) * 50
                boxes[:, 3] = boxes[:, 1] + torch.rand(num_objects) * 50
            labels = torch.randint(1, len(self.class_names), (num_objects,))

            result = {
                "image": image,
                "boxes": boxes,
                "labels": labels,
                "image_path": f"demo_{self.dataset_name}_{idx}.jpg",
                "dataset": self.dataset_name,
                "camera_type": self.camera_type,
            }

            # Add modality-specific tensors
            if "spectral" in self.camera_type:
                result["hsi"] = torch.randn(224, self.img_size // 4, self.img_size // 4) * 0.1
            if "rgbd" in self.camera_type:
                result["depth"] = torch.rand(1, self.img_size // 4, self.img_size // 4)

            return result

        def get_client_info(self):
            return {"dataset_name": self.dataset_name, "num_samples": self.num_samples,
                    "camera_type": self.camera_type, "modality": "synthetic_demo"}

    return DemoDataset(dataset_name, num_samples)
