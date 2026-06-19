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
import tifffile as tiff

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

    CLASS_NAMES = ["rigid_plastic", "cardboard", "metal", "soft_plastic"]
    CAMERA_TYPE = "industrial_rgb_fixed"

    def __init__(self, root: str, split: str = "train", img_size: int = 640, **kwargs):
        self.class_names = self.CLASS_NAMES
        super().__init__(root, split, img_size, **kwargs)

    def _load_data(self):
   
        base_dir = self.root / "splits_final_deblurred"

        split_dir = base_dir / self.split / "data"
        ann_file = base_dir / self.split / "labels.json"

        if not split_dir.exists():
            raise FileNotFoundError(
                f"ZeroWaste dataset không tìm thấy tại {self.root}\n"
                f"Tải về tại: https://zenodo.org/record/6412647"
            )
        self.cat_id_to_idx = {
            1: 0,  # rigid_plastic
            2: 1,  # cardboard
            3: 2,  # metal
            4: 3   # soft_plastic
        }
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
                    "labels": [self.cat_id_to_idx[a["category_id"]] for a in anns],
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
    Classes: background", "film", "basket", "cardboard", "video_tape", "filament", "bag
    Download: https://zenodo.org/records/10880544 | OneDrive của tác giả
    """

    CLASS_NAMES = ["background", "film", "basket", "cardboard", "video_tape", "filament", "bag"]
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

        base_dir = self.root / "spectralwaste_segmentation"

        rgb_dir = base_dir / "rgb" / self.split
        hsi_dir = base_dir / "hyper" / self.split

        rgb_mask_dir = base_dir / "labels_rgb" / self.split
        hsi_mask_dir = base_dir / "labels_hyper_lt" / self.split

        if not rgb_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy RGB directory: {rgb_dir}"
            )

        img_files = []

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
            img_files.extend(rgb_dir.glob(ext))

        img_files = sorted(img_files)

        self.images = []
        self.annotations = []

        for img_path in img_files:

            stem = img_path.stem

            hsi_path = None

            for ext in [".tif", ".tiff", ".npy"]:
                candidate = hsi_dir / f"{stem}{ext}"

                if candidate.exists():
                    hsi_path = candidate
                    break

            rgb_mask = rgb_mask_dir / f"{stem}.png"
            hsi_mask = hsi_mask_dir / f"{stem}.png"

            self.images.append(img_path)

            self.annotations.append({
                "boxes": [],
                "labels": [],
                "masks": [],
                "rgb_mask": str(rgb_mask) if rgb_mask.exists() else None,
                "hsi_mask": str(hsi_mask) if hsi_mask.exists() else None,
                "hsi_path": str(hsi_path) if hsi_path else None,
            })

        print(f"Loaded {len(self.images)} SpectralWaste samples")

    def _load_hsi(self, hsi_path: Optional[str]) -> Optional[torch.Tensor]:
        """Load hyperspectral image (H, W, C) → (C, H, W) tensor"""

        if hsi_path is None:
            return None

        hsi_path = Path(hsi_path)

        if not hsi_path.exists():
            return None

        try:

            if hsi_path.suffix == ".npy":
                hsi = np.load(hsi_path).astype(np.float32)

            else:
                hsi = tiff.imread(hsi_path).astype(np.float32)

            if hsi.ndim == 3:
                hsi = hsi[:, :, self.hsi_bands]

            hsi = (hsi - hsi.min()) / (
                hsi.max() - hsi.min() + 1e-8
            )

            hsi = torch.from_numpy(hsi).permute(2, 0, 1)

            hsi = TF.resize(
                hsi,
                [self.img_size, self.img_size]
            )

            return hsi

        except Exception as e:
            print("HSI LOAD ERROR:", e)
            return None
    def _load_hsi(self, hsi_path):
        """Load hyperspectral image (H, W, C) → (C, H, W) tensor"""
        if hsi_path is None:
            return None
        hsi = tiff.imread(hsi_path).astype(np.float32)
        # (H,W,C)
        if hsi.ndim == 3:
            hsi = hsi[:, :, self.hsi_bands]
        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min() + 1e-8)
        return torch.from_numpy(hsi).permute(2,0,1)

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

# ─── RecyclableHouseholdDataset Dataset ───────────────────────────────────────────────────────
class RecyclableHouseholdDataset(WasteBaseDataset):
    """
    RecyclableHouseholdDataset - crowdsourced từ nhiều loại camera
    Classes: 60 loại rác (subset: 28 supercategories)
    Download: http://tacodataset.org / https://github.com/pedropro/TACO
    
    """
    """
    Cấu trúc của RecyclableHouseholdDataset
    recyclable-household/
    └── images/
        ├── Plastic water bottles/
        │   ├── default/
        │   └── real_world/
        ├── Plastic soda bottles/
        ├── Cardboard boxes/
        ├── Newspaper/
        └── ...
        
    """
    CAMERA_TYPE = "consumer_rgb_mixed"
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        **kwargs
    ):
        super().__init__(root, split, img_size, **kwargs)
        
    def _load_data(self):

        images_root = self.root / "images"

        if not images_root.exists():
            raise FileNotFoundError(
                f"Dataset not found: {images_root}"
            )

        class_dirs = sorted(
            [d for d in images_root.iterdir() if d.is_dir()]
        )

        self.class_names = [d.name for d in class_dirs]

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.class_names)
        }

        self.images = []
        self.annotations = []

        for cls_dir in class_dirs:

            class_name = cls_dir.name
            label = self.class_to_idx[class_name]

            for subset in ["default", "real_world"]:

                subset_dir = cls_dir / subset

                if not subset_dir.exists():
                    continue

                for ext in ["*.png", "*.jpg", "*.jpeg"]:

                    for img_path in subset_dir.glob(ext):

                        self.images.append(img_path)

                        self.annotations.append({
                            "label": label
                        })

        print(
            f"Loaded {len(self.images)} samples "
            f"from {len(self.class_names)} classes"
        )
    def __getitem__(self, idx):

        img_path = self.images[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = self._default_transforms()(image)

        return {
            "image": image_tensor,
            "label": torch.tensor(
                self.annotations[idx]["label"],
                dtype=torch.long
            ),
            "image_path": str(img_path),
            "dataset": "recyclable-household",
            "camera_type": self.CAMERA_TYPE,
        }
    def get_client_info(self):

        return {
            "dataset_name": "RecyclableHousehold",
            "camera_type": self.CAMERA_TYPE,
            "num_classes": len(self.class_names),
            "num_samples": len(self),
            "modality": "RGB",
            "hardware": "Mobile Edge Device",
        }

# # ────────────────────────────────────────────────────────────────

# # ─── GarbageClassification Dataset ─────────────────────────────────────────────────────────────

class GarbageClassificationDataset(WasteBaseDataset):
    """
    Garbage Classification Dataset

    Classes:
    cardboard
    glass
    metal
    paper
    plastic
    trash
    """

    CLASS_NAMES = [
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
        "trash"
    ]

    CAMERA_TYPE = "consumer_rgb"

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 640,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.class_names = self.CLASS_NAMES

        super().__init__(
            root,
            split,
            img_size,
            **kwargs
        )

    def _load_data(self):

        self.images = []
        self.annotations = []

        class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.CLASS_NAMES)
        }

        all_samples = []

        for cls in self.CLASS_NAMES:

            cls_dir = self.root / cls

            if not cls_dir.exists():
                continue

            for ext in ["*.jpg", "*.jpeg", "*.png"]:

                for img_path in cls_dir.glob(ext):

                    all_samples.append(
                        (img_path, class_to_idx[cls])
                    )

        np.random.seed(42)
        np.random.shuffle(all_samples)

        n = len(all_samples)

        train_end = int(n * self.train_ratio)
        val_end = int(
            n * (self.train_ratio + self.val_ratio)
        )

        if self.split == "train":
            samples = all_samples[:train_end]

        elif self.split == "val":
            samples = all_samples[
                train_end:val_end
            ]

        else:
            samples = all_samples[val_end:]

        for img_path, label in samples:

            self.images.append(img_path)

            self.annotations.append({
                "label": label
            })

        print(
            f"Loaded {len(self.images)} "
            f"{self.split} samples"
        )

    def __getitem__(self, idx):

        img_path = self.images[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = self._default_transforms()(image)

        return {
            "image": image_tensor,
            "label": torch.tensor(
                self.annotations[idx]["label"],
                dtype=torch.long
            ),
            "image_path": str(img_path),
            "dataset": "garbage-classification",
            "camera_type": self.CAMERA_TYPE,
        }

    def get_client_info(self):

        return {
            "dataset_name": "GarbageClassification",
            "camera_type": self.CAMERA_TYPE,
            "num_classes": len(self.CLASS_NAMES),
            "num_samples": len(self),
            "modality": "RGB",
            "hardware": "Desktop GPU",
        }
# # ───  ─────────────────────────────────────────────────────────────
# # ─── TACO Dataset ─────────────────────────────────────────────────────────────

# class TACODataset(WasteBaseDataset):
#     """
#     TACO Dataset - crowdsourced từ nhiều loại camera
#     Classes: 60 loại rác (subset: 28 supercategories)
#     Download: http://tacodataset.org / https://github.com/pedropro/TACO
#     """

#     CAMERA_TYPE = "crowdsourced_mobile_varied"
#     # SUPERCATEGORY_FILTER = [
#     #     "Plastic bag & wrapper", "Bottle", "Can", "Cup", "Carton",
#     #     "Paper", "Cigarette", "Glass bottle", "Metal bottle cap", "Other plastic"
#     # ]

#     def __init__(
#         self,
#         root: str,
#         split: str = "train",
#         img_size: int = 640,
#         use_supercategories: bool = True,  # Gom nhóm 60→28 class
#         **kwargs
#     ):
#         self.use_supercategories = use_supercategories
#         super().__init__(root, split, img_size, **kwargs)
#     def _load_data(self):
#         ann_file = self.root / "data" / "annotations.json"
#         img_dir = self.root / "data"

#         with open(ann_file) as f:
#             coco_data = json.load(f)

#         images = coco_data["images"]
#         annotations = coco_data["annotations"]
#         categories = coco_data["categories"]
        
#         # SUPER CATEGORY MAPPING (28 classes)
#         # =========================
#         cat_id_to_super = {
#             c["id"]: c["supercategory"].strip().lower()
#             for c in coco_data["categories"]
#         }
#         supers = sorted(set(cat_id_to_super.values()))
#         super_to_idx = {s: i + 1 for i, s in enumerate(supers)}

#         self.class_names = ["background"] + supers
#         self._cat_id_to_label = {
#             cid: super_to_idx[cat_id_to_super[cid]]
#             for cid in cat_id_to_super
#         }

#         # =========================
#         # IMAGE INDEX (FIX #2)
#         # =========================
#         img_id_to_img = {img["id"]: img for img in images}
#         image_ids = list(img_id_to_img.keys())
#         np.random.shuffle(image_ids)

#         n = len(image_ids)
#         lo, hi = {
#             "train": (0, int(0.7 * n)),
#             "val": (int(0.7 * n), int(0.85 * n)),
#             "test": (int(0.85 * n), n),
#         }.get(self.split, (0, n))

#         selected_ids = image_ids[lo:hi]

#         # =========================
#         # GROUP ANNOTATIONS
#         # =========================
#         img_id_to_anns = {}
#         for ann in annotations:
#             img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

#         # =========================
#         # BUILD DATASET (FIX #3)
#         # =========================
#         for img_id in selected_ids:
#             img_info = img_id_to_img[img_id]
#             anns = img_id_to_anns.get(img_id, [])

#             img_path = img_dir / img_info["file_name"]
#             if not img_path.exists():
#                 img_path = self.root / img_info["file_name"]

#             if not img_path.exists():
#                 continue

#             boxes = []
#             labels = []

#             for a in anns:
#                 cid = a["category_id"]
#                 if cid not in self._cat_id_to_label:
#                     continue

#                 boxes.append(a["bbox"])
#                 labels.append(self._cat_id_to_label[cid])

#             # IMPORTANT FIX
#             if len(boxes) == 0:
#                 continue

#             self.images.append(img_path)
#             self.annotations.append({
#                 "boxes": boxes,
#                 "labels": labels,
#             })
#             # =========================
#             # DEBUG LABEL DISTRIBUTION
#             # =========================
#             from collections import Counter

#             all_labels = []

#             for ann in self.annotations:
#                 all_labels.extend(ann["labels"])

#             # print("\n🔥 LABEL DISTRIBUTION")
#             # print(Counter(all_labels))

#             # print("🔥 NUM IMAGES:", len(self.images))
#             # print("🔥 NUM LABELS:", len(all_labels))
#             # print("🔥 UNIQUE LABELS:", sorted(set(all_labels)))
#     def __getitem__(self, idx: int) -> Dict:
#         img_path = self.images[idx]
#         ann = self.annotations[idx]

#         image = Image.open(img_path).convert("RGB")
#         w_orig, h_orig = image.size
#         image = image.resize((self.img_size, self.img_size))

#         scale_x, scale_y = self.img_size / w_orig, self.img_size / h_orig
#         boxes = []
#         for box in ann["boxes"]:
#             # x, y, bw, bh = box
#             # boxes.append([x * scale_x, y * scale_y, (x + bw) * scale_x, (y + bh) * scale_y])
#             cx = (x + bw / 2) * scale_x / self.img_size
#             cy = (y + bh / 2) * scale_y / self.img_size
#             w = bw * scale_x / self.img_size
#             h = bh * scale_y / self.img_size

#             boxes.append([cx, cy, w, h])


#         if self.transform:
#             image_tensor = self.transform(image)
#         else:
#             image_tensor = self._default_transforms()(image)

#         return {
#             "image": image_tensor,
#             "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
#             "labels": torch.tensor(ann["labels"], dtype=torch.long),
#             "image_path": str(img_path),
#             "dataset": "taco",
#             "camera_type": self.CAMERA_TYPE,
#         }

#     def get_client_info(self) -> Dict:
#         return {
#             "dataset_name": "TACO",
#             "camera_type": self.CAMERA_TYPE,
#             "num_classes": len(self.class_names) - 1,
#             "num_samples": len(self),
#             "modality": "RGB (variable)",
#             "resolution": "Variable (crowdsourced)",
#             "hardware": "NVIDIA Jetson Orin (Edge)",
#         }


# # ─── MJU-Waste Dataset ────────────────────────────────────────────────────────
# class MJUWasteDataset(WasteBaseDataset):
#     """
#     MJU-Waste Dataset

#     Structure:
#     mju-waste/
#     ├── DepthImages/
#     ├── ImageSets/
#     │   └── Segmentation/
#     │       ├── train.txt
#     │       ├── val.txt
#     │       ├── test.txt
#     ├── JPEGImages/
#     ├── SegmentationClass/
#     """

#     CLASS_NAMES = ["background", "waste"]
#     CAMERA_TYPE = "rgbd_kinect"

#     def __init__(
#         self,
#         root: str,
#         split: str = "train",
#         img_size: int = 640,
#         use_depth: bool = True,
#         **kwargs
#     ):
#         self.use_depth = use_depth
#         self.class_names = self.CLASS_NAMES

#         super().__init__(
#             root=root,
#             split=split,
#             img_size=img_size,
#             **kwargs
#         )

#     def _load_data(self):

#         split_file = (
#             self.root
#             / "ImageSets"
#             / "Segmentation"
#             / f"{self.split}.txt"
#         )

#         if not split_file.exists():
#             raise FileNotFoundError(
#                 f"Không tìm thấy split file: {split_file}"
#             )

#         img_dir = self.root / "JPEGImages"
#         depth_dir = self.root / "DepthImages"
#         mask_dir = self.root / "SegmentationClass"

#         with open(split_file, "r") as f:
#             image_ids = [
#                 line.strip()
#                 for line in f.readlines()
#                 if line.strip()
#             ]

#         for image_id in image_ids:

#             img_path = img_dir / f"{image_id}.jpg"

#             if not img_path.exists():
#                 img_path = img_dir / f"{image_id}.png"

#             if not img_path.exists():
#                 continue

#             depth_path = depth_dir / f"{image_id}.png"
#             mask_path = mask_dir / f"{image_id}.png"

#             self.images.append(img_path)

#             self.annotations.append({
#                 "depth_path": (
#                     str(depth_path)
#                     if depth_path.exists()
#                     else None
#                 ),
#                 "mask_path": (
#                     str(mask_path)
#                     if mask_path.exists()
#                     else None
#                 )
#             })

#     def _load_depth(
#         self,
#         depth_path: Optional[str]
#     ) -> Optional[torch.Tensor]:

#         if depth_path is None:
#             return None

#         if not Path(depth_path).exists():
#             return None

#         try:
#             depth = Image.open(depth_path)

#             depth = depth.resize(
#                 (self.img_size, self.img_size),
#                 Image.NEAREST
#             )

#             depth_arr = np.array(
#                 depth,
#                 dtype=np.float32
#             )

#             depth_arr = (
#                 depth_arr - depth_arr.min()
#             ) / (
#                 depth_arr.max()
#                 - depth_arr.min()
#                 + 1e-8
#             )

#             return torch.from_numpy(
#                 depth_arr
#             ).unsqueeze(0)

#         except Exception:
#             return None

#     def _load_mask(
#         self,
#         mask_path: Optional[str]
#     ) -> Optional[torch.Tensor]:

#         if mask_path is None:
#             return None

#         if not Path(mask_path).exists():
#             return None

#         try:
#             mask = Image.open(mask_path)

#             mask = mask.resize(
#                 (self.img_size, self.img_size),
#                 Image.NEAREST
#             )

#             mask = np.array(mask)

#             mask = (mask > 0).astype(np.uint8)

#             return torch.from_numpy(mask).long()

#         except Exception:
#             return None

#     def __getitem__(self, idx: int):

#         img_path = self.images[idx]
#         ann = self.annotations[idx]

#         image = Image.open(img_path).convert("RGB")

#         image = image.resize(
#             (self.img_size, self.img_size)
#         )

#         if self.transform:
#             image_tensor = self.transform(image)
#         else:
#             image_tensor = self._default_transforms()(image)

#         # Depth
#         depth_tensor = None

#         if self.use_depth:
#             depth_tensor = self._load_depth(
#                 ann["depth_path"]
#             )

#         if depth_tensor is None:
#             depth_tensor = torch.zeros(
#                 1,
#                 self.img_size,
#                 self.img_size
#             )

#         # Segmentation Mask
#         mask_tensor = self._load_mask(
#             ann["mask_path"]
#         )

#         if mask_tensor is None:
#             mask_tensor = torch.zeros(
#                 self.img_size,
#                 self.img_size,
#                 dtype=torch.long
#             )
#         # tạo pseudo label từ mask
#         # if mask_tensor.sum() > 0:
#         #     labels = torch.tensor([1], dtype=torch.long)
#         # else:
#         #     labels = torch.tensor([0], dtype=torch.long)
#         if mask_tensor.sum() > 0:
#             ys, xs = torch.where(mask_tensor > 0)
#             x1, x2 = xs.min().item(), xs.max().item()
#             y1, y2 = ys.min().item(), ys.max().item()

#             cx = (x1 + x2) / 2 / self.img_size
#             cy = (y1 + y2) / 2 / self.img_size
#             w = (x2 - x1) / self.img_size
#             h = (y2 - y1) / self.img_size

#             boxes = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)
#             labels = torch.tensor([1])
#         else:
#             boxes = torch.zeros((0,4))
#             labels = torch.zeros((0,), dtype=torch.long)
#         return {
#             "image": image_tensor,
#             "depth": depth_tensor,
#             "mask": mask_tensor,
            
#             "boxes": torch.zeros((0,4), dtype=torch.float32),

#             "labels": torch.tensor([1], dtype=torch.long),
#             "image_path": str(img_path),
#             "dataset": "mjuwaste",
#             "camera_type": self.CAMERA_TYPE,
#         }

#     def get_client_info(self):

#         return {
#             "dataset_name": "MJU-Waste",
#             "camera_type": self.CAMERA_TYPE,
#             "num_classes": 1,
#             "num_samples": len(self),
#             "modality": "RGBD",
#             "resolution": "640x480",
#             "hardware": "Raspberry Pi 4 (IoT Edge)",
#         }

# ─── Dataset Factory ─────────────────────────────────────────────────────────

DATASET_CLASSES = {
    "zerowaste-f-final": ZeroWasteDataset,
    "spectralwaste-segmentation": SpectralWasteDataset,
    "recyclable-household": RecyclableHouseholdDataset,
    "garbage-classification": GarbageClassificationDataset,
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
