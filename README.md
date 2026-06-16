# FedKWAZ: Federated Learning cho Quản lý Rác thải Công nghiệp Đa nhà máy

> **"Xây dựng MLOps Học liên kết Tương hỗ giải quyết Dị thể Kép trong Quản lý Rác thải Đa cơ sở"**  
> Dựa trên FedKWAZ — NeurIPS 2025 | Mở rộng với Camera WAZ cho môi trường nhà máy tái chế

---

## 📐 Kiến trúc Hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                  FedKWAZ MLOps Hub                          │
│                                                             │
│  Proxy Model (ResNet18)                                     │
│  Prototype Aggregation Engine                              │
│  Prototype Bank                                             │
│  Monitoring & Visualization                                 │
│  Checkpoint / Model Registry                                │
└──────────┬──────────────┬──────────────┬────────────────────┘
           │              │              │
           │ Global Prototype Broadcast
           │
    ┌──────▼──┐    ┌──────▼──┐    ┌──────▼──┐    ┌──────▼──┐
    │Client 0 │    │Client 1 │    │Client 2 │    │Client 3 │
    │ZeroWaste│    │Spectral │    │  TACO   │    │MJUWaste │
    │ResNet50 │    │EffNetB3 │    │YOLONano │    │MobileV3 │
    │RGB      │    │RGB+HSI  │    │RGB      │    │RGBD     │
    └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘
         │              │              │              │
         └──────────────┼──────────────┼──────────────┘
                        │
          Feature Prototypes + Class Centers
                        │
                 (No Raw Data Shared)
```
## 📦 Cấu trúc Project

```
fedkwaz_waste/
├── configs/
│   └── config.py              # Cấu hình tập trung
├── datasets/
│   └── waste_datasets.py      # Dataset loaders (4 datasets)
├── models/
│   └── architectures.py       # Mô hình dị thể (ResNet50, EfficientNet, YOLOn, MobileNet)
├── fedkwaz/
│   └── kwaz_core.py           # Core: KWAZDetector + HAPM + KDP + FedKWAZLoss
├── client/
│   └── fl_client.py           # FL Client logic
├── server/
│   └── fl_server.py           # FL Server + aggregation
├── utils/
│   ├── visualization.py       # Training dashboard plots
│   └── metrics.py
├── train.py                   # 🚀 Entry point
└── requirements.txt
```

---

## ⚙️ Cài đặt

```bash
# 1. Clone hoặc extract project
cd fedkwaz_waste

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Cài dependencies
pip install -r requirements.txt
```

---

## 🚀 Chạy thực nghiệm

### Demo Mode (Không cần dataset thực - chạy ngay)
```bash
python train.py --mode demo --num_rounds 20 --batch_size 8
```

### Full Mode (Với dataset thực)
```bash
# Cấu trúc thư mục data cần có:
# data/
#   zerowaste/       ← ZeroWaste-f dataset
#   spectralwaste/   ← SpectralWaste dataset
#   taco/            ← TACO dataset
#   mjuwaste/        ← MJU-Waste dataset

## 📦 Cấu trúc dataset zerowaste-f-final
```
/splits_final_deblurred/
├── test/
│   └── data
    └── sem_seg   
    └── labels.json   
├── train/
│   └── data
    └── sem_seg   
    └── labels.json         
├── val/
│   └── data
    └── sem_seg   
    └── labels.json  

```
## 📦 Cấu trúc dataset TACO_dataset
```
/data/
├── batch_1/  
├── batch_2/
...
├── batch_15/        
├── annotations.json 
├── annotations_unofficial.json 

```

## 📦 Cấu trúc dataset spectralwaste_segmentation
```
/spectralwaste_segmentation/
├── rgb/
│   └── test/
    └── train/   
    └── val/
├── labels_rgb/
│   └── test/
    └── train/   
    └── val/
├── labels_hyper_lt/
│   └── test/
    └── train/   
    └── val/
├── hyper/
│   └── test/
    └── train/   
    └── val/
├── meta.json 

```
## 📦 Cấu trúc dataset mju-waste
```
/mju-waste/
├── DepthImages/
│   └── test/
├── ImageSets/
│   └── Segmentation/
      └── test.txt
      └── train.txt
      └── val.txt
      └── trainval.txt
    └── val/
├── JPEGImages/
├── SegmentationClass/

```
---

python train.py \
    --mode full \
    --data_root ./data \
    --num_rounds 100 \
    --local_epochs 3 \
    --batch_size 16 \
    --img_size 640 \
    --output_dir ./outputs
```

### Resume từ checkpoint
```bash
python train.py --mode full --resume ./outputs/checkpoint_round_50.pt
```

---

## 📥 Download Datasets

| Dataset | Link | Size |
|---------|------|------|
| ZeroWaste-f | https://zenodo.org/record/6412647 | ~2 GB |
| SpectralWaste (preprocessed) | https://zenodo.org/records/10880544 | 23 GB |
| SpectralWaste (OneDrive) | [OneDrive tác giả](https://unizares-my.sharepoint.com/:u:/g/personal/756012_unizar_es/EVJygVCmvs1BrCvA_WEtcIcBkUGbgsmN4fLaWGwr_lLJBw?e=lSPWxs) | 23 GB |
| TACO | `git clone https://github.com/pedropro/TACO && python download.py` | ~7 GB |
| MJU-Waste | https://drive.google.com/file/d/1o101UBJGeeMPpI-DSY6oh-tLk9AHXMny | ~500 MB |

---

## 🧩 Mô tả Các Thành phần

### 1. KWAZDetector — 3 loại vùng nhận thức yếu
- **Semantic WAZ**: Cosine similarity + L2 discrepancy trong representation space
- **Decision WAZ**: KL/JS divergence giữa class probability distributions
- **Camera WAZ** *(Novel)*: Sai lệch do loại camera/sensor khác nhau

### 2. HAPM — Hierarchical Adaptive Patch Mixing
- 3 cấp độ patch: 32×32, 64×64, 128×128
- CutMix ratio adaptive theo KWAZ score
- Tạo ra mẫu "bridging" giữa private và proxy knowledge

### 3. KDP — Knowledge Discrepancy Perceptron
- NT-Xent contrastive loss để align representations
- Chọn top-k mẫu có discrepancy cao nhất làm focal training target

### 4. KWAZ-Aware Aggregation
- **50%** sample-count weight (standard FedAvg)
- **30%** learning quality weight (inverse KWAZ loss)
- **20%** camera diversity bonus (rare camera type ← novelty)

---

## 📊 Outputs

Sau khi train, thư mục `outputs/` chứa:
```
outputs/
├── fedkwaz_training_dashboard.png  # Training curves đẹp
├── training_history.json           # Full metrics history
├── best_model.pt                   # Best checkpoint
└── fedkwaz_training.log            # Training log
```

---

## 📝 Citation

```bibtex
@inproceedings{li2025fedkwaz,
  title={Transforming Gaps into Gains: Bridging Model and Data Heterogeneity
         in Federated Learning via Knowledge Weak-Aware Zones},
  author={Li, Ke and Ding, Yan and Zhu, Zhiqin and Zheng, Shenhai},
  booktitle={NeurIPS},
  year={2025}
}

@article{wang2020multi,
  title={A Multi-Level Approach to Waste Object Segmentation},
  author={Wang, Tao and Cai, Yuanzheng and Liang, Lingyu and Ye, Dongyi},
  journal={Sensors},
  year={2020}
}

@inproceedings{casao2024spectralwaste,
  title={SpectralWaste Dataset: Multimodal Data for Waste Sorting Automation},
  author={Casao, Sara and Pe{\~n}a, Fernando and Sabater, Alberto and others},
  booktitle={IROS},
  year={2024}
}
```
