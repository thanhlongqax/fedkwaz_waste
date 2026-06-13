# 📦 **FedKWAZ-Waste: Federated MLOps for Multi-Facility Waste Management**

> 🚀 **Xây dựng hệ thống học liên kết tương hỗ (Federated Mutual Learning)** giải quyết **dị thể kép (dual heterogeneity)** trong quản lý rác đa cơ sở, dựa trên **FedKWAZ (NeurIPS 2025)** và mở rộng với **Camera Weak-Aware Zones (Camera-WAZ)**.

***

## 📖 **Overview**

Dự án này triển khai một hệ thống **Federated MLOps end-to-end** cho bài toán:

* 🔄 Học liên kết giữa nhiều nhà máy tái chế
* 🎥 Xử lý dữ liệu từ nhiều loại camera:
  * RGB (industrial)
  * Hyperspectral (HSI)
  * RGB-D
  * Mobile cameras
* 🧠 Giải quyết:
  * **Dị thể dữ liệu (Non-IID)**
  * **Dị thể mô hình (ResNet, YOLO, MobileNet, …)**
  * 🚀 **Dị thể camera (Cross-modal heterogeneity – đóng góp chính)**

***

## 🎯 **Key Contributions**

✅ Triển khai **FedKWAZ framework**  
✅ Đề xuất **Camera-WAZ (Cross-modal alignment)**  
✅ Xây dựng **Federated MLOps pipeline thực tế**  
✅ Hỗ trợ multi-factory real-world setup



## 📐 Kiến trúc Hệ thống

```
┌──────────────────────────────────────────────────────────────────┐
│                    FL SERVER (MLOps Hub)                         │
│   ProxyModel (ResNet18) + KWAZ-Aware Aggregation                 │
└──────────┬──────────────┬──────────────┬───────────────────────┘
           │              │              │              │
    ┌──────▼──┐    ┌──────▼──┐    ┌──────▼──┐    ┌──────▼──┐
    │Client 0 │    │Client 1 │    │Client 2 │    │Client 3 │
    │ZeroWaste│    │Spectral │    │  TACO   │    │MJU-Waste│
    │ResNet-50│    │EffNet-B3│    │YOLO-Nano│    │MobileV3 │
    │RGB 4K   │    │RGB+HSI  │    │Variable │    │RGBD     │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘
         ↕ KWAZ Exchange (compressed, no raw data shared)
```

## 🏗️ **Project Architecture**

```
fedkwaz_waste_mlops/
│
├── README.md                     # Mô tả dự án (file hiện tại bạn đang viết)
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (MLflow, paths,...)
│
├── configs/                      # 🔧 Configuration management
│   ├── config.py                 # Config chính (central config)
│   ├── data.yaml                 # Config dataset paths, batch size
│   ├── model.yaml                # Config model (ResNet, YOLO,...)
│   ├── fl.yaml                   # FL settings (rounds, clients,...)
│   └── deployment.yaml           # API + inference config
│
├── data/                         # 📦 Data storage
│   ├── raw/                      # Raw datasets (input)
│   │   ├── zerowaste/
│   │   ├── spectralwaste/
│   │   ├── taco/
│   │   └── mjuwaste/
│   │
│   └── processed/                # Preprocessed data (cached)
│       ├── rgb/
│       ├── hsi/
│       └── depth/
│
├── datasets/                     # 📊 Dataset loaders
│   ├── waste_datasets.py         # Loader 4 datasets
│   └── transforms.py             # Data augmentation + preprocessing
│
├── models/                       # 🧠 Model architectures
│   ├── architectures.py          # Define:
│   │   ├── ResNet50
│   │   ├── EfficientNet
│   │   ├── YOLOv8
│   │   └── MobileNet
│   │
│   ├── yolo_model.py             # YOLO wrapper
│   └── model_utils.py            # Load/save model
│
├── core/                         # 🚀 Core research logic (MOST IMPORTANT)
│   ├── kwaz.py                   # KWAZDetector + KWAZLoss
│   ├── camera_waz.py             # 🔥 Cross-modal (RGB ↔ HSI ↔ Depth)
│   ├── distillation.py           # Knowledge distillation
│   └── hapm_kdp.py               # HAPM + KDP modules
│
├── fedkwaz/                      # (optional: legacy)
│   └── kwaz_core.py              # nếu bạn giữ code gốc
│
├── clients/                      # 🤖 Federated clients (multi-factory)
│   ├── base_client.py            # Base FL client class
│   │
│   ├── rgb_client/
│   │   ├── client.py             # RGB client (ResNet)
│   │   └── dataset.py
│   │
│   ├── hsi_client/
│   │   ├── client.py             # HSI client (EfficientNet)
│   │   └── dataset.py
│   │
│   ├── mobile_client/
│   │   ├── client.py             # YOLO client
│   │   └── dataset.py
│   │
│   └── depth_client/
│       ├── client.py             # RGB-D client
│       └── dataset.py
│
├── server/                       # 🧠 Federated server
│   ├── fl_server.py              # Flower server entrypoint
│   ├── aggregator.py             # FedKWAZ aggregation
│   ├── proxy_model.py            # Global proxy model
│   └── strategy.py               # Custom FL strategy
│
├── pipelines/                    # 🔄 MLOps pipelines
│   ├── data_pipeline.py          # Data ingestion + preprocessing
│   ├── training_pipeline.py      # Federated training orchestration
│   ├── evaluation_pipeline.py    # mAP, PR, cross-domain
│   └── deployment_pipeline.py    # Model deployment
│
├── monitoring/                   # 📊 Tracking / logging
│   ├── mlflow_logger.py          # MLflow integration
│   ├── metrics.py                # mAP, accuracy, F1
│   ├── visualization.py          # Dashboard plots
│   └── dashboard.py              # Live dashboard
│
├── registry/                     # 📦 Model registry (versioning)
│   ├── model_registry.py
│   └── versions/
│       ├── v1_fedavg.pt
│       ├── v2_fedkwaz.pt
│       └── v3_camera_waz.pt
│
├── deployment/                  # 🚀 Production / inference
│   ├── api.py                   # FastAPI server
│   ├── inference.py             # Model inference logic
│   ├── preprocess.py            # Input pipeline
│   └── postprocess.py           # Output formatting
│
├── experiments/                 # 🧪 Experiment tracking (manual)
│   ├── exp_01/
│   ├── exp_02/
│   └── results.csv
│
├── docker/                      # 🐳 Containerization
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   └── docker-compose.yml
│
├── utils/                       # 🔧 Utilities
│   ├── logger.py
│   ├── seed.py                  # Fix random seed
│   └── config_loader.py
│
├── train.py                     # 🚀 Main entrypoint (research mode)
├── serve.py                     # Continuous FL loop (production)
└── scripts/
    ├── run_server.sh
    ├── run_client_rgb.sh
    └── run_all.sh
```

***

```

***

## 🧠 **Core Methods**

### 🔹 1. Federated Learning

* Distributed training without data sharing
* Implemented via **Flower (FLwr)**

***

### 🔹 2. FedKWAZ (NeurIPS 2025)

* Knowledge Weak-Aware Zones (KWAZ)
* Detects representation & decision gaps
* Improves cross-model knowledge transfer

***

### 🔹 3. 🚀 Camera-WAZ (Proposed)

* Extends KWAZ to **cross-modal domain gaps**
* Aligns features across:
  * RGB ↔ HSI
  * RGB ↔ Depth

**Loss function:**

```python
loss = task_loss + α * kwaz_loss + β * camera_waz_loss
```

***

## ⚙️ **Tech Stack**

| Component           | Technology |
| ------------------- | ---------- |
| Deep Learning       | PyTorch    |
| Object Detection    | YOLOv8     |
| Federated Learning  | Flower     |
| Experiment Tracking | MLflow     |
| API Deployment      | FastAPI    |
| Containerization    | Docker     |

***

## 📊 **Evaluation Metrics**

### ✅ Computer Vision

* mAP\@0.5
* mAP\@0.5:0.95
* Precision / Recall / F1

***

### ✅ Federated Learning

* Global accuracy
* Client-wise performance
* Convergence speed

***

### ✅ Cross-Camera (🔥 QUAN TRỌNG)

* Cross-domain mAP:
  * RGB → HSI
  * HSI → RGB-D
* KWAZ Loss (alignment quality)

***

### ✅ System Metrics

* Communication overhead
* Inference latency (FPS)

***

## 📂 **Datasets**

Bạn cần chuẩn bị thư mục:

```
data/
├── zerowaste/
├── spectralwaste/
├── taco/
└── mjuwaste/
```

***

## 🚀 **Getting Started**

### 🔹 1. Installation

```bash
pip install -r requirements.txt
```

***

### 🔹 2. Run Federated Training

👉 Start server:

```bash
python server/fl_server.py
```

👉 Start clients:

```bash
python clients/rgb_client/client.py
python clients/hsi_client/client.py
python clients/mobile_client/client.py
python clients/depth_client/client.py
```

***

### 🔹 3. Run Pipeline (MLOps mode)

```bash
python pipelines/training_pipeline.py
```

***

### 🔹 4. Start Inference API

```bash
python deployment/api.py
```

***

## 🐳 **Docker (Optional)**

```bash
docker-compose up --build
```

***

## 🔄 **Federated MLOps Loop**

```
Collect Data → Local Training → Federated Aggregation →
Model Evaluation → Deployment → Repeat
```

***

## 🎯 **Research Scope**

Dự án này hướng tới:

* ✅ Industrial AI system
* ✅ Multi-sensor learning
* ✅ Privacy-preserving learning
* ✅ Cross-modal generalization

***

## 📌 **Future Work**

* Adaptive modality weighting
* Multi-proxy learning
* Real-time factory deployment
* Edge optimization (Jetson / Raspberry Pi)

***

## 👨‍💻 **Author**

Nguyễn Lâm Thanh Long  
Federated Learning & MLOps Engineer

***

## ⭐ **Citation**

```text
@misc{fedkwaz-waste-2026,
  title={Federated Mutual Learning for Multi-Facility Waste Management},
  author={Nguyen Lam Thanh Long},
  year={2026}
}
```