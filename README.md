# Brain Tumor Detection System

[![CI/CD Pipeline](https://github.com/Osama-Abo-Bakr/brain-tumor-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Osama-Abo-Bakr/brain-tumor-detection/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning application for detecting brain tumors in MRI images using **YOLOv8**, served through a **FastAPI** backend and visualized with a **Streamlit** frontend. Fully containerized with **Docker Compose**.

> **Disclaimer:** This system is for **educational and research purposes only**. It is **not intended for clinical diagnosis**.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                           │
│                                                                 │
│  ┌──────────────────────┐       ┌──────────────────────────┐   │
│  │   Frontend Service    │       │    Backend Service        │   │
│  │   (Streamlit)         │       │    (FastAPI + Uvicorn)    │   │
│  │                       │       │                          │   │
│  │  ┌─────────────────┐ │       │  ┌────────────────────┐  │   │
│  │  │  Upload MRI      │ │ HTTP  │  │  /predict           │  │   │
│  │  │  Images           │─┼──────┼─▶│  POST endpoint      │  │   │
│  │  └─────────────────┘ │       │  └────────┬───────────┘  │   │
│  │                       │       │           │              │   │
│  │  ┌─────────────────┐ │       │  ┌────────▼───────────┐  │   │
│  │  │  Bounding Box    │ │       │  │  YOLOv8s Model     │  │   │
│  │  │  Visualization   │◀┼──────┼──│  (Inference)        │  │   │
│  │  └─────────────────┘ │  JSON │  └────────────────────┘  │   │
│  │                       │       │                          │   │
│  │  Port: 8501           │       │  Port: 8000              │   │
│  └──────────────────────┘       └──────────────────────────┘   │
│                                                                 │
│                        Network: brain_net                        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User uploads MRI image(s)
        │
        ▼
┌───────────────┐    HTTP POST     ┌────────────────┐
│   Streamlit   │ ──────────────▶  │    FastAPI      │
│   Frontend    │   /predict       │    Backend      │
└───────┬───────┘                  └───────┬────────┘
        │                                  │
        │                          ┌───────▼────────┐
        │                          │  Image          │
        │                          │  Validation     │
        │                          │  (type, size)   │
        │                          └───────┬────────┘
        │                                  │
        │                          ┌───────▼────────┐
        │                          │  YOLOv8s        │
        │                          │  Inference      │
        │                          │  (conf >= 0.80) │
        │                          └───────┬────────┘
        │                                  │
        │          JSON response           │
        │  ◀───────────────────────────────┘
        │   {detections, bboxes, confidence}
        │
┌───────▼───────┐
│  Render        │
│  - BBox overlay│
│  - Zoom region │
│  - Stats table │
└───────────────┘
```

---

## Project Structure

```
brain-tumor/
├── backend/
│   ├── backend.py            # FastAPI application with YOLOv8 inference
│   ├── Dockerfile            # Backend container image
│   ├── requirements.txt      # Python dependencies
│   ├── models/
│   │   └── model_yolov8s.pt  # Trained YOLOv8s weights
│   └── Yolo/
│       ├── yolo_arch.md      # YOLOv8 architecture notes
│       └── yolov8.jpg        # Architecture diagram
│
├── frontend/
│   ├── frontend.py           # Streamlit UI application
│   ├── Dockerfile            # Frontend container image
│   └── requirements.txt      # Python dependencies
│
├── notebook-test/
│   ├── Brain_Tumor.ipynb     # Model training notebook
│   ├── YOLO_Brain_Tumor.ipynb# YOLOv8 training notebook
│   └── test-image.jpg        # Sample MRI test image
│
├── tests/
│   ├── test_backend.py       # Backend API tests
│   └── test_frontend.py      # Frontend utility tests
│
├── .github/
│   └── workflows/
│       └── ci.yml            # CI/CD pipeline
│
├── docker-compose.yml        # Multi-container orchestration
├── .dockerignore             # Docker build exclusions
├── pyproject.toml            # Linting & formatting config
└── README.md
```

---

## Model Details

| Property         | Value                              |
|------------------|------------------------------------|
| **Architecture** | YOLOv8s (Small)                    |
| **Framework**    | Ultralytics                        |
| **Task**         | Object Detection                   |
| **Input**        | Brain MRI images (JPG/PNG)         |
| **Output**       | Bounding boxes + class + confidence|
| **Confidence**   | >= 0.80 threshold                  |
| **Max File Size**| 10 MB per image                    |
| **Batch Limit**  | 20 images per request              |

### YOLOv8s Architecture Summary

```
Input (640x640x3)
      │
      ▼
┌─────────────┐
│  Backbone    │  C2f blocks + SPPF
│  (CSPNet)    │  Feature extraction
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Neck      │  FPN (top-down) + PAN (bottom-up)
│  (FPN+PAN)   │  Multi-scale feature fusion
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Head      │  Decoupled + Anchor-free
│ (Detection)  │  BBox regression + Classification
└─────────────┘
```

---

## API Reference

### Health Check

```
GET /
```

**Response:**
```json
{
  "status": "online",
  "service": "Brain Tumor Detection API",
  "version": "1.0.0",
  "model_loaded": true
}
```

### Predict

```
POST /predict
Content-Type: multipart/form-data
```

**Parameters:**
| Field   | Type           | Description                    |
|---------|----------------|--------------------------------|
| `files` | `UploadFile[]` | One or more MRI images (JPG/PNG)|

**Response:**
```json
{
  "count": 1,
  "successful": 1,
  "failed": 0,
  "results": [
    {
      "image_id": "uuid",
      "filename": "mri_scan.jpg",
      "image_size": {"width": 640, "height": 640},
      "detections": [
        {
          "class_id": 0,
          "class_name": "tumor",
          "confidence": 0.92,
          "bbox_xyxy": [120.5, 80.3, 340.2, 290.7]
        }
      ],
      "detection_count": 1,
      "status": "success"
    }
  ]
}
```

### Model Info

```
GET /model/info
```

---

## Quick Start

### Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/Osama-Abo-Bakr/brain-tumor-detection.git
cd brain-tumor-detection

# Build and start all services
docker compose up --build

# Access the application
# Frontend: http://localhost:8501
# Backend:  http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run frontend.py --server.port 8501
```

### Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## Frontend Features

- **Multi-image upload** with drag-and-drop support
- **Batch processing** up to 20 images at once
- **Bounding box visualization** with confidence-based color coding
  - Red: High confidence (>= 80%)
  - Orange: Medium confidence (>= 60%)
  - Yellow: Lower confidence
- **Zoomed tumor regions** for detailed inspection
- **Adjustable confidence threshold** via sidebar slider
- **Session statistics** tracking processed images and detections
- **Docker/Local mode** toggle for flexible deployment

---

## Tech Stack

| Layer      | Technology                     |
|------------|--------------------------------|
| ML Model   | YOLOv8s (Ultralytics)         |
| Backend    | FastAPI + Uvicorn              |
| Frontend   | Streamlit                      |
| Container  | Docker + Docker Compose        |
| Language   | Python 3.10                    |
| CI/CD      | GitHub Actions                 |
| Testing    | pytest + httpx                 |
| Linting    | Ruff                           |

---

## Authors

| Name               | Role          |
|--------------------|---------------|
| **Osama Abo Bakr** | Developer     |
| **Ahmed Nos7y**    | Developer     |
| **Ahmed Fawzy**    | Developer     |
| **Sherief Mohamed** | Developer    |

---

## License

This project is developed as a **Final College Project** for educational purposes.
