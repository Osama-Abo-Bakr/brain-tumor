# 🧠 Brain Tumor Detection System (YOLOv8)

A **deep learning–based brain tumor detection system** built using **YOLOv8**, **FastAPI**, and **Streamlit**, designed as a **final college project**.
The system detects brain tumors from MRI images and visualizes results with bounding boxes and confidence scores.

---

## 📌 Project Overview

This project provides an **end-to-end AI application** that includes:

* 🧠 **YOLOv8** for brain tumor detection
* 🚀 **FastAPI** backend for inference
* 🎨 **Streamlit** frontend for visualization
* 🐳 **Docker & Docker Compose** for deployment
* 📊 Bounding boxes, confidence scores, and zoomed tumor regions

The goal is to demonstrate:

* Practical deep learning deployment
* Frontend–backend separation
* Medical image analysis workflow

> ⚠️ **Note:** This system is for **educational and research purposes only**. It is **not intended for clinical diagnosis**.

---

## 🗂️ Project Structure

```
Brain-Tumor/
├── backend/
│   ├── backend.py              # FastAPI backend (YOLOv8 inference)
│   ├── Dockerfile              # Backend Docker image
│   ├── requirements.txt        # Backend dependencies
│   └── models/
│       └── model_yolov8s.pt    # Trained YOLOv8 model
│
├── frontend/
│   ├── frontend.py             # Streamlit frontend
│   ├── Dockerfile              # Frontend Docker image
│   └── requirements.txt        # Frontend dependencies
│
├── notebook-test/
│   ├── yolov8.ipynb            # Training & testing notebook
│   └── test-image.png          # Sample test image
│
├── docker-compose.yml          # Multi-container orchestration
├── .dockerignore
└── README.md
```

---

## 🧠 Model Details

* **Model:** YOLOv8
* **Task:** Object Detection
* **Input:** Brain MRI images (PNG / JPG)
* **Output:**

  * Bounding box coordinates
  * Class name
  * Confidence score

The model was trained and evaluated using a custom brain tumor dataset.
Evaluation metrics such as **Precision, Recall, F1-score, and Confusion Matrix** were analyzed during training.

---

## 🚀 Backend (FastAPI)

### Features

* Single & batch image inference
* Image validation & error handling
* JSON-based REST API
* Health check endpoint
* Model info endpoint

### Main Endpoint

```
POST /predict
```

**Input:** One or more MRI images
**Output:** Detection results with bounding boxes and confidence scores

---

## 🎨 Frontend (Streamlit)

### Features

* Upload single or multiple images
* Draw bounding boxes on MRI images
* Confidence threshold slider
* Zoomed tumor regions
* Session statistics
* Docker & local mode support

The frontend communicates with the backend via HTTP and visualizes the predictions in real time.

---

## 🐳 Docker & Deployment

This project is fully **containerized** using Docker and Docker Compose.

### Services

* **backend** → FastAPI + YOLOv8
* **frontend** → Streamlit UI

### Run the Project (Recommended)

From the project root:

```bash
docker-compose build
docker-compose up
```

### Access the Application

| Service              | URL                                            |
| -------------------- | ---------------------------------------------- |
| Frontend (Streamlit) | [http://localhost:8501](http://localhost:8501) |
| Backend (FastAPI)    | [http://localhost:8000](http://localhost:8000) |

---

## 🧪 Local Development (Optional)

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn backend:app --reload
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run frontend.py
```

---

## 📊 Evaluation & Visualization

* Confusion Matrix
* Precision–Recall Curve
* F1-score Curve
* Bounding box visualization on MRI images

These are generated during training and testing inside the `notebook-test/` directory.

---

## 🎓 Academic Relevance

This project demonstrates:

* Deep learning model deployment
* Medical image analysis
* REST API design
* Frontend visualization
* Containerized ML systems

It is suitable for:

* Final year projects
* AI / ML coursework
* Medical imaging demonstrations

---

## ⚠️ Disclaimer

This project is **not a medical product** and must **not** be used for real clinical diagnosis.
It is developed strictly for **educational purposes**.

---

Below is a **clean, professional, college-grade README.md** you can **copy–paste directly** into your project root.

It is written in a way that:

* ✅ Professors understand it
* ✅ GitHub viewers understand it
* ✅ You can explain every section in viva
* ❌ No unnecessary over-engineering

---

# 🧠 Brain Tumor Detection System (YOLOv8)

A **deep learning–based brain tumor detection system** built using **YOLOv8**, **FastAPI**, and **Streamlit**, designed as a **final college project**.
The system detects brain tumors from MRI images and visualizes results with bounding boxes and confidence scores.

---

## 📌 Project Overview

This project provides an **end-to-end AI application** that includes:

* 🧠 **YOLOv8** for brain tumor detection
* 🚀 **FastAPI** backend for inference
* 🎨 **Streamlit** frontend for visualization
* 🐳 **Docker & Docker Compose** for deployment
* 📊 Bounding boxes, confidence scores, and zoomed tumor regions

The goal is to demonstrate:

* Practical deep learning deployment
* Frontend–backend separation
* Medical image analysis workflow

> ⚠️ **Note:** This system is for **educational and research purposes only**. It is **not intended for clinical diagnosis**.

---

## 🗂️ Project Structure

```
Brain-Tumor/
├── backend/
│   ├── backend.py              # FastAPI backend (YOLOv8 inference)
│   ├── Dockerfile              # Backend Docker image
│   ├── requirements.txt        # Backend dependencies
│   └── models/
│       └── model_yolov8s.pt    # Trained YOLOv8 model
│
├── frontend/
│   ├── frontend.py             # Streamlit frontend
│   ├── Dockerfile              # Frontend Docker image
│   └── requirements.txt        # Frontend dependencies
│
├── notebook-test/
│   ├── yolov8.ipynb            # Training & testing notebook
│   └── test-image.png          # Sample test image
│
├── docker-compose.yml          # Multi-container orchestration
├── .dockerignore
└── README.md
```

---

## 🧠 Model Details

* **Model:** YOLOv8
* **Task:** Object Detection
* **Input:** Brain MRI images (PNG / JPG)
* **Output:**

  * Bounding box coordinates
  * Class name
  * Confidence score

The model was trained and evaluated using a custom brain tumor dataset.
Evaluation metrics such as **Precision, Recall, F1-score, and Confusion Matrix** were analyzed during training.

---

## 🚀 Backend (FastAPI)

### Features

* Single & batch image inference
* Image validation & error handling
* JSON-based REST API
* Health check endpoint
* Model info endpoint

### Main Endpoint

```
POST /predict
```

**Input:** One or more MRI images
**Output:** Detection results with bounding boxes and confidence scores

---

## 🎨 Frontend (Streamlit)

### Features

* Upload single or multiple images
* Draw bounding boxes on MRI images
* Confidence threshold slider
* Zoomed tumor regions
* Session statistics
* Docker & local mode support

The frontend communicates with the backend via HTTP and visualizes the predictions in real time.

---

## 🐳 Docker & Deployment

This project is fully **containerized** using Docker and Docker Compose.

### Services

* **backend** → FastAPI + YOLOv8
* **frontend** → Streamlit UI

### Run the Project (Recommended)

From the project root:

```bash
docker-compose build
docker-compose up
```

### Access the Application

| Service              | URL                                            |
| -------------------- | ---------------------------------------------- |
| Frontend (Streamlit) | [http://localhost:8501](http://localhost:8501) |
| Backend (FastAPI)    | [http://localhost:8000](http://localhost:8000) |

---

## 🧪 Local Development (Optional)

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn backend:app --reload
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run frontend.py
```

---

## 📊 Evaluation & Visualization

* Confusion Matrix
* Precision–Recall Curve
* F1-score Curve
* Bounding box visualization on MRI images

These are generated during training and testing inside the `notebook-test/` directory.

---

## 🎓 Academic Relevance

This project demonstrates:

* Deep learning model deployment
* Medical image analysis
* REST API design
* Frontend visualization
* Containerized ML systems

It is suitable for:

* Final year projects
* AI / ML coursework
* Medical imaging demonstrations

---

## ⚠️ Disclaimer

This project is **not a medical product** and must **not** be used for real clinical diagnosis.
It is developed strictly for **educational purposes**.

---

## 👨‍💻 Authors

This project was developed as a **Final College Project** by:

1. **Osama Abo Bakr**
2. **Ahmed Nos7y**
3. **Ahmed Fawzy**
4. **Sherief Mohamed**

---

## ⭐ Final Notes

* The system is intentionally **not over-optimized** to keep it understandable.
* The focus is on **architecture, explainability, and correctness**.
* All components can be clearly explained during viva or presentation.