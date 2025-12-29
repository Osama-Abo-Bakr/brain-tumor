import os
import io
import uuid
import logging
from PIL import Image
from typing import List
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for Brain Tumor detection using YOLOv8 model",
    version="1.0.0",
    debug=True,
)

# ============================
# CORS Configuration
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Model Loading
# ============================
MODEL_PATH =  "./backend/models/model_yolov8s.pt"
model = YOLO(MODEL_PATH)
logger.info(f"✅ Model loaded successfully from: {MODEL_PATH}")

# ============================
# Health Check Endpoint
# ============================
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Brain Tumor Detection API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if model else "Not loaded"
    }

# ============================
# Helper Functions
# ============================
def validate_image(file: UploadFile) -> bool:
    """Validate uploaded file is an image"""
    valid_types = ["image/jpeg", "image/jpg", "image/png"]
    return file.content_type in valid_types

def process_image(image_bytes: bytes, filename: str) -> dict:
    """Process a single image and return detections"""
    try:
        # Open and validate image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Check image size (optional: resize if too large)
        max_size = 4096
        if max(image.size) > max_size:
            logger.warning(f"Image {filename} is large ({image.size}). Consider resizing.")
        
        # Run inference
        results = model(image, conf=0.80, verbose=False)
        
        # Extract detections
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            detections.append({
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": float(box.conf),
                "bbox_xyxy": [float(coord) for coord in box.xyxy.tolist()[0]]
            })
        
        return {
            "image_id": str(uuid.uuid4()),
            "filename": filename,
            "image_size": {"width": image.size[0], "height": image.size[1]},
            "detections": detections,
            "detection_count": len(detections),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return {
            "image_id": str(uuid.uuid4()),
            "filename": filename,
            "detections": [],
            "detection_count": 0,
            "status": "error",
            "error": str(e)
        }

# ============================
# Main Prediction Endpoint
# ============================
@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    """
    Process one or multiple MRI images for tumor detection
    
    Args:
        files: List of image files (JPG, PNG)
    
    Returns:
        JSON with detection results for each image
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check server logs."
        )
    
    # Validate input
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files allowed per request"
        )
    
    results_response = []
    failed_files = []
    
    for file in files:
        # Validate file type
        if not validate_image(file):
            logger.warning(f"Invalid file type for {file.filename}: {file.content_type}")
            failed_files.append({
                "filename": file.filename,
                "reason": f"Invalid file type: {file.content_type}"
            })
            continue
        
        try:
            # Read image bytes
            image_bytes = await file.read()
            
            # Validate file size (max 10MB)
            if len(image_bytes) > 10 * 1024 * 1024:
                failed_files.append({
                    "filename": file.filename,
                    "reason": "File size exceeds 10MB"
                })
                continue
            
            # Process image
            result = process_image(image_bytes, file.filename)
            results_response.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            failed_files.append({
                "filename": file.filename,
                "reason": str(e)
            })
    
    # Prepare response
    response = {
        "count": len(results_response),
        "successful": len([r for r in results_response if r["status"] == "success"]),
        "failed": len(failed_files),
        "results": results_response
    }
    
    if failed_files:
        response["failed_files"] = failed_files
    
    return response

# ============================
# Model Info Endpoint
# ============================
@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "YOLOv8",
        "model_path": MODEL_PATH,
        "classes": model.names,
        "num_classes": len(model.names)
    }