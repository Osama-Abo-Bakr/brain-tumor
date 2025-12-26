# import streamlit as st
# import requests
# from PIL import Image, ImageDraw
# import io
# import pandas as pd

# # ==========================
# # App Configuration
# # ==========================
# st.set_page_config(
#     page_title="Brain Tumor Detection",
#     page_icon="🧠",
#     layout="wide"
# )

# st.title("🧠 Brain Tumor Detection System")
# st.markdown(
#     "Upload MRI images to detect brain tumors using **YOLOv8**. "
#     "Detected regions are highlighted with bounding boxes."
# )

# # ==========================
# # Backend API URL
# # ==========================
# # API_URL = "http://backend:8000/predict"  # Docker Compose ready
# # If running locally without Docker, use:
# API_URL = "http://localhost:8000/predict"

# # ==========================
# # File Upload
# # ==========================
# uploaded_files = st.file_uploader(
#     "Upload MRI Image(s)",
#     type=["jpg", "jpeg", "png"],
#     accept_multiple_files=True
# )

# # ==========================
# # Helper: Draw Bounding Boxes
# # ==========================
# def draw_bboxes(image, detections):
#     draw = ImageDraw.Draw(image, "RGBA")

#     for det in detections:
#         x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
#         label = f"{det['class_name']} ({det['confidence']:.2f})"

#         # High-contrast colors for MRI
#         box_color = (255, 255, 0, 255)   # Yellow
#         fill_color = (255, 255, 0, 60)   # Transparent yellow
#         text_bg = (0, 0, 0, 200)         # Black label background

#         # Draw filled bounding box
#         draw.rectangle(
#             [(x1, y1), (x2, y2)],
#             fill=fill_color,
#             outline=box_color,
#             width=5
#         )

#         # Label background
#         text_bbox = draw.textbbox((x1, y1), label)
#         text_w = text_bbox[2] - text_bbox[0]
#         text_h = text_bbox[3] - text_bbox[1]

#         draw.rectangle(
#             [(x1, y1 - text_h - 6), (x1 + text_w + 8, y1)],
#             fill=text_bg
#         )

#         # Label text
#         draw.text(
#             (x1 + 4, y1 - text_h - 4),
#             label,
#             fill=(255, 255, 255, 255)
#         )

#     return image

# # ==========================
# # Helper: Zoom Tumor Region
# # ==========================
# def crop_zoom(image, bbox, padding=40):
#     x1, y1, x2, y2 = map(int, bbox)
#     w, h = image.size

#     x1 = max(0, x1 - padding)
#     y1 = max(0, y1 - padding)
#     x2 = min(w, x2 + padding)
#     y2 = min(h, y2 + padding)

#     return image.crop((x1, y1, x2, y2))

# # ==========================
# # Predict Button
# # ==========================
# if st.button("🔍 Run Detection") and uploaded_files:
#     with st.spinner("Running detection..."):
#         files = [
#             ("files", (file.name, file.getvalue(), file.type))
#             for file in uploaded_files
#         ]

#         response = requests.post(API_URL, files=files)

#     if response.status_code == 200:
#         results = response.json()
#         st.success(f"Detection completed for {results['count']} image(s)")

#         # ==========================
#         # Display Results
#         # ==========================
#         for idx, result in enumerate(results["results"]):
#             st.subheader(f"📷 Image {idx + 1}: {result['filename']}")

#             # Load original image
#             original_image = Image.open(
#                 io.BytesIO(uploaded_files[idx].getvalue())
#             ).convert("RGB")

#             if result["detections"]:
#                 # Draw bounding boxes
#                 image_with_boxes = draw_bboxes(
#                     original_image.copy(),
#                     result["detections"]
#                 )

#                 col1, col2 = st.columns([2, 1])

#                 with col1:
#                     st.image(
#                         image_with_boxes,
#                         caption="Detection Result (Full Image)",
#                         use_column_width=True
#                     )

#                 with col2:
#                     zoom_img = crop_zoom(
#                         original_image,
#                         result["detections"][0]["bbox_xyxy"]
#                     )
#                     st.image(
#                         zoom_img,
#                         caption="🔍 Tumor Region (Zoomed)",
#                         use_column_width=True
#                     )

#                 # Show detection table
#                 df = pd.DataFrame(result["detections"])
#                 st.dataframe(df, use_container_width=True)

#             else:
#                 st.image(
#                     original_image,
#                     caption="No Tumors Detected",
#                     use_column_width=True
#                 )
#                 st.warning("No tumors detected in this image.")

#             st.divider()
#     else:
#         st.error("❌ Failed to connect to backend API.")



import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
import time

# ==========================
# App Configuration
# ==========================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Custom CSS Styling
# ==========================
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Upload section */
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        margin: 2rem 0;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Detection badge */
    .detection-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    
    .badge-positive {
        background: #ff4444;
        color: white;
    }
    
    .badge-negative {
        background: #00C851;
        color: white;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================
# Session State Initialization
# ==========================
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0

# ==========================
# Header
# ==========================
st.markdown("""
    <div class="header-container">
        <div class="header-title">🧠 Brain Tumor Detection System</div>
        <div class="header-subtitle">
            Advanced AI-powered MRI analysis using YOLOv8 deep learning model
        </div>
    </div>
""", unsafe_allow_html=True)

# ==========================
# Sidebar Configuration
# ==========================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Configuration
    use_docker = st.toggle("Docker Environment", value=False)
    if use_docker:
        API_URL = "http://backend:8000/predict"
    else:
        custom_url = st.text_input("API URL", "http://localhost:8000/predict")
        API_URL = custom_url
    
    st.divider()
    
    # Detection Settings
    st.subheader("Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detection"
    )
    
    show_zoom = st.checkbox("Show Zoomed Regions", value=True)
    show_details = st.checkbox("Show Detection Details", value=True)
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Session Statistics")
    st.metric("Images Processed", st.session_state.total_processed)
    st.metric("Tumors Detected", st.session_state.total_detections)
    
    if st.button("🗑️ Clear History"):
        st.session_state.results_history = []
        st.session_state.total_processed = 0
        st.session_state.total_detections = 0
        st.rerun()
    
    st.divider()
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Version:** 2.0
        
        **Model:** YOLOv8
        
        **Supported Formats:**
        - JPG/JPEG
        - PNG
        
        **Features:**
        - Multi-image batch processing
        - Real-time detection
        - Confidence scoring
        - Zoomed tumor regions
        - Session history
        """)

# ==========================
# Main Content Area
# ==========================

# Stats Cards Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-value">🎯</div>
            <div class="stat-label">High Accuracy</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-value">⚡</div>
            <div class="stat-label">Fast Processing</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-value">🔒</div>
            <div class="stat-label">Secure & Private</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="stat-card">
            <div class="stat-value">📊</div>
            <div class="stat-label">Detailed Analysis</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================
# File Upload Section
# ==========================
st.markdown("### 📤 Upload MRI Images")

uploaded_files = st.file_uploader(
    "Drag and drop files here or click to browse",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload one or more MRI images for tumor detection",
    label_visibility="collapsed"
)

if uploaded_files:
    st.info(f"✅ {len(uploaded_files)} file(s) selected")
    
    # Preview thumbnails
    if len(uploaded_files) <= 5:
        st.markdown("**Preview:**")
        cols = st.columns(min(len(uploaded_files), 5))
        for idx, (col, file) in enumerate(zip(cols, uploaded_files)):
            with col:
                img = Image.open(file)
                st.image(img, caption=file.name, use_column_width=True)

# ==========================
# Helper Functions
# ==========================
def draw_bboxes(image, detections, confidence_threshold=0.5):
    """Draw bounding boxes with improved styling"""
    draw = ImageDraw.Draw(image, "RGBA")

    for det in detections:
        if det['confidence'] < confidence_threshold:
            continue
            
        x1, y1, x2, y2 = map(int, det["bbox_xyxy"])
        label = f"{det['class_name']}"
        confidence = f"{det['confidence']:.1%}"

        # Color based on confidence
        if det['confidence'] >= 0.8:
            box_color = (255, 0, 0, 255)      # Red - High confidence
            fill_color = (255, 0, 0, 40)
        elif det['confidence'] >= 0.6:
            box_color = (255, 165, 0, 255)    # Orange - Medium confidence
            fill_color = (255, 165, 0, 40)
        else:
            box_color = (255, 255, 0, 255)    # Yellow - Low confidence
            fill_color = (255, 255, 0, 40)

        # Draw bounding box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            fill=fill_color,
            outline=box_color,
            width=4
        )

        # Label with confidence
        label_text = f"{label} {confidence}"
        
        # Calculate text size
        text_bbox = draw.textbbox((x1, y1), label_text)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Draw label background
        draw.rectangle(
            [(x1, y1 - text_h - 8), (x1 + text_w + 12, y1)],
            fill=(0, 0, 0, 220)
        )

        # Draw label text
        draw.text(
            (x1 + 6, y1 - text_h - 6),
            label_text,
            fill=(255, 255, 255, 255)
        )

    return image

def crop_zoom(image, bbox, padding=50):
    """Crop and zoom into detected region"""
    x1, y1, x2, y2 = map(int, bbox)
    w, h = image.size

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return image.crop((x1, y1, x2, y2))

def get_detection_summary(detections, confidence_threshold):
    """Generate detection summary"""
    filtered = [d for d in detections if d['confidence'] >= confidence_threshold]
    
    if not filtered:
        return "No tumors detected above confidence threshold", "negative"
    
    avg_conf = sum(d['confidence'] for d in filtered) / len(filtered)
    return f"{len(filtered)} tumor(s) detected (avg. confidence: {avg_conf:.1%})", "positive"

# ==========================
# Detection Button
# ==========================
if uploaded_files:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_detection = st.button(
            "🔍 Run Detection Analysis",
            type="primary",
            use_container_width=True
        )
else:
    run_detection = False

# ==========================
# Processing & Results
# ==========================
if run_detection and uploaded_files:
    progress_bar = st.progress(0, text="Initializing detection...")
    
    with st.spinner("Processing images..."):
        files = [
            ("files", (file.name, file.getvalue(), file.type))
            for file in uploaded_files
        ]
        
        progress_bar.progress(30, text="Sending to API...")
        
        try:
            response = requests.post(API_URL, files=files, timeout=60)
            progress_bar.progress(70, text="Processing results...")
            
            if response.status_code == 200:
                results = response.json()
                progress_bar.progress(100, text="Complete!")
                time.sleep(0.5)
                progress_bar.empty()
                
                # Update statistics
                st.session_state.total_processed += results['count']
                total_det = sum(len(r['detections']) for r in results['results'])
                st.session_state.total_detections += total_det
                
                st.success(f"✅ Successfully processed {results['count']} image(s)")
                
                # ==========================
                # Display Results
                # ==========================
                st.markdown("---")
                st.markdown("## 📊 Detection Results")
                
                for idx, result in enumerate(results["results"]):
                    with st.container():
                        st.markdown(f"### 📷 Image {idx + 1}: {result['filename']}")
                        
                        # Load original image
                        original_image = Image.open(
                            io.BytesIO(uploaded_files[idx].getvalue())
                        ).convert("RGB")
                        
                        # Get detection summary
                        summary, status = get_detection_summary(
                            result["detections"],
                            confidence_threshold
                        )
                        
                        # Status badge
                        badge_class = "badge-positive" if status == "positive" else "badge-negative"
                        st.markdown(
                            f'<span class="detection-badge {badge_class}">{summary}</span>',
                            unsafe_allow_html=True
                        )
                        
                        if result["detections"]:
                            # Filter by confidence
                            filtered_detections = [
                                d for d in result["detections"]
                                if d['confidence'] >= confidence_threshold
                            ]
                            
                            if filtered_detections:
                                # Draw bounding boxes
                                image_with_boxes = draw_bboxes(
                                    original_image.copy(),
                                    result["detections"],
                                    confidence_threshold
                                )
                                
                                if show_zoom:
                                    col1, col2 = st.columns([2, 1])
                                    
                                    with col1:
                                        st.image(
                                            image_with_boxes,
                                            caption="🔍 Detection Result (Full Image)",
                                            use_column_width=True
                                        )
                                    
                                    with col2:
                                        for det_idx, det in enumerate(filtered_detections[:3]):
                                            zoom_img = crop_zoom(
                                                original_image,
                                                det["bbox_xyxy"]
                                            )
                                            st.image(
                                                zoom_img,
                                                caption=f"Tumor #{det_idx + 1} (Confidence: {det['confidence']:.1%})",
                                                use_column_width=True
                                            )
                                else:
                                    st.image(
                                        image_with_boxes,
                                        caption="Detection Result",
                                        use_column_width=True
                                    )
                                
                                # Show detection details
                                if show_details:
                                    st.markdown("**Detection Details:**")
                                    df = pd.DataFrame(filtered_detections)
                                    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                                    df = df[['class_name', 'confidence', 'bbox_xyxy']]
                                    df.columns = ['Class', 'Confidence', 'Bounding Box (x1,y1,x2,y2)']
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                            else:
                                st.image(
                                    original_image,
                                    caption="No detections above confidence threshold",
                                    use_column_width=True
                                )
                        else:
                            st.image(
                                original_image,
                                caption="No Tumors Detected",
                                use_column_width=True
                            )
                        
                        st.markdown("---")
                        
            else:
                progress_bar.empty()
                st.error(f"❌ API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            progress_bar.empty()
            st.error("⏱️ Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            progress_bar.empty()
            st.error("❌ Failed to connect to backend API. Please check if the server is running.")
        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ An error occurred: {str(e)}")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Brain Tumor Detection System v2.0</strong></p>
        <p>For research and educational purposes only. Not for clinical diagnosis.</p>
    </div>
""", unsafe_allow_html=True)