import hashlib
import time
import uuid
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Dental-AI Workstation",
    layout="wide",
    page_icon="🦷",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --app-bg: var(--st-background-color);
        --app-panel: var(--st-secondary-background-color);
        --app-panel-soft: var(--st-secondary-background-color);
        --app-text: var(--st-text-color);
        --app-border: var(--st-border-color, rgba(120, 120, 120, 0.28));
        --app-border-soft: var(--st-border-color-light, var(--st-border-color, rgba(120, 120, 120, 0.18)));
        --app-primary: var(--st-primary-color);
        --app-shadow: 0 12px 28px rgba(15, 23, 42, 0.10);
        --app-shadow-strong: 0 18px 36px rgba(15, 23, 42, 0.14);
    }
    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Arial, sans-serif;
    }
    .stApp {
        background: var(--app-bg);
    }
    .block-container {
        max-width: 97%;
        padding-top: 1.1rem;
        padding-bottom: 1.6rem;
    }
    #MainMenu, footer, header {
        visibility: hidden;
    }
    .top-shell,
    .glass-card,
    .kpi-card,
    .viewer-shell,
    .section-shell,
    .thumb-card,
    .diag-card,
    .stMetric,
    [data-testid="stFileUploader"] {
        background: var(--app-panel);
        border: 1px solid var(--app-border-soft);
        box-shadow: var(--app-shadow);
    }
    .top-shell {
        border-radius: 24px;
        padding: 1.1rem 1.2rem;
    }
    .hero-title {
        color: var(--app-text);
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.0;
        margin: 0;
    }
    .hero-subtitle {
        color: var(--app-text);
        opacity: 0.72;
        font-size: 0.98rem;
        margin-top: 0.45rem;
    }
    .hero-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }
    .hero-chip {
        background: var(--app-panel-soft);
        border: 1px solid var(--app-border-soft);
        color: var(--app-text);
        border-radius: 999px;
        padding: 0.34rem 0.7rem;
        font-size: 0.80rem;
        font-weight: 700;
        opacity: 0.92;
    }
    .glass-card {
        border-radius: 20px;
        padding: 1rem;
        box-shadow: var(--app-shadow-strong);
        margin-bottom: 1rem;
    }
    .panel-title {
        color: var(--app-text);
        font-size: 1.1rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .panel-subtitle,
    .muted,
    .kpi-label,
    .diag-meta,
    .thumb-meta {
        color: var(--app-text);
        opacity: 0.72;
    }
    .panel-subtitle {
        font-size: 0.90rem;
    }
    .case-id {
        color: var(--app-text);
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.22rem;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.95rem 0 1rem 0;
    }
    .kpi-card {
        border-radius: 18px;
        padding: 0.95rem 1rem;
    }
    .kpi-label {
        font-size: 0.80rem;
        margin-bottom: 0.35rem;
    }
    .kpi-value,
    .diag-name,
    .diag-count,
    .thumb-title {
        color: var(--app-text);
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1.0;
    }
    .viewer-shell {
        border-radius: 24px;
        padding: 0.8rem;
        box-shadow: var(--app-shadow-strong);
    }
    .section-shell {
        border-radius: 18px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .muted {
        font-size: 0.88rem;
    }
    .divider {
        height: 1px;
        background: var(--app-border-soft);
        margin: 0.65rem 0 0.9rem 0;
    }
    .diag-card {
        border-radius: 16px;
        padding: 0.9rem;
        margin-top: 0.7rem;
    }
    .diag-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.65rem;
    }
    .diag-name {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.95rem;
        font-weight: 800;
    }
    .diag-dot {
        width: 11px;
        height: 11px;
        border-radius: 999px;
        display: inline-block;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.10);
    }
    .diag-count {
        font-size: 1.15rem;
        font-weight: 800;
    }
    .diag-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.55rem;
        font-size: 0.82rem;
    }
    .diag-bar {
        height: 7px;
        border-radius: 999px;
        background: var(--app-border-soft);
        overflow: hidden;
        margin-top: 0.55rem;
    }
    .diag-fill {
        height: 100%;
        border-radius: 999px;
    }
    .thumb-card {
        border-radius: 20px;
        padding: 0.7rem;
    }
    .thumb-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.55rem;
    }
    .thumb-title {
        font-size: 0.95rem;
        font-weight: 800;
    }
    .thumb-pill {
        border-radius: 999px;
        padding: 0.18rem 0.5rem;
        font-size: 0.74rem;
        font-weight: 800;
        color: #08111f;
    }
    .thumb-meta {
        font-size: 0.82rem;
        margin-top: 0.45rem;
        line-height: 1.45;
    }
    .suggestion-list {
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
        margin-top: 0.2rem;
    }
    .suggestion-item {
        background: var(--app-panel-soft);
        border: 1px solid var(--app-border-soft);
        border-radius: 14px;
        padding: 0.8rem 0.85rem;
        color: var(--app-text);
        font-size: 0.90rem;
        line-height: 1.5;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin-top: 0.65rem;
        background: var(--app-panel-soft);
        border: 1px solid var(--app-border-soft);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        color: var(--app-text);
        font-size: 0.82rem;
        font-weight: 700;
    }
    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        display: inline-block;
    }
    /* Base styling for all buttons */
    .stButton button,
    .stDownloadButton button,
    button[data-testid="stBaseButton-secondary"] {
        width: 100% !important;
        border: 1px solid var(--app-border) !important;
        border-radius: 14px !important;
        padding: 0.76rem 1rem !important;
        font-weight: 700 !important;
        background: #ffffff !important; 
        color: #1E293B !important; /* Dark navy text for readability */
        -webkit-text-fill-color: #1E293B !important;
        text-shadow: none !important;
        box-shadow: 0 4px 6px rgba(15, 23, 42, 0.05) !important;
        opacity: 1 !important;
        transition: all 0.2s ease !important;
    }

    /* Target all inner text elements for base buttons */
    .stButton button *,
    .stDownloadButton button *,
    button[data-testid="stBaseButton-secondary"] * {
        color: #1E293B !important;
        fill: #1E293B !important;
        stroke: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
        margin: 0 !important;
    }

    /* Specific styling for Primary buttons (e.g., Reset Session) */
    button[data-testid="stBaseButton-primary"] {
        width: 100% !important;
        border-radius: 14px !important;
        padding: 0.76rem 1rem !important;
        font-weight: 700 !important;
        background: #2563EB !important; /* Strong clinical blue */
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: 1px solid #1D4ED8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s ease !important;
    }

    button[data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        margin: 0 !important;
    }

    /* Hover States */
    .stButton button:hover,
    .stDownloadButton button:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        border-color: #94A3B8 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(15, 23, 42, 0.08) !important;
    }

    button[data-testid="stBaseButton-primary"]:hover {
        background: #1D4ED8 !important;
        border-color: #1E40AF !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.3) !important;
    }

    /* Disabled States */
    .stButton button:disabled,
    .stDownloadButton button:disabled,
    button[data-testid="stBaseButton-primary"]:disabled,
    button[data-testid="stBaseButton-secondary"]:disabled {
        background: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: none !important;
        transform: none !important;
        cursor: not-allowed !important;
    }

    .stButton button:disabled *,
    .stDownloadButton button:disabled *,
    button[data-testid="stBaseButton-primary"]:disabled *,
    button[data-testid="stBaseButton-secondary"]:disabled * {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
    }
    .stRadio > div, .stSlider, .stToggle, .stCheckbox,
    .stMarkdown, .stCaption, label {
        color: var(--app-text);
    }
    [data-testid="stFileUploader"] {
        border-radius: 16px;
        padding: 0.35rem 0.55rem 0.55rem 0.55rem;
    }
    [data-testid="stFileUploader"] section {
        border: 0;
        background: transparent;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--app-panel-soft);
        color: var(--app-text);
        border: 1px solid var(--app-border-soft);
        border-radius: 10px;
        padding: 0.35rem 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        border-color: var(--app-primary) !important;
        box-shadow: inset 0 0 0 1px var(--app-primary);
    }
    .stMetric {
        border-radius: 14px;
        padding: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(r"D:\Hard\Dental-Ai\dental_ai")
SEG_MODEL_PATH = PROJECT_ROOT / "models" / "tooth_segmentor_best.pt"
ID_MODEL_PATH = PROJECT_ROOT / "models" / "tooth_identity_32_best.pt"
PATHOLOGY_MODEL_PATH = PROJECT_ROOT / "runs" / "pathology" / "pathology_v12" / "weights" / "best.pt"

ANTERIOR_ORDER = ["CI", "LI", "C"]
POSTERIOR_TYPES = ["P1", "P2", "M1", "M2", "M3"]

TEAL_BGR = (181, 138, 42)
AMBER_BGR = (17, 158, 237)
WHITE_BGR = (248, 250, 252)
LIGHT_CARD_BGR = (245, 248, 252)
DARK_TEXT_BGR = (32, 48, 68)

PATHOLOGY_CLASS_NAMES = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical lesion",
}
PATHOLOGY_LABEL_TO_ID = {label: class_id for class_id, label in PATHOLOGY_CLASS_NAMES.items()}
PATHOLOGY_ID_TO_LABEL = {class_id: label for class_id, label in PATHOLOGY_CLASS_NAMES.items()}
PATHOLOGY_COLORS = {
    0: (235, 89, 84),
    1: (244, 198, 51),
    2: (91, 183, 255),
}
PATHOLOGY_HEX = {
    0: "#eb5954",
    1: "#f4c633",
    2: "#5bb7ff",
}
PATHOLOGY_THRESHOLDS = {
    0: 0.10,
    1: 0.10,
    2: 0.20,
}
MAX_DETECTIONS_PER_CLASS = {
    0: 8,
    1: 12,
    2: 4,
}

@st.cache_resource
def load_models():
    seg = YOLO(str(SEG_MODEL_PATH))
    ident = YOLO(str(ID_MODEL_PATH))
    pathology = YOLO(str(PATHOLOGY_MODEL_PATH)) if PATHOLOGY_MODEL_PATH.exists() else None
    return seg, ident, pathology

try:
    seg_model, id_model, pathology_model = load_models()
except Exception as exc:
    st.error(
        f"Failed to load models. Check these files: {SEG_MODEL_PATH}, {ID_MODEL_PATH}, {PATHOLOGY_MODEL_PATH}. Error: {exc}"
    )
    st.stop()

if "case_id" not in st.session_state:
    st.session_state.case_id = f"#{str(uuid.uuid4())[:4].upper()}-{str(uuid.uuid4())[:1].upper()}"
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = f"upload_{uuid.uuid4().hex}"
if "image_sig" not in st.session_state:
    st.session_state.image_sig = None
if "raw_img" not in st.session_state:
    st.session_state.raw_img = None
if "teeth_data" not in st.session_state:
    st.session_state.teeth_data = []
if "pathology_data" not in st.session_state:
    st.session_state.pathology_data = []
if "infer_time" not in st.session_state:
    st.session_state.infer_time = 0.0
if "pathology_time" not in st.session_state:
    st.session_state.pathology_time = 0.0
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "uploaded_size" not in st.session_state:
    st.session_state.uploaded_size = 0
if "render_png" not in st.session_state:
    st.session_state.render_png = None
if "pathology_debug" not in st.session_state:
    st.session_state.pathology_debug = {"model_loaded": pathology_model is not None, "raw_boxes": 0, "kept_boxes": 0}

def reset_session():
    st.session_state.case_id = f"#{str(uuid.uuid4())[:4].upper()}-{str(uuid.uuid4())[:1].upper()}"
    st.session_state.uploader_key = f"upload_{uuid.uuid4().hex}"
    st.session_state.image_sig = None
    st.session_state.raw_img = None
    st.session_state.teeth_data = []
    st.session_state.pathology_data = []
    st.session_state.infer_time = 0.0
    st.session_state.pathology_time = 0.0
    st.session_state.uploaded_name = None
    st.session_state.uploaded_size = 0
    st.session_state.render_png = None
    st.session_state.pathology_debug = {"model_loaded": pathology_model is not None, "raw_boxes": 0, "kept_boxes": 0}

def human_file_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"

def get_geometric_quadrant(cx, cy, img_w, img_h):
    is_upper = cy < (img_h / 2)
    is_patient_right = cx < (img_w / 2)
    if is_upper and is_patient_right:
        return "UR"
    if is_upper and not is_patient_right:
        return "UL"
    if not is_upper and is_patient_right:
        return "LR"
    return "LL"

def get_box_quadrant(box, img_w, img_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return get_geometric_quadrant(cx, cy, img_w, img_h)

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (
        (x2_1 - x1_1) * (y2_1 - y1_1)
        + (x2_2 - x1_2) * (y2_2 - y1_2)
        - inter_area
    )
    return inter_area / union_area if union_area > 0 else 0.0

def filter_overlapping_teeth(raw_teeth, iou_threshold=0.45):
    raw_teeth.sort(key=lambda x: x["seg_conf"], reverse=True)
    keep_teeth = []
    for current_tooth in raw_teeth:
        has_overlap = any(
            calculate_iou(current_tooth["box"], kept["box"]) > iou_threshold
            for kept in keep_teeth
        )
        if not has_overlap:
            keep_teeth.append(current_tooth)
    return keep_teeth

def refine_tooth_identities(teeth_data, img_w):
    quadrants = {"UR": [], "UL": [], "LR": [], "LL": []}
    for tooth in teeth_data:
        quadrants[tooth["quad"]].append(tooth)
    refined_teeth = []
    midline = img_w / 2
    for quad, teeth in quadrants.items():
        if not teeth:
            continue
        teeth.sort(key=lambda x: abs(x["cx"] - midline))
        for i, tooth in enumerate(teeth):
            base_type = tooth["raw_type"]
            if i == 0:
                final_type = "CI"
            elif i == 1:
                final_type = "LI"
            elif i == 2:
                final_type = "C"
            else:
                final_type = base_type.split("-")[1] if "-" in base_type else base_type
            if final_type in ANTERIOR_ORDER:
                final_type = POSTERIOR_TYPES[min(i - 3, len(POSTERIOR_TYPES) - 1)]
            tooth["final_type"] = final_type
            tooth["final_label"] = f"{quad}-{final_type}"
            refined_teeth.append(tooth)
    return refined_teeth

def process_image(raw_img):
    start_time = time.time()
    img_h, img_w = raw_img.shape[:2]
    seg_results = seg_model(raw_img, verbose=False)[0]
    if seg_results.masks is None:
        return [], 0.0
    masks = seg_results.masks.data.cpu().numpy()
    boxes = seg_results.boxes.xyxy.cpu().numpy()
    seg_confs = seg_results.boxes.conf.cpu().numpy()
    raw_teeth = []
    for mask, box, seg_conf in zip(masks, boxes, seg_confs):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        geom_quad = get_geometric_quadrant(cx, cy, img_w, img_h)
        mask_resized = cv2.resize(mask, (img_w, img_h))
        tooth_mask = mask_resized[y1:y2, x1:x2]
        crop = raw_img[y1:y2, x1:x2].copy()
        crop[tooth_mask < 0.5] = 0
        id_results = id_model(crop, verbose=False)[0]
        raw_teeth.append(
            {
                "cx": cx,
                "cy": cy,
                "box": (x1, y1, x2, y2),
                "quad": geom_quad,
                "raw_type": id_model.names[id_results.probs.top1],
                "id_conf": float(id_results.probs.top1conf.cpu().numpy()),
                "seg_conf": float(seg_conf),
                "mask": mask_resized,
            }
        )
    filtered_teeth = filter_overlapping_teeth(raw_teeth)
    img_cy = img_h / 2
    upper_teeth = sorted(
        [tooth for tooth in filtered_teeth if tooth["cy"] < img_cy],
        key=lambda tooth: tooth["cx"],
    )
    lower_teeth = sorted(
        [tooth for tooth in filtered_teeth if tooth["cy"] >= img_cy],
        key=lambda tooth: tooth["cx"],
        reverse=True,
    )
    ordered_teeth = upper_teeth + lower_teeth
    for idx, tooth in enumerate(ordered_teeth, start=1):
        tooth["id"] = idx
    infer_time = time.time() - start_time
    return refine_tooth_identities(ordered_teeth, img_w), infer_time

def detect_pathology(raw_img):
    if pathology_model is None:
        return [], {"model_loaded": False, "raw_boxes": 0, "kept_boxes": 0}, 0.0
    start_time = time.time()
    min_conf = min(PATHOLOGY_THRESHOLDS.values())
    results = pathology_model.predict(
        source=raw_img,
        imgsz=1024,
        conf=min_conf,
        iou=0.45,
        max_det=50,
        verbose=False,
    )
    if not results:
        return [], {"model_loaded": True, "raw_boxes": 0, "kept_boxes": 0}, time.time() - start_time
    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return [], {"model_loaded": True, "raw_boxes": 0, "kept_boxes": 0}, time.time() - start_time
    img_h, img_w = raw_img.shape[:2]
    detections = []
    raw_boxes = len(boxes_obj)
    for box in boxes_obj:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        threshold = PATHOLOGY_THRESHOLDS.get(cls_id, 0.10)
        if conf < threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "class_id": cls_id,
                "label": PATHOLOGY_CLASS_NAMES.get(cls_id, f"Class {cls_id}"),
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
            }
        )
    detections.sort(key=lambda d: d["conf"], reverse=True)
    grouped = defaultdict(list)
    for det in detections:
        grouped[det["class_id"]].append(det)
    final_detections = []
    for cls_id, dets in grouped.items():
        final_detections.extend(dets[: MAX_DETECTIONS_PER_CLASS.get(cls_id, len(dets))])
    final_detections.sort(key=lambda d: d["conf"], reverse=True)
    return final_detections, {
        "model_loaded": True,
        "raw_boxes": raw_boxes,
        "kept_boxes": len(final_detections),
    }, time.time() - start_time

def summarize_pathology(detections):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det["label"]].append(det)
    summary = []
    for label, items in grouped.items():
        items = sorted(items, key=lambda d: d["conf"], reverse=True)
        summary.append(
            {
                "label": label,
                "count": len(items),
                "top_conf": items[0]["conf"],
                "detections": items,
            }
        )
    summary.sort(key=lambda item: item["top_conf"], reverse=True)
    return summary

def filter_pathology_by_selection(detections, selected_labels):
    if not selected_labels:
        return []
    selected_ids = {
        PATHOLOGY_LABEL_TO_ID[label]
        for label in selected_labels
        if label in PATHOLOGY_LABEL_TO_ID
    }
    return [det for det in detections if det["class_id"] in selected_ids]

def build_findings_df(teeth_data, conf_thresh):
    rows = []
    for tooth in sorted(teeth_data, key=lambda t: t["id"]):
        rows.append(
            {
                "ID": tooth["id"],
                "Label": tooth["final_label"],
                "Quad": tooth["quad"],
                "Type": tooth["final_type"],
                "ID Conf": round(tooth["id_conf"], 2),
                "Seg Conf": round(tooth["seg_conf"], 2),
                "Status": "Review" if tooth["id_conf"] < conf_thresh else "OK",
            }
        )
    return pd.DataFrame(rows)

def build_pathology_df(pathology_data, img_w, img_h):
    rows = []
    for idx, det in enumerate(sorted(pathology_data, key=lambda d: d["conf"], reverse=True), start=1):
        x1, y1, x2, y2 = det["bbox"]
        rows.append(
            {
                "#": idx,
                "Finding": det["label"],
                "Quad": get_box_quadrant(det["bbox"], img_w, img_h),
                "Confidence": round(det["conf"], 2),
                "Box": f"[{x1}, {y1}, {x2}, {y2}]",
            }
        )
    return pd.DataFrame(rows)

def build_doctor_suggestions(teeth_data, pathology_data, visible_pathology_count, review_count, selected_quadrant):
    suggestions = []
    labels_present = {det["label"] for det in pathology_data}
    if "Impacted" in labels_present:
        suggestions.append(
            "Review eruption path, angulation, and distal contact on the adjacent molar before final planning."
        )
    if "Caries" in labels_present:
        suggestions.append(
            "Correlate suspected caries with clinical exam or bitewing views before restorative treatment planning."
        )
    if "Periapical lesion" in labels_present:
        suggestions.append(
            "Check vitality, percussion, and endodontic history for teeth with apical change, then confirm with targeted imaging if needed."
        )
    if review_count > 0:
        suggestions.append(
            f"Manually verify {review_count} tooth label(s) flagged for review before final charting."
        )
    if len(teeth_data) < 28:
        suggestions.append(
            "Detected tooth count is reduced. Review for missing, unerupted, impacted, or partially captured teeth."
        )
    if selected_quadrant != "All" and visible_pathology_count > 0:
        suggestions.append(
            f"Current focus is {selected_quadrant}. Confirm the visible findings match the clinical complaint and side of interest."
        )
    if not suggestions:
        suggestions.append(
            "No major AI findings are highlighted with the current filter. Continue routine radiographic review and clinical correlation."
        )
    return suggestions[:4]

def quadrant_counts(teeth_data):
    return {quad: sum(1 for tooth in teeth_data if tooth["quad"] == quad) for quad in ["UR", "UL", "LL", "LR"]}

def build_map_lines(teeth_data, conf_thresh, view_mode):
    counts = quadrant_counts(teeth_data)
    review_counts = {
        quad: sum(1 for tooth in teeth_data if tooth["quad"] == quad and tooth["id_conf"] < conf_thresh)
        for quad in ["UR", "UL", "LL", "LR"]
    }
    if view_mode == "Presentation Mode":
        return [
            f"UR  {counts['UR']} teeth   {review_counts['UR']} review",
            f"UL  {counts['UL']} teeth   {review_counts['UL']} review",
            f"LL  {counts['LL']} teeth   {review_counts['LL']} review",
            f"LR  {counts['LR']} teeth   {review_counts['LR']} review",
        ]
    lines = []
    for quad in ["UR", "UL", "LL", "LR"]:
        quad_teeth = sorted([t for t in teeth_data if t["quad"] == quad], key=lambda t: t["id"])
        short = " ".join(t["final_type"] for t in quad_teeth[:6]) if quad_teeth else "--"
        lines.append(f"{quad}  {counts[quad]} teeth  |  {short}")
    return lines

def draw_overlay_card(img, x, y, width, title, lines, accent=(31, 162, 225), alpha=0.84):
    line_gap = 18
    title_gap = 20
    pad_x = 12
    height = 16 + title_gap + max(len(lines), 1) * line_gap + 10
    x = max(0, min(x, img.shape[1] - width - 2))
    y = max(0, min(y, img.shape[0] - height - 2))
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), LIGHT_CARD_BGR, -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + 4), accent, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + width, y + height), (202, 216, 231), 1)
    cv2.putText(
        img,
        title,
        (x + pad_x, y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        DARK_TEXT_BGR,
        1,
        cv2.LINE_AA,
    )
    cursor_y = y + 24 + title_gap
    for line in lines:
        cv2.putText(
            img,
            line,
            (x + pad_x, cursor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            DARK_TEXT_BGR,
            1,
            cv2.LINE_AA,
        )
        cursor_y += line_gap

def encode_png(img_bgr):
    ok, buffer = cv2.imencode(".png", img_bgr)
    return buffer.tobytes() if ok else None

def build_focus_counts(teeth_data, pathology_data, selected_quadrant):
    if selected_quadrant == "All":
        return len(teeth_data), len(pathology_data)
    tooth_count = sum(1 for tooth in teeth_data if tooth["quad"] == selected_quadrant)
    pathology_count = sum(1 for det in pathology_data if det.get("quad") == selected_quadrant)
    return tooth_count, pathology_count

def attach_pathology_quadrants(pathology_data, img_w, img_h):
    for det in pathology_data:
        det["quad"] = get_box_quadrant(det["bbox"], img_w, img_h)
    return pathology_data

def get_preview_cards(raw_img, pathology_detections, padding=28):
    if raw_img is None:
        return []
    previews = []
    img_h, img_w = raw_img.shape[:2]
    grouped = defaultdict(list)
    for det in pathology_detections:
        grouped[det["class_id"]].append(det)
    for class_id in sorted(PATHOLOGY_CLASS_NAMES.keys()):
        label = PATHOLOGY_CLASS_NAMES[class_id]
        dets = sorted(grouped.get(class_id, []), key=lambda d: d["conf"], reverse=True)
        items = []
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(img_w, x2 + padding)
            py2 = min(img_h, y2 + padding)
            crop = raw_img[py1:py2, px1:px2].copy()
            bx1, by1, bx2, by2 = x1 - px1, y1 - py1, x2 - px1, y2 - py1
            color = PATHOLOGY_COLORS.get(class_id, (0, 255, 255))
            cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, 2)
            items.append({
                "image": crop,
                "quad": det.get("quad", "--"),
                "conf": det["conf"],
                "bbox": det["bbox"],
            })
        previews.append({
            "label": label,
            "class_id": class_id,
            "items": items,
            "count": len(items),
        })
    return previews

def render_diagnosis_cards(pathology_summary):
    summary_map = {item["label"]: item for item in pathology_summary}
    html_parts = []
    for class_id in sorted(PATHOLOGY_CLASS_NAMES.keys()):
        label = PATHOLOGY_CLASS_NAMES[class_id]
        color = PATHOLOGY_HEX[class_id]
        item = summary_map.get(label)
        count = item["count"] if item else 0
        top_conf = item["top_conf"] if item else 0.0
        fill = max(6, int(top_conf * 100)) if count > 0 else 6
        html_parts.append(
            f"""
            <div class="diag-card">
            <div class="diag-top">
            <div class="diag-name"><span class="diag-dot" style="background:{color};"></span>{label}</div>
            <div class="diag-count">{count}</div>
            </div>
            <div class="diag-meta">
            <span>Top confidence</span>
            <span>{top_conf:.2f}</span>
            </div>
            <div class="diag-bar"><div class="diag-fill" style="width:{fill}%; background:{color};"></div></div>
            </div>
            """
        )
    st.markdown("".join(html_parts), unsafe_allow_html=True)

def render_suggestions(suggestions):
    if not suggestions:
        st.markdown('<div class="muted">No case specific suggestions available.</div>', unsafe_allow_html=True)
        return
    html = ['<div class="suggestion-list">']
    for item in suggestions:
        html.append(f'<div class="suggestion-item">{item}</div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)

def render_preview_strip(previews):
    cols = st.columns(3)
    for col, preview in zip(cols, previews):
        with col:
            color = PATHOLOGY_HEX[preview["class_id"]]
            badge_text = preview["label"] if preview["count"] > 0 else "No finding"
            st.markdown(
                f"""
                <div class="thumb-card">
                <div class="thumb-head">
                <div class="thumb-title">{preview['label']}</div>
                <div class="thumb-pill" style="background:{color};">{badge_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if preview["count"] == 0:
                placeholder = np.full((220, 320, 3), 18, dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    "No visible finding",
                    (55, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (170, 190, 210),
                    2,
                    cv2.LINE_AA,
                )
                st.image(placeholder, channels="BGR", use_container_width=True)
                meta = "No active detection for the selected filters."
                st.markdown(f'<div class="thumb-meta">{meta}</div></div>', unsafe_allow_html=True)
                continue
            idx_key = f"preview_idx_{preview['class_id']}"
            max_index = preview["count"] - 1
            if idx_key not in st.session_state:
                st.session_state[idx_key] = 0
            st.session_state[idx_key] = max(0, min(st.session_state[idx_key], max_index))
            selected_num = st.select_slider(
                "Browse detections",
                options=list(range(1, preview["count"] + 1)),
                value=st.session_state[idx_key] + 1,
                key=f"preview_slider_{preview['class_id']}",
                label_visibility="collapsed",
            )
            st.session_state[idx_key] = selected_num - 1
            current = preview["items"][st.session_state[idx_key]]
            st.image(current["image"], channels="BGR", use_container_width=True)
            meta = (
                f"Finding {st.session_state[idx_key] + 1} of {preview['count']}<br>"
                f"Quad {current['quad']}<br>"
                f"Confidence {current['conf']:.2f}"
            )
            st.markdown(f'<div class="thumb-meta">{meta}</div></div>', unsafe_allow_html=True)

def render_image(
    raw_img,
    teeth_data,
    pathology_data,
    case_id,
    infer_time,
    pathology_time,
    selected_quadrant,
    view_mode,
    overlay_mode,
    show_masks,
    show_boxes,
    show_full_labels,
    mask_opacity,
    conf_thresh,
    show_pathology_conf,
):
    display_img = raw_img.copy()
    img_h, img_w = display_img.shape[:2]
    if selected_quadrant != "All":
        display_img = cv2.convertScaleAbs(display_img, alpha=0.42, beta=0)
    low_conf_count = sum(1 for tooth in teeth_data if tooth["id_conf"] < conf_thresh)
    tooth_focus_count, pathology_focus_count = build_focus_counts(teeth_data, pathology_data, selected_quadrant)
    draw_teeth = overlay_mode in ["Full Overlay", "Teeth Only"]
    draw_pathology = overlay_mode in ["Full Overlay", "Pathology Only"]
    if draw_teeth:
        mask_overlay = np.zeros_like(display_img, dtype=np.uint8)
        for tooth in teeth_data:
            quad = tooth["quad"]
            if selected_quadrant != "All" and quad != selected_quadrant:
                continue
            x1, y1, x2, y2 = tooth["box"]
            is_low = tooth["id_conf"] < conf_thresh
            color = AMBER_BGR if is_low else TEAL_BGR
            mask_binary = (tooth["mask"] > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if show_masks:
                mask_overlay[mask_binary > 0] = color
            cv2.drawContours(display_img, contours, -1, color, 1 if view_mode == "Presentation Mode" else 2)
            if show_boxes:
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 1)
            if view_mode == "Presentation Mode":
                label = str(tooth["id"])
                font_scale = 0.34
            else:
                label = tooth["final_label"] if show_full_labels else str(tooth["id"])
                font_scale = 0.36 if len(label) < 9 else 0.31
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            label_x = max(4, min(x1, img_w - tw - 8))
            label_y = max(th + 8, y1 - 6)
            cv2.rectangle(
                display_img,
                (label_x - 3, label_y - th - 5),
                (label_x + tw + 5, label_y + 3),
                (32, 38, 47),
                -1,
            )
            cv2.putText(
                display_img,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                WHITE_BGR,
                1,
                cv2.LINE_AA,
            )
        if show_masks:
            cv2.addWeighted(mask_overlay, mask_opacity, display_img, 1.0, 0, display_img)
    if draw_pathology:
        for det in pathology_data:
            quad = det.get("quad", "All")
            if selected_quadrant != "All" and quad != selected_quadrant:
                continue
            cls_id = det["class_id"]
            color = PATHOLOGY_COLORS.get(cls_id, (0, 255, 255))
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2 if view_mode == "Presentation Mode" else 3)
            label = det["label"] if not show_pathology_conf else f"{det['label']} {det['conf']:.2f}"
            font_scale = 0.54 if len(label) < 18 else 0.46
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            text_x = x1
            text_y = max(y1 - 10, th + 10)
            cv2.rectangle(
                display_img,
                (text_x, text_y - th - 10),
                (text_x + tw + 10, text_y + base - 2),
                color,
                -1,
            )
            cv2.putText(
                display_img,
                label,
                (text_x + 5, text_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (9, 18, 28),
                2,
                cv2.LINE_AA,
            )
    focus_label = selected_quadrant if selected_quadrant != "All" else "Full Arch"
    draw_overlay_card(
        display_img,
        18,
        18,
        238,
        "Case Overview",
        [
            f"Case  {case_id}",
            f"Focus  {focus_label}",
            f"View  {overlay_mode}",
        ],
        accent=(39, 181, 255),
    )
    draw_overlay_card(
        display_img,
        img_w - 315,
        18,
        295,
        "Tooth Map",
        build_map_lines(teeth_data, conf_thresh, view_mode),
        accent=(91, 183, 255),
    )
    draw_overlay_card(
        display_img,
        18,
        img_h - 112,
        275,
        "Clinical Summary",
        [
            f"Visible teeth  {tooth_focus_count}",
            f"Visible findings  {pathology_focus_count}",
            f"Teeth AI  {infer_time:.2f}s",
            f"Pathology AI  {pathology_time:.2f}s",
        ],
        accent=(244, 198, 51),
    )
    return display_img, low_conf_count

with st.container():
    left, mid, btn1, btn2 = st.columns([5.5, 1.8, 1.4, 1.6])
    with left:
        st.markdown(
            f"""
            <div class="top-shell">
            <div class="hero-title">Dental-AI Workstation</div>
            <div class="hero-subtitle">Dark dashboard UI for tooth mapping and three diagnosis review.</div>
            <div class="hero-chip-row">
            <div class="hero-chip">Impacted</div>
            <div class="hero-chip">Caries</div>
            <div class="hero-chip">Periapical lesion</div>
            <div class="hero-chip">Panoramic workflow</div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mid:
        st.markdown(
            f"""
            <div class="glass-card">
            <div class="panel-subtitle">Case ID</div>
            <div class="case-id">{st.session_state.case_id}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with btn1:
        st.markdown("<div style='height:1.0rem'></div>", unsafe_allow_html=True)
        if st.button("Reset Session", key="reset_session_btn", type="primary", use_container_width=True):
            reset_session()
            st.rerun()
    with btn2:
        st.markdown("<div style='height:1.0rem'></div>", unsafe_allow_html=True)
        export_csv = None
        if st.session_state.pathology_data:
            raw_img_for_export = st.session_state.raw_img
            if raw_img_for_export is not None:
                export_df = build_pathology_df(st.session_state.pathology_data, raw_img_for_export.shape[1], raw_img_for_export.shape[0])
                export_csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Pathology CSV",
            data=export_csv or b"",
            file_name=f"dental_ai_pathology_{st.session_state.case_id.replace('#', '').replace('-', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not bool(export_csv),
        )

controls_left, controls_mid, controls_right = st.columns([3.4, 3.2, 3.4])
with controls_left:
    selected_quadrant = st.radio("Focus Area", ["All", "UR", "UL", "LL", "LR"], horizontal=True)
with controls_mid:
    view_mode = st.radio("Interface Mode", ["Presentation Mode", "Review Mode"], horizontal=True)
with controls_right:
    overlay_mode = st.radio("Overlay View", ["Full Overlay", "Pathology Only", "Teeth Only", "Original"], horizontal=True)

show_masks = view_mode == "Review Mode"
show_boxes = False
show_full_labels = view_mode == "Review Mode"
mask_opacity = 0.22
conf_thresh = 0.60
show_pathology_conf = view_mode == "Review Mode"

main_left, main_right = st.columns([7.8, 2.2])
with main_right:
    st.markdown(
        """
        <div class="glass-card">
        <div class="panel-title">Case Input</div>
        <div class="panel-subtitle">Upload one panoramic radiograph for live review.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload Radiograph",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=st.session_state.uploader_key,
    )
    if st.session_state.uploaded_name:
        st.markdown(
            f"""
            <div class="glass-card">
            <div class="panel-title">Loaded Image</div>
            <div class="panel-subtitle">{st.session_state.uploaded_name}</div>
            <div class="muted">{human_file_size(st.session_state.uploaded_size)} uploaded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    detector_ready = pathology_model is not None
    status_color = "#28c76f" if detector_ready else "#eb5757"
    status_label = "Detector ready" if detector_ready else "Detector unavailable"
    st.markdown(
        f"""
        <div class="glass-card">
        <div class="panel-title">Diagnosis Rail</div>
        <div class="panel-subtitle">Three active diagnosis classes and confidence preview.</div>
        <div class="status-badge"><span class="status-dot" style="background:{status_color};"></span>{status_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Pathology Filter</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Turn each diagnosis on or off.</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    show_impacted = st.toggle("Show impacted", value=True)
    show_caries = st.toggle("Show caries", value=True)
    show_periapical = st.toggle("Show periapical lesion", value=True)
    selected_pathology_labels = []
    if show_impacted:
        selected_pathology_labels.append("Impacted")
    if show_caries:
        selected_pathology_labels.append("Caries")
    if show_periapical:
        selected_pathology_labels.append("Periapical lesion")
    st.markdown('</div>', unsafe_allow_html=True)
    if view_mode == "Review Mode":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Viewer Controls</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-subtitle">Review overlays and confidence thresholds.</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        show_masks = st.toggle("Show tooth masks", value=True)
        show_boxes = st.toggle("Show tooth boxes", value=False)
        show_full_labels = st.toggle("Show full tooth labels", value=True)
        show_pathology_conf = st.toggle("Show pathology confidence", value=True)
        mask_opacity = st.slider("Mask opacity", 0.10, 0.70, 0.22, 0.01)
        conf_thresh = st.slider("Tooth warning threshold", 0.10, 0.95, 0.60, 0.01)
        st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    image_sig = hashlib.md5(file_bytes).hexdigest()
    if st.session_state.image_sig != image_sig:
        file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
        raw_img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
        if raw_img is None:
            st.error("Could not decode the uploaded image.")
            st.stop()
        with st.spinner("Processing radiograph"):
            teeth_data, infer_time = process_image(raw_img)
            pathology_data, pathology_debug, pathology_time = detect_pathology(raw_img.copy())
            pathology_data = attach_pathology_quadrants(pathology_data, raw_img.shape[1], raw_img.shape[0])
            st.session_state.image_sig = image_sig
            st.session_state.raw_img = raw_img
            st.session_state.teeth_data = teeth_data
            st.session_state.pathology_data = pathology_data
            st.session_state.pathology_debug = pathology_debug
            st.session_state.infer_time = infer_time
            st.session_state.pathology_time = pathology_time
            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.uploaded_size = len(file_bytes)
            
    raw_img = st.session_state.raw_img
    teeth_data = st.session_state.teeth_data or []
    pathology_data = st.session_state.pathology_data or []
    pathology_debug = st.session_state.pathology_debug
    infer_time = st.session_state.infer_time
    pathology_time = st.session_state.pathology_time
    
    if raw_img is not None:
        total_detected = len(teeth_data)
        review_count = sum(1 for tooth in teeth_data if tooth["id_conf"] < conf_thresh)
        high_conf_count = max(total_detected - review_count, 0)
        filtered_pathology = filter_pathology_by_selection(pathology_data, selected_pathology_labels)
        if selected_quadrant == "All":
            visible_pathology = filtered_pathology
        else:
            visible_pathology = [det for det in filtered_pathology if det.get("quad") == selected_quadrant]
        focus_tooth_count, visible_pathology_count = build_focus_counts(teeth_data, visible_pathology, selected_quadrant)
        pathology_summary = summarize_pathology(visible_pathology)
        doctor_suggestions = build_doctor_suggestions(
            teeth_data=teeth_data,
            pathology_data=visible_pathology,
            visible_pathology_count=visible_pathology_count,
            review_count=review_count,
            selected_quadrant=selected_quadrant,
        )
        preview_cards = get_preview_cards(raw_img, visible_pathology)
        st.markdown(
            f"""
            <div class="kpi-grid">
            <div class="kpi-card"><div class="kpi-label">Detected teeth</div><div class="kpi-value">{total_detected}</div></div>
            <div class="kpi-card"><div class="kpi-label">High confidence</div><div class="kpi-value">{high_conf_count}</div></div>
            <div class="kpi-card"><div class="kpi-label">Needs review</div><div class="kpi-value">{review_count}</div></div>
            <div class="kpi-card"><div class="kpi-label">Visible findings</div><div class="kpi-value">{visible_pathology_count}</div></div>
            <div class="kpi-card"><div class="kpi-label">Pathology AI</div><div class="kpi-value">{pathology_time:.2f}s</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with main_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Diagnosis Overview</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel-subtitle">Focus {selected_quadrant}. Teeth {focus_tooth_count}. Findings {visible_pathology_count}.</div>',
                unsafe_allow_html=True,
            )
            render_diagnosis_cards(pathology_summary)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">AI Clinical Suggestions</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-subtitle">Case specific review points from visible findings.</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            render_suggestions(doctor_suggestions)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Export Options</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-subtitle">Download the annotated image or pathology table.</div>', unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            png_data = st.session_state.render_png or b""
            st.download_button(
                "Download annotated PNG",
                data=png_data,
                file_name=f"annotated_{st.session_state.case_id.replace('#', '').replace('-', '_')}.png",
                mime="image/png",
                use_container_width=True,
                disabled=st.session_state.render_png is None,
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        rendered_img, low_conf_count = render_image(
            raw_img=raw_img,
            teeth_data=teeth_data,
            pathology_data=visible_pathology,
            case_id=st.session_state.case_id,
            infer_time=infer_time,
            pathology_time=pathology_time,
            selected_quadrant=selected_quadrant,
            view_mode=view_mode,
            overlay_mode=overlay_mode,
            show_masks=show_masks,
            show_boxes=show_boxes,
            show_full_labels=show_full_labels,
            mask_opacity=mask_opacity,
            conf_thresh=conf_thresh,
            show_pathology_conf=show_pathology_conf,
        )
        st.session_state.render_png = encode_png(rendered_img)
        
        with main_left:
            st.markdown('<div class="viewer-shell">', unsafe_allow_html=True)
            st.image(rendered_img, channels="BGR", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="section-shell">
                <div class="panel-title">Finding Preview Strip</div>
                <div class="panel-subtitle">Browse every visible detection for each diagnosis class, one by one.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_preview_strip(preview_cards)
            
            if view_mode == "Review Mode":
                findings_df = build_findings_df(teeth_data, conf_thresh)
                pathology_df = build_pathology_df(visible_pathology, raw_img.shape[1], raw_img.shape[0])
                st.markdown(
                    """
                    <div class="section-shell">
                    <div class="panel-title">Detailed Review</div>
                    <div class="panel-subtitle">Teeth inventory, quadrant view, pathology detections, and system checks.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                metric_left, metric_mid, metric_right = st.columns(3)
                with metric_left:
                    st.metric("Focus teeth", focus_tooth_count)
                with metric_mid:
                    st.metric("Needs review", low_conf_count)
                with metric_right:
                    st.metric("Pathology findings", visible_pathology_count)
                tabs = st.tabs(["Overview", "UR", "UL", "LL", "LR", "Pathology", "System"])
                with tabs[0]:
                    overview_df = findings_df.copy()
                    if selected_quadrant != "All":
                        overview_df = overview_df[overview_df["Quad"] == selected_quadrant]
                    if overview_df.empty:
                        st.info("No tooth rows available for the current focus.")
                    else:
                        st.dataframe(overview_df, use_container_width=True, hide_index=True, height=340)
                for tab, quad in zip(tabs[1:5], ["UR", "UL", "LL", "LR"]):
                    with tab:
                        quad_df = findings_df[findings_df["Quad"] == quad]
                        if quad_df.empty:
                            st.info(f"No detections for {quad}.")
                        else:
                            st.dataframe(quad_df, use_container_width=True, hide_index=True, height=340)
                with tabs[5]:
                    if pathology_df.empty:
                        st.info("No pathology detections match the current filters.")
                    else:
                        st.dataframe(pathology_df, use_container_width=True, hide_index=True, height=340)
                with tabs[6]:
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Detector", "Ready" if pathology_debug.get("model_loaded") else "Unavailable")
                    with c2:
                        st.metric("Raw boxes", pathology_debug.get("raw_boxes", 0))
                    with c3:
                        st.metric("Kept boxes", pathology_debug.get("kept_boxes", 0))
                    with c4:
                        st.metric("Visible", visible_pathology_count)
                    st.caption("Version: pathology_v12")
                    st.caption(
                        f"Thresholds: impacted={PATHOLOGY_THRESHOLDS[0]}, caries={PATHOLOGY_THRESHOLDS[1]}, periapical={PATHOLOGY_THRESHOLDS[2]}"
                    )
else:
    with main_left:
        empty_img = np.full((720, 1280, 3), 16, dtype=np.uint8)
        cv2.putText(
            empty_img,
            "Awaiting panoramic radiograph upload",
            (260, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.02,
            (190, 205, 225),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            empty_img,
            "The dark viewer, diagnosis rail, and preview cards will appear here.",
            (220, 395),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (135, 155, 180),
            2,
            cv2.LINE_AA,
        )
        st.markdown('<div class="viewer-shell">', unsafe_allow_html=True)
        st.image(empty_img, channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-shell">
            <div class="panel-title">Finding Preview Strip</div>
            <div class="panel-subtitle">Upload a radiograph to show top crops for impacted, caries, and periapical lesion.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_preview_strip(get_preview_cards(None, []))
        with main_right:
            st.markdown(
                """
                <div class="glass-card">
                <div class="panel-title">Diagnosis Overview</div>
                <div class="panel-subtitle">Upload an image to populate the three diagnosis cards.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_diagnosis_cards([])
            st.markdown(
                """
                <div class="glass-card">
                <div class="panel-title">AI Clinical Suggestions</div>
                <div class="panel-subtitle">Case specific review points will appear after inference.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )