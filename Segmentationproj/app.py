import streamlit as st
import cv2
import numpy as np
import joblib

from feature_pipeline import extract_features, isolate_leaf

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Leaf And Fruits Disease Detection System",
    layout="wide"
)

LOWER_HSV = np.array([10, 50, 20])
UPPER_HSV = np.array([30, 255, 255])
KERNEL = np.ones((5, 5), np.uint8)
CLASS_CONFIDENCE_THRESHOLD = 70.0
SAMPLE_TYPE_CLASS_CONFIDENCE_THRESHOLD = 40.0
FRUIT_CLASS_TOKENS = ("mango", "apple", "orange", "banana", "grape", "fruit")
LEAF_CLASS_TOKENS = ("leaf",)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    artifact = joblib.load("model.pkl")

    if isinstance(artifact, dict) and "model" in artifact:
        return (
            artifact["model"],
            artifact.get("class_names", ["Healthy", "Diseased"]),
            set(artifact.get("healthy_labels", ["Healthy"])),
        )

    model = artifact
    scaler = joblib.load("scaler.pkl")
    return (model, scaler), ["Healthy", "Diseased"], {"Healthy"}

model_bundle, class_names, healthy_labels = load_model()

# =========================
# FUNCTIONS
# =========================
def segment_leaf(img):
    img = cv2.resize(img, (400, 400))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    affected_percent = (np.count_nonzero(mask) / mask.size) * 100

    result = img.copy()
    result[mask > 0] = (0, 0, 255)

    return img, mask, result, affected_percent


def prepare_leaf_focus(img):
    isolated_leaf, leaf_mask = isolate_leaf(img)
    display_leaf = cv2.resize(isolated_leaf, (400, 400))
    display_mask = cv2.resize(leaf_mask, (400, 400), interpolation=cv2.INTER_NEAREST)
    return isolated_leaf, leaf_mask, display_leaf, display_mask


def get_severity(affected_percent):
    if affected_percent < 10:
        return "Low"
    if affected_percent < 25:
        return "Moderate"
    return "High"


def estimate_sample_type(mask, prediction_label, confidence):
    label_text = prediction_label.lower()
    if confidence >= SAMPLE_TYPE_CLASS_CONFIDENCE_THRESHOLD:
        if any(token in label_text for token in FRUIT_CLASS_TOKENS):
            return "fruit"
        if any(token in label_text for token in LEAF_CLASS_TOKENS):
            return "leaf"

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "plant sample"

    largest = max(contours, key=cv2.contourArea)
    area = max(cv2.contourArea(largest), 1.0)
    perimeter = max(cv2.arcLength(largest, True), 1.0)
    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    _, _, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / max(h, 1)
    hull = cv2.convexHull(largest)
    hull_area = max(cv2.contourArea(hull), 1.0)
    solidity = area / hull_area

    if circularity > 0.65 and 0.7 <= aspect_ratio <= 1.5 and solidity > 0.8:
        return "fruit"
    if aspect_ratio > 1.7 or solidity < 0.72:
        return "leaf"
    return "plant sample"


def estimate_disease_pattern(img, leaf_mask, prediction_label):
    if prediction_label in healthy_labels:
        return "No visible disease pattern detected"

    masked = cv2.bitwise_and(img, img, mask=leaf_mask)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    lesion_mask = ((value < 95) & (saturation > 40) & (leaf_mask > 0)).astype(np.uint8) * 255
    lesion_mask = cv2.medianBlur(lesion_mask, 5)

    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 12]
    lesion_count = len(lesion_areas)
    lesion_area_ratio = float(np.count_nonzero(lesion_mask) / max(np.count_nonzero(leaf_mask), 1))

    if lesion_count >= 12 and lesion_area_ratio < 0.18:
        return "Likely spot disease pattern"
    if lesion_area_ratio >= 0.22 and lesion_count <= 8:
        return "Likely rot or blight pattern"
    if lesion_count >= 8 and lesion_area_ratio >= 0.12:
        return "Likely mixed spot and blight pattern"
    return "General visible infection pattern"


def build_result_summary(sample_type, prediction_label, confidence, affected_percent, disease_pattern):
    severity = get_severity(affected_percent)
    health_status = "Healthy" if prediction_label in healthy_labels else "Diseased"
    class_reliable = confidence >= CLASS_CONFIDENCE_THRESHOLD

    if health_status == "Diseased":
        class_text = (
            f"The model's most likely class is {prediction_label}. "
            if class_reliable
            else f"The model's disease class estimate is {prediction_label}, but confidence is too low to treat that label as definitive. "
        )
        return (
            f"The uploaded {sample_type} is predicted as diseased with "
            f"{confidence:.2f}% confidence. Visible symptoms cover approximately "
            f"{affected_percent:.2f}% of the detected sample area, which places the case in the "
            f"{severity.lower()} severity range. {class_text}Based on the visible lesion structure, the image also shows a "
            f"{disease_pattern.lower()}. This diagnosis is image-based and not a laboratory-confirmed result."
        )

    return (
        f"The uploaded {sample_type} is predicted as healthy with "
        f"{confidence:.2f}% confidence. The highlighted affected area is "
        f"{affected_percent:.2f}%, which does not show a strong disease signature in the current image. "
        f"No specific disease type is suggested by the present visual pattern."
    )


def render_status_card(title, value, tone="neutral"):
    tones = {
        "danger": ("#fff1f2", "#be123c"),
        "success": ("#ecfdf5", "#047857"),
        "neutral": ("#eff6ff", "#1d4ed8"),
        "warning": ("#fffbeb", "#b45309"),
    }
    bg_color, text_color = tones[tone]
    st.markdown(
        f"""
        <div style="
            background:{bg_color};
            color:{text_color};
            padding:0.9rem 1rem;
            border-radius:14px;
            border:1px solid rgba(15,23,42,0.08);
            min-height:96px;
        ">
            <div style="font-size:0.9rem; opacity:0.85;">{title}</div>
            <div style="font-size:2rem; font-weight:700; margin-top:0.35rem;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# UI
# =========================
st.title("Leaf and Fruit Disease Detection using ML")
st.caption("Upload a leaf or fruit image to inspect visible damage and view the model diagnosis.")

uploaded_file = st.file_uploader(
    "Upload a leaf or fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    original, mask, result, affected = segment_leaf(img)
    focused_leaf_raw, leaf_mask_raw, focused_leaf, leaf_mask = prepare_leaf_focus(img)

    # ML Prediction
    features = extract_features(img)
    if isinstance(model_bundle, tuple):
        legacy_model, scaler = model_bundle
        prediction = legacy_model.predict(scaler.transform([features]))[0]
        confidence = np.max(legacy_model.predict_proba(scaler.transform([features]))) * 100
    else:
        prediction = model_bundle.predict([features])[0]
        confidence = np.max(model_bundle.predict_proba([features])) * 100

    prediction_label = class_names[prediction]
    sample_type = estimate_sample_type(leaf_mask_raw, prediction_label, confidence)
    health_status = "Healthy" if prediction_label in healthy_labels else "Diseased"
    class_reliable = confidence >= CLASS_CONFIDENCE_THRESHOLD
    disease_pattern = estimate_disease_pattern(focused_leaf_raw, leaf_mask_raw, prediction_label)
    severity = get_severity(affected)
    summary_text = build_result_summary(
        sample_type,
        prediction_label,
        confidence,
        affected,
        disease_pattern,
    )
    diagnosis_tone = "danger" if health_status == "Diseased" else "success"

    st.subheader("Result Dashboard")
    banner_color = "#7f1d1d" if health_status == "Diseased" else "#166534"
    banner_bg = "#fef2f2" if health_status == "Diseased" else "#f0fdf4"
    st.markdown(
        f"""
        <div style="
            background:{banner_bg};
            border-left:6px solid {banner_color};
            border-radius:14px;
            padding:1rem 1.1rem;
            margin-bottom:1rem;
        ">
            <div style="font-size:0.95rem; color:#475569;">Diagnosis</div>
            <div style="font-size:1.7rem; font-weight:700; color:{banner_color};">
                {health_status}
            </div>
            <div style="margin-top:0.35rem; color:#334155;">
                {summary_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dash1, dash2, dash3, dash4, dash5 = st.columns(5)
    with dash1:
        render_status_card("Prediction", health_status, diagnosis_tone)
    with dash2:
        render_status_card("Confidence", f"{confidence:.2f}%", "neutral")
    with dash3:
        render_status_card("Affected Area", f"{affected:.2f}%", "warning")
    with dash4:
        render_status_card("Severity", severity, "danger" if severity == "High" else "warning")
    with dash5:
        render_status_card(
            "Predicted Class",
            (
                prediction_label
                if health_status == "Diseased" and class_reliable
                else "Low-confidence estimate" if health_status == "Diseased"
                else "Healthy"
            ),
            diagnosis_tone if class_reliable or health_status == "Healthy" else "warning",
        )

    image_col1, image_col2 = st.columns([1.35, 1])
    with image_col1:
        st.subheader("Visual Analysis")
        viz1, viz2 = st.columns(2)
        with viz1:
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original Image")
            st.image(mask, clamp=True, caption="Disease Mask")
        with viz2:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Areas")
            st.image(cv2.cvtColor(focused_leaf, cv2.COLOR_BGR2RGB), caption="Leaf Focus")

    with image_col2:
        st.subheader("Written Result")
        st.markdown(
            f"""
            <div style="
                background:#f8fafc;
                border:1px solid #e2e8f0;
                border-radius:14px;
                padding:1rem;
                min-height:360px;
            ">
                <div style="font-size:0.9rem; color:#64748b; margin-bottom:0.6rem;">Summary</div>
                <div style="font-size:1rem; color:#0f172a; line-height:1.6;">
                    {summary_text}
                </div>
                <hr style="border:none; border-top:1px solid #e2e8f0; margin:1rem 0;">
                <div style="font-size:0.95rem; color:#334155;">
                    <strong>Sample type:</strong> {sample_type.capitalize()}
                </div>
                <div style="font-size:0.95rem; color:#334155; margin-top:0.45rem;">
                    <strong>Predicted class:</strong> {"Low-confidence estimate: " + prediction_label if health_status == "Diseased" and not class_reliable else prediction_label}
                </div>
                <div style="font-size:0.95rem; color:#334155; margin-top:0.45rem;">
                    <strong>Visual pattern estimate:</strong> {disease_pattern}
                </div>
                <div style="font-size:0.95rem; color:#334155; margin-top:0.45rem;">
                    <strong>Recommended interpretation:</strong> {health_status} with {severity.lower()} severity based on the current image.
                </div>
                <div style="font-size:0.9rem; color:#64748b; margin-top:0.75rem;">
                    The mask and highlighted regions show where the visible symptoms are concentrated. The predicted class comes from the trained multi-class model; the visual pattern estimate is supporting context only. Exact disease names should be trusted only when class confidence is high.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if health_status == "Diseased" and not class_reliable:
            st.warning(
                f"Specific disease class is low confidence ({confidence:.2f}%). Treat '{prediction_label}' as a tentative estimate, not a final label."
            )

        st.image(leaf_mask, clamp=True, caption="Leaf Isolation Mask")
