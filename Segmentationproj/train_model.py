import os

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from feature_pipeline import extract_features

DATASET_PATH = "dataset"
KAGGLE_DATASETS = [
    "ateebnoone/fruits-dataset-for-fruit-disease-classification",
    "vipoooool/new-plant-diseases-dataset",
]
DISEASE_TOKENS = [
    "bacterial",
    "blight",
    "canker",
    "curl",
    "disease",
    "esca",
    "healthy",
    "mildew",
    "mites",
    "mold",
    "mosaic",
    "rot",
    "rust",
    "scab",
    "septoria",
    "spot",
    "virus",
]


def normalize_label(name):
    return name.strip().lower()


def prettify_label(name):
    clean = name.replace("_", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    clean = " ".join(clean.split())
    return clean.title()


def classify_folder(folder_name):
    normalized = normalize_label(folder_name)

    if normalized in {"healthy", "diseased"}:
        return normalized.title()

    if normalized.endswith("_healthy") or normalized.endswith(" healthy") or "healthy" in normalized:
        return prettify_label(folder_name)

    if any(token in normalized for token in DISEASE_TOKENS):
        return prettify_label(folder_name)

    return None


def build_class_map(root_path):
    class_map = {}

    for current_root, dirs, _files in os.walk(root_path):
        for directory in dirs:
            label = classify_folder(directory)
            if label is None:
                continue
            class_map.setdefault(label, []).append(os.path.join(current_root, directory))

    return class_map


def resolve_dataset_folders():
    if os.path.isdir(DATASET_PATH):
        local_map = build_class_map(DATASET_PATH)
        if len(local_map) >= 2:
            print(f"Using local dataset at: {DATASET_PATH}")
            return local_map

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Dataset folder not found and kagglehub is not installed. "
            "Install dependencies with `pip install -r requirements.txt` "
            "or provide a local dataset directory."
        ) from exc

    for kaggle_dataset in KAGGLE_DATASETS:
        print(f"Downloading dataset from Kaggle: {kaggle_dataset}")
        download_path = kagglehub.dataset_download(kaggle_dataset)
        print(f"Kaggle dataset downloaded to: {download_path}")

        class_map = build_class_map(download_path)
        if len(class_map) >= 2:
            print(f"Using {len(class_map)} class folders from Kaggle dataset")
            return class_map

    raise RuntimeError("No supported dataset source contains recognizable class folders.")


def is_healthy_label(label_name):
    return "healthy" in normalize_label(label_name)


X, y = [], []
dataset_folders = resolve_dataset_folders()
class_names = sorted(dataset_folders.keys())

print("📥 Loading dataset...")
for class_name in class_names:
    for folder_path in dataset_folders[class_name]:
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            X.append(extract_features(img))
            y.append(class_name)

X = np.array(X)
y = np.array(y)

print(f"✅ Total samples: {len(X)}")
print(f"✅ Total classes: {len(class_names)}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=200, svd_solver="randomized", random_state=42)),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=3.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
decoded_y_test = label_encoder.inverse_transform(y_test)
decoded_y_pred = label_encoder.inverse_transform(y_pred)

print("\n📊 Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(decoded_y_test, decoded_y_pred))
print("Mean prediction confidence:", float(np.max(y_prob, axis=1).mean()))

joblib.dump(
    {
        "model": model,
        "class_names": label_encoder.classes_.tolist(),
        "healthy_labels": [label for label in label_encoder.classes_.tolist() if is_healthy_label(label)],
        "feature_extractor": "feature_pipeline.extract_features",
    },
    "model.pkl",
)

print("\n💾 model.pkl saved successfully")
