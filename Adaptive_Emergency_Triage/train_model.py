import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load MIMIC-IV-ED dataset
# -----------------------------
data = pd.read_csv("data/triage.csv.gz")

print("Original rows:", len(data))

# -----------------------------
# 2. Select vital-sign features
# -----------------------------
features = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp"
]

target = "acuity"

# Keep rows where actual triage label exists
data = data.dropna(subset=[target])

X = data[features]
y = data[target].astype(int)

print("Labeled rows:", len(data))
print("\nAcuity distribution:")
print(y.value_counts().sort_index())

# -----------------------------
# 3. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. ML Pipeline
# -----------------------------
model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    )
])

# -----------------------------
# 5. Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 6. Test Model
# -----------------------------
predictions = model.predict(X_test)

print("\nAccuracy:")
print(round(accuracy_score(y_test, predictions), 3))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# -----------------------------
# 7. Save Model
# -----------------------------
joblib.dump(model, "triage_model.pkl")

print("\nModel saved successfully!")