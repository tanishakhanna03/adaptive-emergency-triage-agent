import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =========================================================
# 1. LOAD DATA
# =========================================================

data = pd.read_csv("data/triage.csv.gz")

print("\nOriginal rows:", len(data))

# Remove rows where target/acuity is missing
data = data.dropna(subset=["acuity"]).copy()

data["acuity"] = data["acuity"].astype(int)

print("Labeled rows:", len(data))


# =========================================================
# 2. SELECT FEATURES
# =========================================================

numeric_features = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp"
]

text_feature = "chiefcomplaint"

target = "acuity"

# Handle missing complaint text
data[text_feature] = (
    data[text_feature]
    .fillna("unknown")
    .astype(str)
)

X = data[numeric_features + [text_feature]]

y = data[target]


print("\nClass Distribution:")
print(y.value_counts().sort_index())


# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# 4. NUMERIC PREPROCESSING
# =========================================================

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# =========================================================
# 5. TEXT PREPROCESSING
# =========================================================

text_transformer = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2)
)


# =========================================================
# 6. COMBINE PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_transformer,
        numeric_features
    ),
    (
        "text",
        text_transformer,
        text_feature
    )
])


# =========================================================
# 7. MODELS TO COMPARE
# =========================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
}


best_model = None
best_model_name = None
best_f1 = -1


# =========================================================
# 8. TRAIN AND TEST BOTH MODELS
# =========================================================

for model_name, classifier in models.items():

    print("\n====================================")
    print(model_name)
    print("====================================")

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ])

    # Train
    pipeline.fit(
        X_train,
        y_train
    )

    # Predict
    predictions = pipeline.predict(
        X_test
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save best performing model
    if f1 > best_f1:

        best_f1 = f1

        best_model = pipeline

        best_model_name = model_name


# =========================================================
# 9. SAVE BEST V2 MODEL
# =========================================================

joblib.dump(
    best_model,
    "triage_model_v2.pkl"
)


print("\n====================================")
print("BEST MODEL:", best_model_name)

print(
    f"BEST WEIGHTED F1: "
    f"{best_f1 * 100:.2f}%"
)

print(
    "Saved as: triage_model_v2.pkl"
)

print("====================================")