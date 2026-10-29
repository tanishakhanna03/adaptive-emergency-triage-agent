import joblib
import pandas as pd

# Load trained model
model = joblib.load("triage_model.pkl")

FEATURES = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp"
]


def predict_triage(patient_data):
    patient_df = pd.DataFrame(
        [patient_data],
        columns=FEATURES
    )

    prediction = int(model.predict(patient_df)[0])

    probabilities = model.predict_proba(patient_df)[0]
    confidence = float(max(probabilities))

    return {
        "acuity": prediction,
        "confidence": confidence
    }


def adaptive_reassessment(previous_acuity, new_patient_data):
    result = predict_triage(new_patient_data)

    new_acuity = result["acuity"]

    if previous_acuity is None:
        status = "Initial assessment"
    elif new_acuity < previous_acuity:
        status = "Priority increased"
    elif new_acuity > previous_acuity:
        status = "Priority decreased"
    else:
        status = "Priority unchanged"

    return {
        "previous_acuity": previous_acuity,
        "new_acuity": new_acuity,
        "confidence": result["confidence"],
        "status": status
    }


def analyze_vital_timeline(vitals_file, stay_id):
    vitals = pd.read_csv(vitals_file)

    patient = vitals[
        vitals["stay_id"] == stay_id
    ].copy()

    if patient.empty:
        print("No records found for this stay ID.")
        return

    patient["charttime"] = pd.to_datetime(
        patient["charttime"]
    )

    patient = patient.sort_values("charttime")

    previous_acuity = None

    print("\n--- ADAPTIVE TRIAGE TIMELINE ---")

    for _, row in patient.iterrows():

        patient_data = {
            "temperature": row["temperature"],
            "heartrate": row["heartrate"],
            "resprate": row["resprate"],
            "o2sat": row["o2sat"],
            "sbp": row["sbp"],
            "dbp": row["dbp"]
        }

        result = predict_triage(patient_data)

        current_acuity = result["acuity"]
        confidence = result["confidence"]

        if previous_acuity is None:
            status = "Initial assessment"
        elif current_acuity < previous_acuity:
            status = "PRIORITY INCREASED"
        elif current_acuity > previous_acuity:
            status = "Priority decreased"
        else:
            status = "No change"

        print(
            row["charttime"],
            "| Acuity:",
            current_acuity,
            "| Confidence:",
            f"{confidence * 100:.2f}%",
            "|",
            status
        )

        previous_acuity = current_acuity


if __name__ == "__main__":

    analyze_vital_timeline(
        "data/vitalsign.csv.gz",
        32287800
    )