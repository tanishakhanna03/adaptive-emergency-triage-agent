import pandas as pd

# Load repeated vital-sign measurements
vitals = pd.read_csv("data/vitalsign.csv.gz")

print("\n--- DATASET SHAPE ---")
print(vitals.shape)

print("\n--- COLUMNS ---")
print(vitals.columns.tolist())

print("\n--- FIRST 10 ROWS ---")
print(vitals.head(10))

print("\n--- MISSING VALUES ---")
print(vitals.isnull().sum())

print("\n--- DATA TYPES ---")
print(vitals.dtypes)

# Check how many measurements exist for each ED stay
print("\n--- MEASUREMENTS PER STAY ---")
print(
    vitals.groupby("stay_id")
    .size()
    .sort_values(ascending=False)
    .head(10)
)
# ---------------------------------------
# Inspect one patient's vital-sign timeline
# ---------------------------------------

sample_stay_id = 32287800

patient_timeline = vitals[
    vitals["stay_id"] == sample_stay_id
].copy()

patient_timeline["charttime"] = pd.to_datetime(
    patient_timeline["charttime"]
)

patient_timeline = patient_timeline.sort_values("charttime")

columns_to_show = [
    "stay_id",
    "charttime",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp"
]

print("\n--- PATIENT VITAL TIMELINE ---")

print(
    patient_timeline[columns_to_show]
    .to_string(index=False)
)