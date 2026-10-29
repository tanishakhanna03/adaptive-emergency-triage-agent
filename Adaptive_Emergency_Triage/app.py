import streamlit as st
import pandas as pd

from triage_agent import predict_triage, adaptive_reassessment

st.set_page_config(
    page_title="Adaptive Emergency Triage Agent",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Adaptive Emergency Triage Agent")

st.caption(
    "AI-based educational prototype using de-identified "
    "MIMIC-IV-ED demo data. Not intended for clinical use."
)

# =========================================================
# TAB STRUCTURE
# =========================================================

tab1, tab2 = st.tabs([
    "🩺 Patient Assessment",
    "📈 Adaptive Timeline"
])


# =========================================================
# TAB 1 — PATIENT ASSESSMENT
# =========================================================

with tab1:

    st.header("Patient Vital Signs")

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.number_input(
            "Temperature (°F)",
            min_value=80.0,
            max_value=115.0,
            value=98.6
        )

        heartrate = st.number_input(
            "Heart Rate (bpm)",
            min_value=20,
            max_value=250,
            value=80
        )

        resprate = st.number_input(
            "Respiratory Rate (breaths/min)",
            min_value=5,
            max_value=80,
            value=18
        )

    with col2:

        o2sat = st.number_input(
            "Oxygen Saturation (%)",
            min_value=50,
            max_value=100,
            value=98
        )

        sbp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=40,
            max_value=250,
            value=120
        )

        dbp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=20,
            max_value=150,
            value=80
        )

    patient_data = {
        "temperature": temperature,
        "heartrate": heartrate,
        "resprate": resprate,
        "o2sat": o2sat,
        "sbp": sbp,
        "dbp": dbp
    }

    if st.button(
        "Assess Patient",
        type="primary"
    ):

        result = predict_triage(patient_data)

        st.session_state.previous_acuity = result["acuity"]

        st.subheader("Triage Result")

        m1, m2 = st.columns(2)

        m1.metric(
            "Predicted Acuity",
            result["acuity"]
        )

        m2.metric(
            "Model Confidence",
            f"{result['confidence'] * 100:.2f}%"
        )

        acuity = result["acuity"]

        if acuity == 1:
            st.error(
                "Level 1 — Highest predicted urgency"
            )

        elif acuity == 2:
            st.warning(
                "Level 2 — High predicted urgency"
            )

        elif acuity == 3:
            st.info(
                "Level 3 — Moderate predicted urgency"
            )

        elif acuity == 4:
            st.success(
                "Level 4 — Lower predicted urgency"
            )

        else:
            st.success(
                "Level 5 — Lowest predicted urgency"
            )

    st.divider()

    st.subheader("🔄 Manual Reassessment")

    st.write(
        "Change the vital signs above and reassess "
        "to simulate updated patient measurements."
    )

    if st.button("Reassess Patient"):

        if "previous_acuity" not in st.session_state:

            st.warning(
                "Perform an initial assessment first."
            )

        else:

            reassessment = adaptive_reassessment(
                st.session_state.previous_acuity,
                patient_data
            )

            previous = reassessment["previous_acuity"]
            new = reassessment["new_acuity"]

            c1, c2 = st.columns(2)

            c1.metric(
                "Previous Acuity",
                previous
            )

            c2.metric(
                "New Acuity",
                new
            )

            if new < previous:

                st.error(
                    "⚠️ Model priority increased."
                )

            elif new > previous:

                st.success(
                    "Model priority decreased."
                )

            else:

                st.info(
                    "Model priority unchanged."
                )

            st.write(
                "Confidence:",
                f"{reassessment['confidence'] * 100:.2f}%"
            )

            st.session_state.previous_acuity = new


# =========================================================
# TAB 2 — REAL MIMIC VITAL TIMELINE
# =========================================================

with tab2:

    st.header("📈 Adaptive Patient Timeline")

    st.write(
        "Explore repeated vital-sign measurements from "
        "a de-identified MIMIC-IV-ED demo stay."
    )

    vitals = pd.read_csv(
        "data/vitalsign.csv.gz"
    )

    available_stays = (
        vitals["stay_id"]
        .value_counts()
    )

    repeated_stays = available_stays[
        available_stays > 1
    ].index.tolist()

    stay_id = st.selectbox(
        "Select ED Stay ID",
        repeated_stays
    )

    patient = vitals[
        vitals["stay_id"] == stay_id
    ].copy()

    patient["charttime"] = pd.to_datetime(
        patient["charttime"]
    )

    patient = patient.sort_values(
        "charttime"
    )

    st.write(
        f"Number of recorded measurements: "
        f"{len(patient)}"
    )

    timeline_results = []

    previous_acuity = None

    for _, row in patient.iterrows():

        patient_data = {
            "temperature": row["temperature"],
            "heartrate": row["heartrate"],
            "resprate": row["resprate"],
            "o2sat": row["o2sat"],
            "sbp": row["sbp"],
            "dbp": row["dbp"]
        }

        result = predict_triage(
            patient_data
        )

        current_acuity = result["acuity"]

        if previous_acuity is None:

            status = "Initial"

        elif current_acuity < previous_acuity:

            status = "Priority Increased"

        elif current_acuity > previous_acuity:

            status = "Priority Decreased"

        else:

            status = "No Change"

        timeline_results.append({

            "Time": row["charttime"],

            "Heart Rate": row["heartrate"],

            "Respiratory Rate": row["resprate"],

            "SpO2": row["o2sat"],

            "SBP": row["sbp"],

            "DBP": row["dbp"],

            "Predicted Acuity": current_acuity,

            "Confidence (%)":
                round(
                    result["confidence"] * 100,
                    2
                ),

            "Status": status
        })

        previous_acuity = current_acuity


    timeline_df = pd.DataFrame(
        timeline_results
    )


    # =====================================================
    # TIMELINE TABLE
    # =====================================================

    st.subheader(
        "🕒 Model Reassessment Timeline"
    )

    st.dataframe(
        timeline_df,
        use_container_width=True
    )


    # =====================================================
    # ACUITY GRAPH
    # =====================================================

    st.subheader(
        "Predicted Acuity Over Time"
    )

    acuity_chart = (
        timeline_df[
            ["Time", "Predicted Acuity"]
        ]
        .set_index("Time")
    )

    st.line_chart(
        acuity_chart
    )


    # =====================================================
    # VITAL SIGNS GRAPH
    # =====================================================

    st.subheader(
        "Vital Sign Trends"
    )

    vital_chart = (
        timeline_df[
            [
                "Time",
                "Heart Rate",
                "Respiratory Rate",
                "SpO2"
            ]
        ]
        .set_index("Time")
    )

    st.line_chart(
        vital_chart
    )


    # =====================================================
    # PRIORITY CHANGE SUMMARY
    # =====================================================

    increased = (
        timeline_df["Status"]
        == "Priority Increased"
    ).sum()

    decreased = (
        timeline_df["Status"]
        == "Priority Decreased"
    ).sum()

    st.subheader(
        "Adaptive Reassessment Summary"
    )

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Measurements",
        len(timeline_df)
    )

    s2.metric(
        "Priority Increases",
        int(increased)
    )

    s3.metric(
        "Priority Decreases",
        int(decreased)
    )


st.divider()

st.caption(
    "Research/educational prototype only. "
    "Predictions have not been clinically validated."
)