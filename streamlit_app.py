from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.frp_core import DISEASES, engineer_features, project_dirs


ROOT = Path(__file__).resolve().parent
DIRS = project_dirs(ROOT)


st.set_page_config(
    page_title="FRP Multimorbidity Risk Dashboard",
    page_icon="medical",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #10231d;
        --muted: #52665f;
        --green: #087f5b;
        --green-dark: #064e3b;
        --green-soft: rgba(223, 247, 238, 0.74);
        --panel: rgba(255, 255, 255, 0.88);
        --line: rgba(8, 127, 91, 0.18);
        --danger: #c92a2a;
        --warning: #9a6700;
        --success: #087f5b;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(16, 185, 129, 0.16), transparent 28%),
            radial-gradient(circle at 86% 12%, rgba(5, 150, 105, 0.10), transparent 26%),
            linear-gradient(135deg, #f8fffb 0%, #effaf5 48%, #ffffff 100%);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.15rem;
        padding-bottom: 2.4rem;
    }

    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.78);
        border-right: 1px solid var(--line);
        backdrop-filter: blur(18px);
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: var(--ink);
    }

    .hero {
        background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(225,250,239,0.86));
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 20px 46px rgba(8, 127, 91, 0.13);
        backdrop-filter: blur(14px);
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
    }

    .hero h1 {
        color: var(--ink);
        font-size: clamp(1.55rem, 2.3vw, 2.3rem);
        line-height: 1.08;
        margin: 0;
        letter-spacing: 0;
    }

    .hero p {
        color: #34574b;
        margin: 0.45rem 0 0;
        max-width: 820px;
        font-size: 0.98rem;
    }

    .badge {
        color: var(--green-dark);
        background: rgba(255,255,255,0.76);
        border: 1px solid rgba(8,127,91,0.24);
        border-radius: 999px;
        padding: 0.45rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 800;
        white-space: nowrap;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 12px 28px rgba(8, 127, 91, 0.08);
        backdrop-filter: blur(12px);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--ink);
    }

    .section-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 800;
        margin: 1.1rem 0 0.65rem;
    }

    .risk-card, .panel, .recommendation {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 14px 32px rgba(8, 127, 91, 0.10);
        backdrop-filter: blur(12px);
    }

    .risk-card {
        border-top: 5px solid var(--green);
        padding: 1rem;
        min-height: 150px;
    }

    .risk-card.high { border-top-color: var(--danger); }
    .risk-card.moderate { border-top-color: var(--warning); }
    .risk-card.low { border-top-color: var(--success); }

    .risk-label {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .risk-value {
        color: var(--ink);
        font-size: 2.25rem;
        font-weight: 850;
        line-height: 1;
        margin: 0.45rem 0 0.28rem;
    }

    .risk-level {
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 800;
    }

    .risk-model {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.55rem;
    }

    .recommendation {
        border-left: 5px solid var(--green);
        padding: 0.78rem 0.9rem;
        margin-bottom: 0.55rem;
    }

    .recommendation.high { border-left-color: var(--danger); }
    .recommendation.moderate { border-left-color: var(--warning); }
    .recommendation.low { border-left-color: var(--success); }

    .recommendation b {
        display: block;
        color: var(--ink);
        margin-bottom: 0.14rem;
    }

    .recommendation span, .muted {
        color: var(--muted);
    }

    .profile-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.55rem;
    }

    .profile-item {
        background: var(--green-soft);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
    }

    .profile-item span {
        display: block;
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 700;
    }

    .profile-item strong {
        display: block;
        color: var(--ink);
        font-size: 0.95rem;
        margin-top: 0.12rem;
    }

    div[data-testid="stTabs"] button {
        color: var(--ink);
        font-weight: 750;
    }

    @media (max-width: 760px) {
        .hero {
            align-items: flex-start;
            flex-direction: column;
        }
        .profile-grid {
            grid-template-columns: 1fr;
        }
        .badge {
            white-space: normal;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundles() -> dict[str, dict]:
    bundles = {}
    for disease in DISEASES:
        path = DIRS["models"] / f"{disease.lower()}_bundle.pkl"
        if path.exists():
            bundles[disease] = joblib.load(path)
    return bundles


def age_to_brfss_group(age: int) -> int:
    bins = [
        (18, 24),
        (25, 29),
        (30, 34),
        (35, 39),
        (40, 44),
        (45, 49),
        (50, 54),
        (55, 59),
        (60, 64),
        (65, 69),
        (70, 74),
        (75, 79),
        (80, 120),
    ]
    for idx, (lo, hi) in enumerate(bins, start=1):
        if lo <= age <= hi:
            return idx
    return 1


def risk_class(probability: float) -> str:
    if probability >= 0.60:
        return "high"
    if probability >= 0.30:
        return "moderate"
    return "low"


def risk_label(probability: float) -> str:
    return risk_class(probability).title()


def bool01(value: bool) -> int:
    return 1 if value else 0


def build_patient_rows() -> dict[str, dict]:
    gender_val = 1 if gender == "Male" else 0
    smoker_val = 0 if smoking == "Never" else 1
    physically_active = 1 if physical_activity in {"Medium", "High"} else 0
    high_chol = bool01(cholesterol >= 240)
    high_bp = bool01(bp_sys >= 130 or bp_dia >= 80 or existing_hypertension == "Yes")
    heart_val = bool01(heart_disease == "Yes")
    prior_stroke_val = bool01(prior_stroke == "Yes")

    diabetes = {
        "Age": age_to_brfss_group(age),
        "Gender": gender_val,
        "High_Cholesterol": high_chol,
        "Chol_Check": 1,
        "BMI": bmi,
        "Smoker": smoker_val,
        "Heart_Disease": heart_val,
        "Physical_Activity": physically_active,
        "Fruits": 1 if nutrition_quality != "Low" else 0,
        "Veggies": 1 if nutrition_quality == "High" else physically_active,
        "Heavy_Alcohol": bool01(alcohol == "Yes"),
        "General_Health": general_health,
        "Mental_Health_Days": {0: 2, 1: 8, 2: 18}[stress_level],
        "Physical_Health_Days": max(0, (general_health - 1) * 4),
        "Difficulty_Walking": bool01(bmi >= 35 or general_health >= 4),
        "Stroke": prior_stroke_val,
        "High_BP": high_bp,
    }

    hypertension = {
        "Age": age,
        "Gender": gender_val,
        "Chest_Pain_Type": chest_pain_type,
        "Resting_BP": bp_sys,
        "Cholesterol": cholesterol,
        "Fasting_Blood_Sugar": bool01(glucose >= 120),
        "Rest_ECG": rest_ecg,
        "Max_Heart_Rate": max_heart_rate,
        "Exercise_Angina": bool01(exercise_angina == "Yes"),
        "ST_Depression": st_depression,
        "ST_Slope": st_slope,
        "Num_Vessels": num_vessels,
        "Thalassemia": thalassemia,
    }

    stroke = {
        "Gender": gender_val,
        "Age": age,
        "Hypertension": high_bp,
        "Heart_Disease": heart_val,
        "Ever_Married": bool01(ever_married == "Yes"),
        "Work_Type": work_type,
        "Residence_Type": bool01(residence == "Urban"),
        "Avg_Glucose": glucose,
        "BMI": bmi,
        "Smoking_Status": smoker_val,
    }

    return {"Diabetes": diabetes, "Hypertension": hypertension, "Stroke": stroke}


def predict_risks(bundles: dict[str, dict], rows: dict[str, dict]) -> dict[str, dict]:
    results = {}
    for disease, bundle in bundles.items():
        frame = pd.DataFrame([rows[disease]])
        engineered = engineer_features(disease, frame)
        X = engineered.reindex(columns=bundle["feature_columns"])
        probability = float(bundle["model"].predict_proba(X)[0, 1])
        results[disease] = {
            "probability": probability,
            "level": risk_label(probability),
            "model": bundle["model_name"],
            "warnings": bundle.get("warnings", []),
        }
    return results


def recommendation(disease: str, result: dict) -> str:
    level = result["level"]
    probability = result["probability"] * 100
    if level == "High":
        action = "Prioritize clinical review and confirm risk factors with a qualified professional."
    elif level == "Moderate":
        action = "Schedule follow-up, monitor modifiable risks, and review lifestyle/clinical markers."
    else:
        action = "Continue routine screening and reinforce healthy lifestyle habits."
    cls = risk_class(result["probability"])
    return (
        f'<div class="recommendation {cls}">'
        f"<b>{disease}: {level} risk ({probability:.1f}%)</b>"
        f"<span>{action}</span>"
        "</div>"
    )


with st.sidebar:
    st.markdown("## Patient Configuration")
    st.caption("Inputs are mapped into each trained disease model using the same engineered feature logic.")

    with st.expander("Demographics", expanded=True):
        age = st.slider("Age", 18, 90, 52)
        gender = st.selectbox("Gender", ["Female", "Male"])
        ever_married = st.selectbox("Ever married", ["Yes", "No"])
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        work_type = st.selectbox(
            "Work type",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: {
                0: "Private",
                1: "Self-employed",
                2: "Government",
                3: "Children/Student",
                4: "Other",
            }[x],
            index=0,
        )

    with st.expander("Clinical Measurements", expanded=True):
        bmi = st.slider("BMI", 12.0, 60.0, 28.5, 0.5)
        glucose = st.slider("Average / fasting glucose (mg/dL)", 55, 280, 112)
        bp_sys = st.slider("Systolic BP (mmHg)", 85, 210, 128)
        bp_dia = st.slider("Diastolic BP (mmHg)", 50, 130, 82)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 420, 210)
        max_heart_rate = st.slider("Max heart rate", 70, 210, 152)

    with st.expander("Lifestyle", expanded=False):
        smoking = st.selectbox("Smoking status", ["Never", "Former", "Current"])
        physical_activity = st.selectbox("Physical activity", ["Low", "Medium", "High"], index=1)
        nutrition_quality = st.selectbox("Fruit/vegetable intake", ["Low", "Medium", "High"], index=1)
        alcohol = st.selectbox("Heavy alcohol use", ["No", "Yes"])
        stress_level = st.slider("Stress level", 0, 2, 1, help="0 = low, 1 = moderate, 2 = high")
        general_health = st.slider("General health", 1, 5, 3, help="1 = excellent, 5 = poor")

    with st.expander("Medical History", expanded=False):
        existing_hypertension = st.selectbox("Existing hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
        prior_stroke = st.selectbox("Prior stroke", ["No", "Yes"])
        chest_pain_type = st.selectbox(
            "Chest pain type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical/none",
                1: "Atypical",
                2: "Non-anginal",
                3: "Asymptomatic/high concern",
            }[x],
        )
        exercise_angina = st.selectbox("Exercise angina", ["No", "Yes"])
        rest_ecg = st.selectbox("Resting ECG", [0, 1, 2], index=1)
        st_depression = st.slider("ST depression", 0.0, 6.5, 1.0, 0.1)
        st_slope = st.selectbox("ST slope", [0, 1, 2], index=1)
        num_vessels = st.selectbox("Major vessels", [0, 1, 2, 3, 4])
        thalassemia = st.selectbox("Thalassemia", [0, 1, 2, 3], index=2)


st.markdown(
    """
<div class="hero">
    <div>
        <h1>FRP Multimorbidity Risk Dashboard</h1>
        <p>Cleaned datasets, validated model selection, and patient-level risk estimates for diabetes, hypertension-risk proxy, and stroke.</p>
    </div>
    <div class="badge">Research use only</div>
</div>
""",
    unsafe_allow_html=True,
)

bundles = load_bundles()
if len(bundles) != len(DISEASES):
    st.error("Model bundles are missing. Run `python train_models.py` first.")
    st.stop()

patient_rows = build_patient_rows()
results = predict_risks(bundles, patient_rows)

highest_disease, highest_result = max(results.items(), key=lambda item: item[1]["probability"])
avg_risk = sum(item["probability"] for item in results.values()) / len(results)
high_count = sum(1 for item in results.values() if item["level"] == "High")

metric_cols = st.columns(4)
metric_cols[0].metric("Highest risk", highest_disease, f"{highest_result['probability'] * 100:.1f}%")
metric_cols[1].metric("Average risk", f"{avg_risk * 100:.1f}%")
metric_cols[2].metric("High-risk flags", high_count)
metric_cols[3].metric("Diseases modeled", len(results))

tab_assess, tab_quality, tab_visuals = st.tabs(["Risk Assessment", "Model Quality", "Data & Visuals"])

with tab_assess:
    st.markdown('<div class="section-title">Risk Assessment</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, disease in zip(cols, DISEASES):
        result = results[disease]
        cls = risk_class(result["probability"])
        with col:
            st.markdown(
                f"""
                <div class="risk-card {cls}">
                    <div class="risk-label">{disease}</div>
                    <div class="risk-value">{result["probability"] * 100:.1f}%</div>
                    <div class="risk-level">{result["level"]} risk</div>
                    <div class="risk-model">Selected model: {result["model"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown('<div class="section-title">Clinical Recommendations</div>', unsafe_allow_html=True)
        for disease in DISEASES:
            st.markdown(recommendation(disease, results[disease]), unsafe_allow_html=True)
        st.info(
            "This dashboard is for research and education. It must not replace diagnosis, treatment, or professional medical judgment."
        )

    with right:
        st.markdown('<div class="section-title">Patient Profile</div>', unsafe_allow_html=True)
        profile = {
            "Age": f"{age} years",
            "Gender": gender,
            "BMI": f"{bmi:.1f}",
            "Glucose": f"{glucose} mg/dL",
            "Blood pressure": f"{bp_sys}/{bp_dia} mmHg",
            "Cholesterol": f"{cholesterol} mg/dL",
            "Smoking": smoking,
            "Activity": physical_activity,
            "General health": f"{general_health}/5",
            "Known hypertension": existing_hypertension,
        }
        cards = "".join(
            f'<div class="profile-item"><span>{key}</span><strong>{value}</strong></div>'
            for key, value in profile.items()
        )
        st.markdown(f'<div class="profile-grid">{cards}</div>', unsafe_allow_html=True)

with tab_quality:
    st.markdown('<div class="section-title">Model Selection Results</div>', unsafe_allow_html=True)
    quality_tabs = st.tabs(DISEASES)
    for disease, quality_tab in zip(DISEASES, quality_tabs):
        with quality_tab:
            metrics_path = DIRS["reports"] / f"metrics_{disease.lower()}.csv"
            features_path = DIRS["reports"] / f"features_{disease.lower()}.csv"
            if not metrics_path.exists():
                st.warning(f"Metrics missing for {disease}. Run training first.")
                continue

            metrics = pd.read_csv(metrics_path, index_col=0)
            selected = bundles[disease]["model_name"]
            selected_row = metrics.loc[selected]
            a, b, c = st.columns(3)
            a.metric("Selected model", selected)
            b.metric("CV ROC-AUC", f"{selected_row['CV ROC-AUC']:.2f}%")
            c.metric("Test ROC-AUC", f"{selected_row['ROC-AUC']:.2f}%")

            for warning in bundles[disease].get("warnings", []):
                st.warning(warning)

            st.dataframe(metrics.round(3), width="stretch")
            if features_path.exists():
                features = pd.read_csv(features_path).head(12)
                st.markdown('<div class="section-title">Top Feature Signal</div>', unsafe_allow_html=True)
                st.dataframe(features.round(4), width="stretch")

with tab_visuals:
    st.markdown('<div class="section-title">Generated Reports & Visuals</div>', unsafe_allow_html=True)
    controls = st.columns([0.38, 0.62])
    selected_disease = controls[0].selectbox("Disease", DISEASES)
    plot_type = controls[1].selectbox(
        "Plot",
        [
            "Dashboard Summary",
            "Model Comparison",
            "ROC Curves",
            "Confusion Matrix",
            "Feature Importance",
        ],
    )
    plot_map = {
        "Dashboard Summary": "dashboard_summary.png",
        "Model Comparison": f"comparison_{selected_disease.lower()}.png",
        "ROC Curves": f"roc_{selected_disease.lower()}.png",
        "Confusion Matrix": f"cm_{selected_disease.lower()}.png",
        "Feature Importance": f"fi_{selected_disease.lower()}.png",
    }
    plot_path = DIRS["plots"] / plot_map[plot_type]
    if plot_path.exists():
        st.image(str(plot_path), width="stretch")
    else:
        st.warning(f"Plot missing: {plot_path.name}. Run `python train_models.py`.")

    report_path = DIRS["reports"] / "experiment_report.txt"
    if report_path.exists():
        with st.expander("Training report"):
            st.text(report_path.read_text(encoding="utf-8"))

st.markdown("---")
st.markdown(
    '<p class="muted" style="text-align:center;font-size:0.82rem;">FRP Project | White-green medical UI | Research use only</p>',
    unsafe_allow_html=True,
)
