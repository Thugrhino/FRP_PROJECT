from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency guard
    xgb = None

try:
    import shap
except Exception:  # pragma: no cover - optional dependency guard
    shap = None

warnings.filterwarnings("ignore", category=UserWarning)


RANDOM_STATE = 42
DISEASES = ["Diabetes", "Hypertension", "Stroke"]
MEDICAL_GREEN = "#087f5b"
DEEP_GREEN = "#064e3b"
SOFT_GREEN = "#dff7ee"
INK = "#10231d"


@dataclass(frozen=True)
class DatasetConfig:
    disease: str
    raw_file: str
    target: str
    rename: dict[str, str]


DATASETS: dict[str, DatasetConfig] = {
    "Diabetes": DatasetConfig(
        disease="Diabetes",
        raw_file="diabetes_data.csv",
        target="Diabetes",
        rename={
            "Sex": "Gender",
            "HighChol": "High_Cholesterol",
            "CholCheck": "Chol_Check",
            "HeartDiseaseorAttack": "Heart_Disease",
            "PhysActivity": "Physical_Activity",
            "HvyAlcoholConsump": "Heavy_Alcohol",
            "GenHlth": "General_Health",
            "MentHlth": "Mental_Health_Days",
            "PhysHlth": "Physical_Health_Days",
            "DiffWalk": "Difficulty_Walking",
            "HighBP": "High_BP",
        },
    ),
    "Hypertension": DatasetConfig(
        disease="Hypertension",
        raw_file="hypertension_data.csv",
        target="Hypertension",
        rename={
            "age": "Age",
            "sex": "Gender",
            "cp": "Chest_Pain_Type",
            "trestbps": "Resting_BP",
            "chol": "Cholesterol",
            "fbs": "Fasting_Blood_Sugar",
            "restecg": "Rest_ECG",
            "thalach": "Max_Heart_Rate",
            "exang": "Exercise_Angina",
            "oldpeak": "ST_Depression",
            "slope": "ST_Slope",
            "ca": "Num_Vessels",
            "thal": "Thalassemia",
            "target": "Hypertension",
        },
    ),
    "Stroke": DatasetConfig(
        disease="Stroke",
        raw_file="stroke_data.csv",
        target="Stroke",
        rename={
            "sex": "Gender",
            "age": "Age",
            "hypertension": "Hypertension",
            "heart_disease": "Heart_Disease",
            "ever_married": "Ever_Married",
            "work_type": "Work_Type",
            "Residence_type": "Residence_Type",
            "avg_glucose_level": "Avg_Glucose",
            "bmi": "BMI",
            "smoking_status": "Smoking_Status",
            "stroke": "Stroke",
        },
    ),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_dirs(root: Path | None = None) -> dict[str, Path]:
    root = root or project_root()
    dirs = {
        "root": root,
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "models": root / "models",
        "plots": root / "plots",
        "reports": root / "reports",
    }
    for path in dirs.values():
        if path != root:
            path.mkdir(parents=True, exist_ok=True)
    return dirs


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def load_raw_dataset(disease: str, root: Path | None = None) -> pd.DataFrame:
    dirs = project_dirs(root)
    config = DATASETS[disease]
    return pd.read_csv(dirs["raw"] / config.raw_file)


def standardize_columns(disease: str, df: pd.DataFrame) -> pd.DataFrame:
    config = DATASETS[disease]
    out = df.rename(columns=config.rename).copy()
    out.columns = [str(col).strip() for col in out.columns]
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _invalid_to_nan(df: pd.DataFrame, columns: list[str], allowed: set[int]) -> None:
    for col in columns:
        if col in df.columns:
            df.loc[~df[col].isin(allowed), col] = np.nan


def clean_dataset(disease: str, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = DATASETS[disease]
    df = standardize_columns(disease, raw_df)
    before = len(df)
    missing_before = int(df.isna().sum().sum())

    df = df.drop_duplicates().copy()
    df = df[df[config.target].isin([0, 1])].copy()

    if disease == "Diabetes":
        binary_cols = [
            "Gender",
            "High_Cholesterol",
            "Chol_Check",
            "Smoker",
            "Heart_Disease",
            "Physical_Activity",
            "Fruits",
            "Veggies",
            "Heavy_Alcohol",
            "Difficulty_Walking",
            "Stroke",
            "High_BP",
        ]
        _invalid_to_nan(df, binary_cols, {0, 1})
        df = df[df["Age"].between(1, 13)]
        df["BMI"] = df["BMI"].clip(12, 60)
        df["General_Health"] = df["General_Health"].clip(1, 5)
        df["Mental_Health_Days"] = df["Mental_Health_Days"].clip(0, 30)
        df["Physical_Health_Days"] = df["Physical_Health_Days"].clip(0, 30)

    elif disease == "Hypertension":
        categorical_cols = [
            "Gender",
            "Chest_Pain_Type",
            "Fasting_Blood_Sugar",
            "Rest_ECG",
            "Exercise_Angina",
            "ST_Slope",
            "Num_Vessels",
            "Thalassemia",
        ]
        allowed_values = {
            "Gender": {0, 1},
            "Chest_Pain_Type": {0, 1, 2, 3},
            "Fasting_Blood_Sugar": {0, 1},
            "Rest_ECG": {0, 1, 2},
            "Exercise_Angina": {0, 1},
            "ST_Slope": {0, 1, 2},
            "Num_Vessels": {0, 1, 2, 3, 4},
            "Thalassemia": {0, 1, 2, 3},
        }
        for col in categorical_cols:
            _invalid_to_nan(df, [col], allowed_values[col])
        df = df[df["Age"].between(18, 95)].copy()
        df["Resting_BP"] = df["Resting_BP"].clip(80, 220)
        df["Cholesterol"] = df["Cholesterol"].clip(100, 600)
        df["Max_Heart_Rate"] = df["Max_Heart_Rate"].clip(60, 230)
        df["ST_Depression"] = df["ST_Depression"].clip(0, 7)

    elif disease == "Stroke":
        categorical_cols = [
            "Gender",
            "Hypertension",
            "Heart_Disease",
            "Ever_Married",
            "Work_Type",
            "Residence_Type",
            "Smoking_Status",
        ]
        allowed_values = {
            "Gender": {0, 1},
            "Hypertension": {0, 1},
            "Heart_Disease": {0, 1},
            "Ever_Married": {0, 1},
            "Work_Type": {0, 1, 2, 3, 4},
            "Residence_Type": {0, 1},
            "Smoking_Status": {0, 1},
        }
        for col in categorical_cols:
            _invalid_to_nan(df, [col], allowed_values[col])
        df = df[df["Age"].between(0, 105)].copy()
        df["Avg_Glucose"] = df["Avg_Glucose"].clip(50, 300)
        df["BMI"] = df["BMI"].clip(12, 60)

    df[config.target] = df[config.target].astype(int)
    report = {
        "disease": disease,
        "rows_raw": before,
        "rows_cleaned": len(df),
        "rows_removed": before - len(df),
        "duplicate_rows_removed": before - len(standardize_columns(disease, raw_df).drop_duplicates()),
        "missing_values_before": missing_before,
        "missing_values_after_cleaning": int(df.isna().sum().sum()),
        "positive_rate": round(float(df[config.target].mean()), 4),
    }
    return df.reset_index(drop=True), report


def _bin(series: pd.Series, bins: list[float], labels: list[int]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True).astype(float)


def engineer_features(disease: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if disease == "Diabetes":
        out["BMI_Category"] = _bin(out["BMI"], [0, 18.5, 25, 30, 40, 100], [0, 1, 2, 3, 4])
        out["Age_Group"] = _bin(out["Age"], [0, 4, 8, 11, 14], [0, 1, 2, 3])
        out["Cardio_Risk_Index"] = (
            out["High_BP"].fillna(0)
            + out["High_Cholesterol"].fillna(0)
            + out["Heart_Disease"].fillna(0)
            + out["Stroke"].fillna(0)
        )
        out["Lifestyle_Index"] = (
            out["Physical_Activity"].fillna(0)
            + out["Fruits"].fillna(0)
            + out["Veggies"].fillna(0)
            - out["Smoker"].fillna(0)
            - out["Heavy_Alcohol"].fillna(0)
        )
        out["Health_Burden"] = out["Mental_Health_Days"].fillna(0) + out["Physical_Health_Days"].fillna(0)
        out["BMI_Age_Interaction"] = out["BMI"] * out["Age"]

    elif disease == "Hypertension":
        out["Age_Group"] = _bin(out["Age"], [0, 40, 55, 70, 120], [0, 1, 2, 3])
        out["BP_Category"] = _bin(out["Resting_BP"], [0, 120, 130, 140, 300], [0, 1, 2, 3])
        out["Cholesterol_Category"] = _bin(out["Cholesterol"], [0, 200, 240, 600], [0, 1, 2])
        out["Heart_Rate_Reserve"] = (220 - out["Age"]) - out["Max_Heart_Rate"]
        out["ST_Risk"] = out["ST_Depression"] * (out["ST_Slope"].fillna(0) + 1)
        out["Vascular_Findings"] = out["Num_Vessels"].fillna(0) + out["Thalassemia"].fillna(0)
        out["Chol_HR_Ratio"] = out["Cholesterol"] / out["Max_Heart_Rate"].replace(0, np.nan)

    elif disease == "Stroke":
        out["BMI_Category"] = _bin(out["BMI"], [0, 18.5, 25, 30, 40, 100], [0, 1, 2, 3, 4])
        out["Age_Group"] = _bin(out["Age"], [0, 30, 50, 65, 80, 120], [0, 1, 2, 3, 4])
        out["Glucose_Category"] = _bin(out["Avg_Glucose"], [0, 100, 126, 200, 400], [0, 1, 2, 3])
        out["Vascular_Risk"] = out["Hypertension"].fillna(0) + out["Heart_Disease"].fillna(0)
        out["Age_Glucose"] = out["Age"] * out["Avg_Glucose"]
        out["BMI_Glucose"] = out["BMI"] * out["Avg_Glucose"]

    return out


def prepare_dataset(disease: str, root: Path | None = None) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    raw = load_raw_dataset(disease, root)
    cleaned, report = clean_dataset(disease, raw)
    engineered = engineer_features(disease, cleaned)
    target = DATASETS[disease].target
    X = engineered.drop(columns=[target])
    y = engineered[target].astype(int)
    return X, y, report


def save_cleaned_datasets(root: Path | None = None) -> dict[str, dict[str, Any]]:
    dirs = project_dirs(root)
    reports: dict[str, dict[str, Any]] = {}
    for disease in DISEASES:
        raw = load_raw_dataset(disease, root)
        cleaned, report = clean_dataset(disease, raw)
        engineered = engineer_features(disease, cleaned)
        cleaned.to_csv(dirs["processed"] / f"{disease.lower()}_cleaned.csv", index=False)
        engineered.to_csv(dirs["processed"] / f"{disease.lower()}_engineered.csv", index=False)
        reports[disease] = report
    (dirs["reports"] / "data_quality_report.json").write_text(
        json.dumps(_json_safe(reports), indent=2),
        encoding="utf-8",
    )
    return reports


def build_model_candidates(pos_weight: float) -> dict[str, Pipeline]:
    candidates: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2500,
                        class_weight="balanced",
                        C=0.65,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=260,
                        max_depth=3,
                        min_samples_leaf=80,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        max_leaf_nodes=8,
                        max_depth=2,
                        learning_rate=0.04,
                        l2_regularization=2.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    if xgb is not None:
        candidates["XGBoost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    xgb.XGBClassifier(
                        n_estimators=160,
                        max_depth=2,
                        learning_rate=0.035,
                        subsample=0.78,
                        colsample_bytree=0.78,
                        min_child_weight=80,
                        reg_lambda=10,
                        reg_alpha=0.25,
                        scale_pos_weight=pos_weight,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return candidates


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "F1-Score": f1_score(y_true, y_pred, zero_division=0) * 100,
        "ROC-AUC": roc_auc_score(y_true, y_prob) * 100,
        "PR-AUC": average_precision_score(y_true, y_prob) * 100,
        "Brier": brier_score_loss(y_true, y_prob),
    }


def suspicious_performance(row: dict[str, Any]) -> bool:
    return bool(
        row["CV ROC-AUC"] >= 99.5
        or row["ROC-AUC"] >= 99.5
        or row["Accuracy"] >= 99.5
    )


def select_best_model(metrics: pd.DataFrame) -> str:
    stable = metrics[~metrics["Suspicious"]].copy()
    if stable.empty:
        stable = metrics.copy()
    stable = stable.sort_values(
        by=["CV ROC-AUC", "ROC-AUC", "Brier"],
        ascending=[False, False, True],
    )
    return str(stable.index[0])


def _feature_screening_report(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    imputed = SimpleImputer(strategy="median").fit_transform(X)
    mi = mutual_info_classif(imputed, y, random_state=RANDOM_STATE)
    report = pd.DataFrame({"Feature": X.columns, "Mutual_Info": mi})
    return report.sort_values("Mutual_Info", ascending=False).reset_index(drop=True)


def train_one_disease(disease: str, root: Path | None = None) -> dict[str, Any]:
    dirs = project_dirs(root)
    X, y, data_report = prepare_dataset(disease, root)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    pos_weight = round(neg / pos, 3) if pos else 1.0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_model_candidates(pos_weight)
    fitted: dict[str, Pipeline] = {}
    rows: list[dict[str, Any]] = []

    for model_name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        row = evaluate_predictions(y_test, y_prob)
        row.update(
            {
                "Model": model_name,
                "CV ROC-AUC": cv_scores.mean() * 100,
                "CV Std": cv_scores.std() * 100,
            }
        )
        row["Suspicious"] = suspicious_performance(row)
        rows.append(row)
        fitted[model_name] = model

    metrics = pd.DataFrame(rows).set_index("Model")
    best_name = select_best_model(metrics)
    best_model = fitted[best_name]

    warnings_list: list[str] = []
    suspicious_models = metrics.index[metrics["Suspicious"]].tolist()
    if suspicious_models:
        warnings_list.append(
            "Suspiciously perfect validation detected for: "
            + ", ".join(suspicious_models)
            + ". Selection used the strongest stable non-perfect model where available."
        )
    if disease == "Hypertension":
        warnings_list.append(
            "The provided hypertension CSV is a heart-disease-style dataset; results should be treated as hypertension-risk proxy modeling, not a clinical hypertension diagnosis."
        )

    feature_report = _feature_screening_report(X, y)
    feature_report.to_csv(dirs["reports"] / f"features_{disease.lower()}.csv", index=False)
    metrics.round(4).to_csv(dirs["reports"] / f"metrics_{disease.lower()}.csv")

    bundle = {
        "disease": disease,
        "target": DATASETS[disease].target,
        "model_name": best_name,
        "model": best_model,
        "feature_columns": list(X.columns),
        "data_report": data_report,
        "metrics": metrics.round(4).to_dict(orient="index"),
        "warnings": warnings_list,
        "positive_rate": float(y.mean()),
    }
    joblib.dump(bundle, dirs["models"] / f"{disease.lower()}_bundle.pkl")
    joblib.dump(best_model, dirs["models"] / f"{disease.lower()}_best_model.pkl")

    plot_model_comparison(disease, metrics, dirs["plots"])
    plot_roc_curves(disease, fitted, X_test, y_test, dirs["plots"])
    plot_confusion(disease, best_name, best_model, X_test, y_test, dirs["plots"])
    plot_feature_importance(disease, best_model, X_test, y_test, dirs["plots"])
    plot_shap_explanations(disease, best_name, best_model, X_train, X_test, dirs["plots"])

    return {
        "disease": disease,
        "best_model": best_name,
        "metrics": metrics.round(4),
        "warnings": warnings_list,
        "rows": len(X),
        "features": len(X.columns),
    }


def train_all(root: Path | None = None) -> dict[str, Any]:
    dirs = project_dirs(root)
    data_reports = save_cleaned_datasets(root)
    registry: dict[str, Any] = {"datasets": data_reports, "models": {}}
    summaries = []

    for disease in DISEASES:
        result = train_one_disease(disease, root)
        registry["models"][disease] = {
            "best_model": result["best_model"],
            "rows": result["rows"],
            "features": result["features"],
            "warnings": result["warnings"],
        }
        summaries.append(result)

    plot_dashboard_summary(summaries, dirs["reports"], dirs["plots"])
    (dirs["reports"] / "model_registry.json").write_text(
        json.dumps(_json_safe(registry), indent=2),
        encoding="utf-8",
    )
    write_experiment_report(summaries, dirs["reports"])
    return registry


def plot_model_comparison(disease: str, metrics: pd.DataFrame, plot_dir: Path) -> None:
    cols = ["Accuracy", "F1-Score", "ROC-AUC", "CV ROC-AUC"]
    data = metrics[cols].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=data, x="Model", y="Score", hue="Metric", ax=ax, palette="Greens")
    ax.set_ylim(0, 105)
    ax.set_title(f"{disease} Model Comparison", color=INK, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(plot_dir / f"comparison_{disease.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    disease: str,
    models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plot_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="#8a9b94", label="Random")
    palette = ["#064e3b", "#087f5b", "#10b981", "#65a30d"]
    for (name, model), color in zip(models.items(), palette):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} ({roc_auc_score(y_test, y_prob):.3f})")
    ax.set_title(f"{disease} ROC Curves", color=INK, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / f"roc_{disease.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    disease: str,
    model_name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plot_dir: Path,
) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax,
    )
    ax.set_title(f"{disease} Confusion Matrix - {model_name}", color=INK, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(plot_dir / f"cm_{disease.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    disease: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plot_dir: Path,
) -> None:
    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)
    y_sample = y_test.loc[X_sample.index]
    importance = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    imp = (
        pd.DataFrame({"Feature": X_test.columns, "Importance": importance.importances_mean})
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.barplot(data=imp, y="Feature", x="Importance", color=MEDICAL_GREEN, ax=ax)
    ax.set_title(f"{disease} Top Permutation Importance", color=INK, fontweight="bold")
    ax.set_xlabel("Mean ROC-AUC decrease")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(plot_dir / f"fi_{disease.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pipeline_model_input(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    transformed: Any = X
    feature_names = list(X.columns)
    for _, step in model.steps[:-1]:
        transformed = step.transform(transformed)
        if hasattr(step, "get_feature_names_out"):
            try:
                feature_names = list(step.get_feature_names_out(feature_names))
            except Exception:
                feature_names = [f"feature_{idx}" for idx in range(transformed.shape[1])]
        elif hasattr(transformed, "shape"):
            feature_names = [f"feature_{idx}" for idx in range(transformed.shape[1])]
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def _positive_class_shap_values(values: Any) -> np.ndarray:
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[-1] > 1:
            values = values[:, :, 1]
        elif values.shape[0] > 1:
            values = values[1]
        else:
            values = values[:, :, 0]
    return values


def plot_shap_explanations(
    disease: str,
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    plot_dir: Path,
) -> None:
    if shap is None:
        return

    train_size = min(200, len(X_train))
    explain_size = min(300, len(X_test))
    X_background = X_train.sample(train_size, random_state=RANDOM_STATE)
    X_explain_raw = X_test.sample(explain_size, random_state=RANDOM_STATE)
    X_background_model = _pipeline_model_input(model, X_background)
    X_explain_model = _pipeline_model_input(model, X_explain_raw)
    estimator = model.named_steps.get("model", model)

    try:
        if isinstance(estimator, LogisticRegression):
            explainer = shap.LinearExplainer(estimator, X_background_model)
            shap_values = explainer.shap_values(X_explain_model)
        else:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_explain_model)
    except Exception:
        return

    shap_values = _positive_class_shap_values(shap_values)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_explain_model,
        show=False,
        plot_type="dot",
        color_bar=True,
        max_display=15,
    )
    plt.title(f"{disease} SHAP Summary - {model_name}", color=INK, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(plot_dir / f"shap_summary_{disease.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(
        shap_values,
        X_explain_model,
        show=False,
        plot_type="bar",
        color=MEDICAL_GREEN,
        max_display=15,
    )
    plt.title(f"{disease} SHAP Feature Importance - {model_name}", color=INK, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(plot_dir / f"shap_bar_{disease.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_dashboard_summary(summaries: list[dict[str, Any]], report_dir: Path, plot_dir: Path) -> None:
    rows = []
    for item in summaries:
        disease = item["disease"]
        metrics = pd.read_csv(report_dir / f"metrics_{disease.lower()}.csv", index_col=0)
        best = item["best_model"]
        row = metrics.loc[best].to_dict()
        row["Disease"] = disease
        row["Best Model"] = best
        rows.append(row)
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, metric in zip(axes, ["Accuracy", "F1-Score", "ROC-AUC"]):
        sns.barplot(data=df, x="Disease", y=metric, hue="Best Model", dodge=False, palette="Greens", ax=ax)
        ax.set_ylim(0, 105)
        ax.set_title(metric, fontweight="bold", color=INK)
        ax.set_xlabel("")
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Selected Model Performance Summary", color=INK, fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(plot_dir / "dashboard_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_metric_table(metrics: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    columns = columns or [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC",
        "CV ROC-AUC",
        "CV Std",
        "Brier",
    ]
    table = metrics.reset_index()
    columns = ["Model"] + [column for column in columns if column in table.columns]
    table = table[columns].copy()
    for column in table.columns:
        if column != "Model":
            table[column] = table[column].map(lambda value: f"{float(value):.3f}")

    widths = {
        column: max(len(column), *(len(str(value)) for value in table[column]))
        for column in table.columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in table.columns)
    divider = "-+-".join("-" * widths[column] for column in table.columns)
    rows = [
        " | ".join(str(row[column]).ljust(widths[column]) for column in table.columns).rstrip()
        for _, row in table.iterrows()
    ]
    return [header, divider, *rows]


def write_experiment_report(summaries: list[dict[str, Any]], report_dir: Path) -> None:
    lines = [
        "FRP PROJECT - MODEL TRAINING REPORT",
        "=" * 44,
        "",
        "Selection policy:",
        "Models are ranked by cross-validated ROC-AUC. Any model with near-perfect",
        "validation or test scores is flagged as suspicious and skipped when a stable",
        "non-perfect model is available.",
        "",
    ]
    for item in summaries:
        disease = item["disease"]
        metrics = pd.read_csv(report_dir / f"metrics_{disease.lower()}.csv", index_col=0)
        lines.extend(
            [
                f"{disease}",
                "-" * len(disease),
                f"Rows after cleaning: {item['rows']:,}",
                f"Engineered features: {item['features']}",
                f"Selected model: {item['best_model']}",
                "",
                "Full metrics table:",
                *_format_metric_table(metrics.round(3)),
                "",
                "Logistic Regression and Random Forest focus:",
                *_format_metric_table(
                    metrics.loc[
                        metrics.index.intersection(["Logistic Regression", "Random Forest"])
                    ].round(3),
                    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV ROC-AUC", "CV Std"],
                ),
                "",
            ]
        )
        for warning in item["warnings"]:
            lines.append(f"Warning: {warning}")
        lines.append("")
    (report_dir / "experiment_report.txt").write_text("\n".join(lines), encoding="utf-8")


def load_model_bundle(disease: str, root: Path | None = None) -> dict[str, Any]:
    dirs = project_dirs(root)
    return joblib.load(dirs["models"] / f"{disease.lower()}_bundle.pkl")


def predict_from_base_row(disease: str, base_row: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    bundle = load_model_bundle(disease, root)
    frame = pd.DataFrame([base_row])
    engineered = engineer_features(disease, frame)
    features = bundle["feature_columns"]
    X = engineered.reindex(columns=features)
    probability = float(bundle["model"].predict_proba(X)[0, 1])
    if probability >= 0.6:
        level = "High"
    elif probability >= 0.3:
        level = "Moderate"
    else:
        level = "Low"
    return {
        "disease": disease,
        "model": bundle["model_name"],
        "probability": probability,
        "risk_level": level,
        "warnings": bundle.get("warnings", []),
    }
