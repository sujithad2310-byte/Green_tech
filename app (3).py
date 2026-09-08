import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Green-Tech Sustainability Classifier",
    page_icon="🌱",
    layout="wide",
)

FEATURES = ["carbon_emissions", "energy_output", "renewability_index", "cost_efficiency"]
TARGET = "sustainability"

FEATURE_BOUNDS = {
    "carbon_emissions": (50.0, 400.0, "Emissions in hypothetical units. Lower is better for sustainability."),
    "energy_output": (100.0, 1000.0, "Energy produced, in hypothetical units."),
    "renewability_index": (0.0, 1.0, "0 = fully non-renewable, 1 = fully renewable."),
    "cost_efficiency": (0.5, 5.0, "Lower score = more cost efficient."),
}


# ----------------------------------------------------------------------------
# Data + model (cached so this only runs once per session / when inputs change)
# ----------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_data(num_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Recreates the exact synthetic 'green tech' dataset from the notebook."""
    rng = np.random.RandomState(seed)
    carbon_emissions = rng.uniform(50, 400, num_samples)
    energy_output = rng.uniform(100, 1000, num_samples)
    renewability_index = rng.uniform(0, 1, num_samples)
    cost_efficiency = rng.uniform(0.5, 5, num_samples)

    sustainability = [
        1 if (e < 200 and r > 0.5 and c < 3) else 0
        for e, r, c in zip(carbon_emissions, renewability_index, cost_efficiency)
    ]

    return pd.DataFrame(
        {
            "carbon_emissions": carbon_emissions,
            "energy_output": energy_output,
            "renewability_index": renewability_index,
            "cost_efficiency": cost_efficiency,
            "sustainability": sustainability,
        }
    )


@st.cache_resource
def train_model(df: pd.DataFrame, test_size: float, random_state: int):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, X_train, X_test, y_train, y_test, y_pred, y_proba


# ----------------------------------------------------------------------------
# Sidebar — data source & training controls
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Data & Model Settings")

data_source = st.sidebar.radio(
    "Dataset source",
    ["Use built-in synthetic data (from notebook)", "Upload my own CSV"],
)

if data_source == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader("CSV with columns: " + ", ".join(FEATURES + [TARGET]), type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing_cols = set(FEATURES + [TARGET]) - set(data.columns)
        if missing_cols:
            st.sidebar.error(f"Missing required columns: {missing_cols}")
            st.stop()
        data = data.dropna()
    else:
        st.sidebar.info("Upload a CSV, or switch back to the built-in dataset.")
        data = generate_synthetic_data()
else:
    num_samples = st.sidebar.slider("Number of synthetic samples", 50, 1000, 100, step=50)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    data = generate_synthetic_data(num_samples=num_samples, seed=seed)

test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, step=0.05)
random_state = st.sidebar.number_input("Train/test split random state", value=42, step=1)

model, X_train, X_test, y_train, y_test, y_pred, y_proba = train_model(data, test_size, random_state)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This app reproduces the Logistic Regression notebook: synthetic green-tech data → "
    "train/test split → LogisticRegression → evaluation, plus an interactive predictor."
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🌱 Green-Tech Sustainability Classifier")
st.write(
    "A Logistic Regression model that predicts whether a green-tech scenario is "
    "**sustainable** based on carbon emissions, energy output, renewability index, "
    "and cost efficiency."
)

tab_predict, tab_explore, tab_performance, tab_about = st.tabs(
    ["🔮 Predict", "📊 Explore Data", "📈 Model Performance", "ℹ️ About"]
)

# ----------------------------------------------------------------------------
# Tab 1: Interactive prediction
# ----------------------------------------------------------------------------
with tab_predict:
    st.subheader("Try it yourself")
    st.write("Adjust the sliders to describe a scenario, then see the model's prediction live.")

    col1, col2 = st.columns([1, 1])

    input_values = {}
    with col1:
        for feat in FEATURES:
            lo, hi, help_text = FEATURE_BOUNDS[feat]
            default = float(data[feat].mean())
            step = (hi - lo) / 100
            input_values[feat] = st.slider(
                feat.replace("_", " ").title(),
                min_value=float(lo),
                max_value=float(hi),
                value=float(np.clip(default, lo, hi)),
                step=float(step),
                help=help_text,
            )

    input_df = pd.DataFrame([input_values])[FEATURES]
    proba = model.predict_proba(input_df)[0]
    prediction = model.predict(input_df)[0]

    with col2:
        st.markdown("#### Prediction")
        if prediction == 1:
            st.success("✅ **Sustainable**")
        else:
            st.error("❌ **Not Sustainable**")

        st.metric("Probability of being Sustainable", f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]))

        st.markdown("#### Input summary")
        st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)

    st.markdown("---")
    st.subheader("Batch prediction")
    st.write("Upload a CSV with the same feature columns to score many rows at once.")
    batch_file = st.file_uploader(
        "CSV with columns: " + ", ".join(FEATURES), type=["csv"], key="batch_upload"
    )
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing = set(FEATURES) - set(batch_df.columns)
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            batch_df["predicted_sustainability"] = model.predict(batch_df[FEATURES])
            batch_df["probability_sustainable"] = model.predict_proba(batch_df[FEATURES])[:, 1]
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "⬇️ Download predictions as CSV",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

# ----------------------------------------------------------------------------
# Tab 2: Explore the data
# ----------------------------------------------------------------------------
with tab_explore:
    st.subheader("Dataset preview")
    st.dataframe(data.head(20), use_container_width=True)
    st.caption(f"Full dataset: {data.shape[0]} rows × {data.shape[1]} columns")

    st.subheader("Class balance")
    counts = data[TARGET].value_counts().rename({0: "Not Sustainable", 1: "Sustainable"})
    fig, ax = plt.subplots(figsize=(4, 3))
    counts.plot(kind="bar", color=["#d9534f", "#5cb85c"], ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    st.pyplot(fig)

    st.subheader("Feature distributions by class")
    feat_choice = st.selectbox("Choose a feature", FEATURES)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.histplot(data=data, x=feat_choice, hue=TARGET, kde=True, palette=["#d9534f", "#5cb85c"], ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation heatmap")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.heatmap(data[FEATURES + [TARGET]].corr(), annot=True, cmap="Blues", ax=ax3)
    st.pyplot(fig3)

# ----------------------------------------------------------------------------
# Tab 3: Model performance
# ----------------------------------------------------------------------------
with tab_performance:
    acc = accuracy_score(y_test, y_pred)
    st.subheader("Summary metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc*100:.1f}%")
    m2.metric("Train samples", len(X_train))
    m3.metric("Test samples", len(X_test))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig4, ax4 = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Sustainable", "Sustainable"],
            yticklabels=["Not Sustainable", "Sustainable"],
            ax=ax4,
        )
        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("Actual")
        st.pyplot(fig4)

    with col_b:
        st.markdown("#### ROC Curve")
        if y_test.nunique() > 1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig5, ax5 = plt.subplots(figsize=(4, 4))
            ax5.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#5cb85c")
            ax5.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax5.set_xlabel("False Positive Rate")
            ax5.set_ylabel("True Positive Rate")
            ax5.legend(loc="lower right")
            st.pyplot(fig5)
        else:
            st.info("ROC curve needs both classes present in the test set.")

    st.markdown("#### Classification Report")
    report = classification_report(
        y_test, y_pred, target_names=["Not Sustainable", "Sustainable"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.markdown("#### Feature Coefficients")
    coefficients = pd.DataFrame(model.coef_.T, index=FEATURES, columns=["Coefficient"]).sort_values(
        "Coefficient", ascending=False
    )
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    colors = ["#5cb85c" if v > 0 else "#d9534f" for v in coefficients["Coefficient"]]
    ax6.barh(coefficients.index, coefficients["Coefficient"], color=colors)
    ax6.set_xlabel("Coefficient value")
    st.pyplot(fig6)
    st.caption(
        "Positive coefficients push the prediction toward 'Sustainable'; "
        "negative coefficients push toward 'Not Sustainable'."
    )
    st.dataframe(coefficients, use_container_width=True)

# ----------------------------------------------------------------------------
# Tab 4: About
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
        ### About this app

        This Streamlit app is an interactive wrapper around the notebook
        **`3___Logistic_regression_.ipynb`**. It reproduces the same pipeline:

        1. Generate (or upload) green-tech scenario data with features:
           `carbon_emissions`, `energy_output`, `renewability_index`, `cost_efficiency`.
        2. Split into train/test sets.
        3. Fit a `sklearn.linear_model.LogisticRegression` model.
        4. Evaluate with accuracy, a confusion matrix, ROC/AUC, and a classification report.
        5. Let you interactively predict on new inputs, one at a time or in batch via CSV upload.

        **Note on the synthetic label rule** — in the original notebook, a scenario is
        labeled *Sustainable* when `carbon_emissions < 200`, `renewability_index > 0.5`,
        and `cost_efficiency < 3`. This is a synthetic demo rule, not a real-world standard.
        """
    )
