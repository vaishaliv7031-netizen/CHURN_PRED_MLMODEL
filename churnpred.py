import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    classification_report,
    roc_auc_score,
    roc_curve
)

# -------------------------------
# 1. Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Train and compare Random Forest, Logistic Regression, "
    "and Gradient Boosting models."
)

# -------------------------------
# 2. Load Dataset
# -------------------------------
st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Churn_Modelling.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Upload your Churn_Modelling.csv file in the sidebar to begin.")
    st.stop()

# Check required columns before preprocessing
required_columns = [
    "RowNumber", "CustomerId", "Surname",
    "Geography", "Gender", "Exited"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df))
col2.metric("Total Features", df.shape[1])
col3.metric("Churn Rate", f"{df['Exited'].mean() * 100:.2f}%")

# -------------------------------
# 3. Train Models
# -------------------------------
if st.button("🚀 Train and Evaluate Models", type="primary"):

    with st.spinner("Training models. Please wait..."):

        try:
            # Work on a copy
            data = df.copy()

            # Remove identifier columns
            data.drop(
                ["RowNumber", "CustomerId", "Surname"],
                axis=1,
                inplace=True
            )

            # Encode categorical features
            data = pd.get_dummies(
                data,
                columns=["Geography"],
                drop_first=True,
                dtype=int
            )

            le = LabelEncoder()
            data["Gender"] = le.fit_transform(data["Gender"])

            # Features and target
            X = data.drop("Exited", axis=1)
            y = data["Exited"]

            if y.nunique() != 2:
                st.error("Exited must contain two classes: 0 and 1.")
                st.stop()

            if y.value_counts().min() < 2:
                st.error("Each target class needs at least two samples.")
                st.stop()

            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            # Scale features for Logistic Regression and GB
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define models
            models = {
                "Random Forest": (
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42,
                        class_weight="balanced"
                    ),
                    X_train,
                    X_test
                ),
                "Logistic Regression": (
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000
                    ),
                    X_train_scaled,
                    X_test_scaled
                ),
                "Gradient Boosting": (
                    GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42
                    ),
                    X_train_scaled,
                    X_test_scaled
                )
            }

            results = {}
            reports = {}
            matrices = {}
            roc_data = {}

            # Train and evaluate each model
            for name, (model, Xtr, Xte) in models.items():

                model.fit(Xtr, y_train)

                probabilities = model.predict_proba(Xte)[:, 1]

                # Preserve your Logistic Regression threshold
                if name == "Logistic Regression":
                    predictions = (
                        probabilities > 0.3
                    ).astype(int)
                else:
                    predictions = model.predict(Xte)

                results[name] = {
                    "Accuracy": accuracy_score(
                        y_test, predictions
                    ),
                    "Recall": recall_score(
                        y_test, predictions, zero_division=0
                    ),
                    "ROC AUC": roc_auc_score(
                        y_test, probabilities
                    )
                }

                reports[name] = classification_report(
                    y_test,
                    predictions,
                    output_dict=True,
                    zero_division=0
                )

                matrices[name] = confusion_matrix(
                    y_test, predictions, labels=[0, 1]
                )

                fpr, tpr, _ = roc_curve(
                    y_test, probabilities
                )

                roc_data[name] = (fpr, tpr)

            # Save results for the rest of this run
            st.session_state["churn_results"] = results
            st.session_state["churn_reports"] = reports
            st.session_state["churn_matrices"] = matrices
            st.session_state["churn_roc"] = roc_data
            st.session_state["churn_features"] = list(X.columns)

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

# -------------------------------
# 4. Display Evaluation Results
# -------------------------------
if "churn_results" in st.session_state:

    st.success("All three models trained successfully!")

    st.subheader("Model Comparison")

    results_df = pd.DataFrame(
        st.session_state["churn_results"]
    ).T

    st.dataframe(
        results_df.style.format("{:.3f}"),
        use_container_width=True
    )

    st.bar_chart(results_df)

    # Select a model for detailed results
    selected_model = st.selectbox(
        "View detailed evaluation",
        list(st.session_state["churn_results"].keys())
    )

    st.subheader(f"{selected_model}: Evaluation")

    metrics = st.session_state["churn_results"][selected_model]

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    c2.metric("Recall", f"{metrics['Recall']:.3f}")
    c3.metric("ROC AUC", f"{metrics['ROC AUC']:.3f}")

    # Confusion matrix
    st.write("**Confusion Matrix**")
    cm = st.session_state["churn_matrices"][selected_model]

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    image = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion Matrix"
    )

    for i in range(2):
        for j in range(2):
            ax_cm.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", color="black"
            )

    fig_cm.colorbar(image, ax=ax_cm)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # Classification report
    st.write("**Classification Report**")

    report_df = pd.DataFrame(
        st.session_state["churn_reports"][selected_model]
    ).T

    st.dataframe(report_df.round(3), use_container_width=True)

    # ROC curve for all models
    st.subheader("ROC Curve Comparison")

    fig_roc, ax_roc = plt.subplots(figsize=(8, 5))

    for name, (fpr, tpr) in st.session_state["churn_roc"].items():
        auc = st.session_state["churn_results"][name]["ROC AUC"]
        ax_roc.plot(
            fpr, tpr,
            label=f"{name} (AUC = {auc:.3f})"
        )

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve Comparison")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)

    st.pyplot(fig_roc)
    plt.close(fig_roc)

    st.caption(
        "Logistic Regression uses a 0.3 probability threshold; "
        "the other models use their default classification threshold."
    )
