import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

# ---------- LOAD MODEL + SCALER (Fixed Cloud Paths) ----------
# ---------- LOAD MODEL + SCALER (Point to the app folder) ----------
# Get the absolute path of the directory containing app.py (notebook/app/)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Point directly to the files inside the same folder as app.py
model_path = os.path.join(APP_DIR, "diabetes_model.pkl")
scaler_path = os.path.join(APP_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ---------- LOAD DATA (for comparison charts) ----------
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    names=columns
)
cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_to_fix:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# ---------- TITLE ----------
st.title("🩺 Diabetes Risk Predictor")
st.caption("A logistic regression model trained on the Pima Indians Diabetes dataset, with class-balancing and L2 regularization.")

tab1, tab2 = st.tabs(["🔍 Predict", "📊 Dataset Insights"])

# ================= TAB 1: PREDICTION =================
with tab1:
    st.subheader("Enter patient details")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 1)
        glucose = st.slider("Glucose", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure", 0, 130, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    with col2:
        insulin = st.slider("Insulin", 0, 850, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age", 18, 100, 30)

    if st.button("Predict Risk", type="primary"):
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]], columns=feature_names)
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]  # prob of class 1

        st.divider()
        res_col1, res_col2 = st.columns([1, 1])

        # ---- Result Verdict ----
        with res_col1:
            if prediction == 1:
                st.error(f"### ⚠️ High Risk of Diabetes")
            else:
                st.success(f"### ✅ Low Risk of Diabetes")
            st.metric("Predicted Probability", f"{probability*100:.1f}%")

        # ---- Gauge Chart ----
        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Diabetes Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if probability > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d4f4dd"},
                        {'range': [30, 60], 'color': "#fff3cd"},
                        {'range': [60, 100], 'color': "#f8d7da"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': probability * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        # ---- Feature Importance (model coefficients) ----
        st.subheader("What drives this prediction?")
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.coef_[0]
        }).sort_values("Importance", key=abs, ascending=True)

        fig_importance = px.bar(
            coef_df, x="Importance", y="Feature", orientation='h',
            color="Importance", color_continuous_scale="RdBu",
            title="Feature Influence (Logistic Regression Coefficients)"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        st.caption("Positive = pushes prediction toward diabetes. Negative = pushes toward no diabetes.")

        # ---- Comparison vs population averages ----
        st.subheader("How you compare to dataset averages")
        diabetic_avg = df[df["Outcome"] == 1][feature_names].mean()
        nondiabetic_avg = df[df["Outcome"] == 0][feature_names].mean()

        compare_df = pd.DataFrame({
            "Feature": feature_names,
            "You": input_data.iloc[0].values,
            "Avg (Diabetic)": diabetic_avg.values,
            "Avg (Non-Diabetic)": nondiabetic_avg.values
        })

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(name="You", x=compare_df["Feature"], y=compare_df["You"]))
        fig_compare.add_trace(go.Bar(name="Avg (Diabetic)", x=compare_df["Feature"], y=compare_df["Avg (Diabetic)"]))
        fig_compare.add_trace(go.Bar(name="Avg (Non-Diabetic)", x=compare_df["Feature"], y=compare_df["Avg (Non-Diabetic)"]))
        fig_compare.update_layout(barmode='group', height=450)
        st.plotly_chart(fig_compare, use_container_width=True)

# ================= TAB 2: DATASET INSIGHTS =================
with tab2:
    st.subheader("Explore the training data")

    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("X-axis", feature_names, index=1)
    with c2:
        y_axis = st.selectbox("Y-axis", feature_names, index=5)

    fig_scatter = px.scatter(
        df, x=x_axis, y=y_axis, color=df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"}),
        title=f"{x_axis} vs {y_axis}", opacity=0.7
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    fig_hist = px.histogram(
        df, x="Glucose", color=df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"}),
        barmode="overlay", nbins=30, title="Glucose Distribution by Outcome"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Class Distribution")
    fig_pie = px.pie(df, names=df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"}),
                      title="Diabetes vs No Diabetes in Dataset")
    st.plotly_chart(fig_pie, use_container_width=True)