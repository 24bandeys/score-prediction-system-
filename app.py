# Streamlit GUI - Lab 7 Part A
# Name    : Shreyashi Hemant Bandey
# Roll no.: 64
# Section : B3-B2

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(
    page_title="Student Score Prediction", page_icon="🎓", layout="centered"
)

st.title("🎓 Student Score Prediction")
st.write("Predict a student's exam score based on study hours (1 to 100 hours).")

# ---------------------------------------------------------------- Dataset
data = {
    "Hours": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
    ],
    "Score": [
        35, 40, 45, 50, 55,
        60, 62, 65, 68, 70,
        72, 75, 78, 80, 82,
        85, 87, 90, 92, 95,
    ],
}
df = pd.DataFrame(data)

X = df[["Hours"]]
Y = df["Score"]


# ---------------------------------------------------------------- Training
@st.cache_resource
def train_models():
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree Regression": DecisionTreeRegressor(
            random_state=42, max_depth=3
        ),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=3
        ),
    }

    results = []
    for name, mdl in models.items():
        mdl.fit(X_train_s, Y_train)
        pred = mdl.predict(X_test_s)
        mse = mean_squared_error(Y_test, pred)
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(Y_test, pred),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2": r2_score(Y_test, pred),
            }
        )

    metrics_df = pd.DataFrame(results)
    best_model_name = metrics_df.loc[metrics_df["R2"].idxmax(), "Model"]
    return models, scaler, metrics_df, best_model_name


models, scaler, metrics_df, best_model_name = train_models()


def predict_with(model, hours):
    """Scale the input the same way as the training data, then predict."""
    scaled = scaler.transform(pd.DataFrame({"Hours": [hours]}))
    return float(model.predict(scaled)[0])


# ---------------------------------------------------------------- Input
st.subheader("Enter Student Details")
hours = st.number_input(
    "Hours Studied",
    min_value=1.0,
    max_value=100.0,
    value=12.0,
    step=1.0,
)

if st.button("Predict Score", type="primary", use_container_width=True):
    # Prediction from the best model (highest R2 on the test set)
    raw_score = predict_with(models[best_model_name], hours)

    # Enforce strict maximum score cap at 100
    predicted_score = min(100.0, raw_score)

    st.success(f"Predicted Exam Score: {predicted_score:.2f}")
    st.balloons()
    st.write(f"**Hours Studied:** {hours:.0f}")
    st.write(f"**Model Used (Best R²):** {best_model_name}")

    # Prediction from every model, like the notebook's comparison table
    all_preds = pd.DataFrame(
        {
            "MODEL": list(models.keys()),
            "PREDICTED SCORE": [
                min(100.0, predict_with(m, hours)) for m in models.values()
            ],
        }
    ).round(2)
    with st.expander("View predictions from all models"):
        st.dataframe(all_preds, use_container_width=True, hide_index=True)

st.divider()

with st.expander("View Model Performance"):
    st.dataframe(metrics_df.round(4), use_container_width=True, hide_index=True)
    st.caption("R² Score by model")
    st.bar_chart(metrics_df.set_index("Model")["R2"])
    st.caption("RMSE by model")
    st.bar_chart(metrics_df.set_index("Model")["RMSE"])

with st.expander("View Training Dataset"):
    st.dataframe(df, use_container_width=True)
