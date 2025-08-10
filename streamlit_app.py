import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

st.title("🛠️ Steel Plates Faults Prediction")


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Besma1a/steel-plates-data/refs/heads/main/Steel_Plates_Faults.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip().str.replace('\t', '', regex=True)
    return df

df = load_data()

label_cols = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 
              'Dirtiness', 'Bumps', 'Other_Faults']

X = df.drop(columns=label_cols)
y = df[label_cols]

@st.cache_resource
def train_model():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = MultiOutputClassifier(
        GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    )
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()


st.sidebar.header("Input Features")
input_data = {}
for feature in X.columns:
    val = st.sidebar.number_input(f"Input {feature}", value=0.0)
    input_data[feature] = val

input_df = pd.DataFrame([input_data])


input_scaled = scaler.transform(input_df)

if st.button("Predict Faults"):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader("Predicted Faults (1 means fault detected)")
    pred_dict = {label_cols[i]: int(prediction[0][i]) for i in range(len(label_cols))}
    st.write(pred_dict)

    st.subheader("Prediction Probabilities")
    proba_dict = {label_cols[i]: float(prediction_proba[i][0][1]) for i in range(len(label_cols))}
    st.write(proba_dict)

st.sidebar.subheader("Model Info")
st.sidebar.write("Model: Gradient Boosting")
st.sidebar.write("Training on full dataset inside the app")

