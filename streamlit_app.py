import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

st.title("🛠️ Steel Plates Faults Prediction")
st.markdown("Predict faults in steel plates based on input features.")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Besma1a/steel-plates-data/main/Steel_Plates_Faults.csv"
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

desc = X.describe()

input_data = {}
for feature in X.columns:
    min_val = float(desc[feature]['min'])
    max_val = float(desc[feature]['max'])
    mean_val = float(desc[feature]['mean'])
    val = st.sidebar.slider(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val)
    input_data[feature] = val

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

if st.button("Predict Faults"):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader("Predicted Faults (1 means fault detected)")
    pred_dict = {label_cols[i]: int(prediction[0][i]) for i in range(len(label_cols))}
    pred_df = pd.DataFrame.from_dict(pred_dict, orient='index', columns=['Fault Detected'])
    st.dataframe(pred_df.style.applymap(lambda x: 'background-color : lightgreen' if x==1 else ''))

    st.subheader("Prediction Probabilities")
    proba_dict = {label_cols[i]: float(prediction_proba[i][0][1]) for i in range(len(label_cols))}
    proba_df = pd.DataFrame.from_dict(proba_dict, orient='index', columns=['Probability'])

    chart = alt.Chart(proba_df.reset_index()).mark_bar(color='steelblue').encode(
        x=alt.X('index', sort=None, title='Fault Type'),
        y=alt.Y('Probability', scale=alt.Scale(domain=[0,1]))
    ).properties(width=600)
    st.altair_chart(chart)

st.header("Feature Distributions")
feature_to_plot = st.selectbox("Select feature to visualize", options=X.columns)

fig, ax = plt.subplots()
sns.histplot(df[feature_to_plot], bins=30, kde=True, ax=ax, color='steelblue')
ax.set_title(f'Distribution of {feature_to_plot}')
st.pyplot(fig)

st.sidebar.subheader("Model Info")
st.sidebar.write("Model: Gradient Boosting")
st.sidebar.write("Training on full dataset inside the app")
