# fertilizer_chatbot.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor   # ✅ changed here
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Data Loading
# ----------------------------
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Convert cost and efficiency to numeric safely
    df['Cost_per_kg'] = pd.to_numeric(df['Cost_per_kg'], errors='coerce')
    df['Efficiency'] = pd.to_numeric(df['Efficiency'], errors='coerce')
    # Handle missing values only on numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean(numeric_only=True))
    return df

# ----------------------------
# Compute Fertilizer Score
# ----------------------------
def compute_score(df, weight_eff=0.6, weight_cost=0.4):
    df = df.copy()
    df['norm_eff'] = (df['Efficiency'] - df['Efficiency'].min()) / (df['Efficiency'].max() - df['Efficiency'].min() + 1e-9)
    df['norm_cost'] = (df['Cost_per_kg'] - df['Cost_per_kg'].min()) / (df['Cost_per_kg'].max() - df['Cost_per_kg'].min() + 1e-9)
    df['computed_score'] = (weight_eff * df['norm_eff'] + weight_cost * (1 - df['norm_cost'])) / (weight_eff + weight_cost)
    return df

# ----------------------------
# Train ML Model
# ----------------------------
def train_model(df):
    features = df[['Cost_per_kg', 'Efficiency']]
    target = df['computed_score']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # ✅ replaced RandomForest with GradientBoosting
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    st.sidebar.write("### Model Performance")
    st.sidebar.write(f"MAE: {mae:.4f}")
    st.sidebar.write(f"RMSE: {rmse:.4f}")
    st.sidebar.write(f"R²: {r2:.4f}")

    return model

# ----------------------------
# Recommendation Function
# ----------------------------
def recommend_fertilizers(df, crop_name, model=None, top_n=5, weight_eff=0.6, weight_cost=0.4, prefer_type=None):
    df_scored = compute_score(df, weight_eff, weight_cost)
    crop_df = df_scored[df_scored['Crop'].str.lower() == crop_name.lower()].copy()

    if prefer_type:
        crop_df = crop_df[crop_df['Fertiliser_Type'].str.lower() == prefer_type.lower()]

    if model:
        crop_df['model_score'] = model.predict(crop_df[['Cost_per_kg', 'Efficiency']])
    else:
        crop_df['model_score'] = crop_df['computed_score']

    crop_df = crop_df.sort_values(by='computed_score', ascending=False)
    return crop_df.head(top_n)

# ----------------------------
# Streamlit App
# ----------------------------
st.title("🌱 Fertilizer Recommendation Chatbot")

uploaded_file = st.file_uploader("Upload fertilizer.csv", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    df = compute_score(df)
    model = train_model(df)

    # Chatbot-like interface
    st.subheader("💬 Chat with the Fertilizer Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    query = st.chat_input("Ask me: e.g. Recommend for Paddy (rice) organic")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Basic parsing
        words = query.lower().split()
        crop_name = None
        prefer_type = None
        for crop in df['Crop'].unique():
            if crop.lower() in query.lower():
                crop_name = crop
                break
        if "organic" in words:
            prefer_type = "Organic"
        elif "chemical" in words:
            prefer_type = "Chemical"

        if crop_name:
            recs = recommend_fertilizers(df, crop_name, model=model, top_n=5, prefer_type=prefer_type)
            response = f"Here are the top fertilizers for **{crop_name}**"
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
                st.dataframe(recs[['Crop','Fertiliser_Type','Fertiliser_Name','Brand','Cost_per_kg','Efficiency','computed_score','model_score']])
        else:
            response = "Sorry, I couldn’t find that crop in the dataset."
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
