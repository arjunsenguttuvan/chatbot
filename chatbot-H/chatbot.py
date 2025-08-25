import streamlit as st
import pandas as pd
import os

# --- Load dataset safely ---
file_path = os.path.join(os.path.dirname(__file__), "tamilnadu_crop_fertilizer_dataset_with_ph_temp.csv")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"❌ CSV file not found at: {file_path}")
    st.stop()

st.title("🌱 Tamil Nadu Farmer Fertilizer Chatbot")

# Step 1: Get farmer input
crop = st.selectbox("Select your crop:", sorted(df["Crop"].unique()))
acres = st.number_input("Enter total acres:", min_value=1, step=1)

soil_ph = st.number_input("Enter soil pH (optional):", min_value=0.0, max_value=14.0, step=0.1, format="%.1f")
temp = st.number_input("Enter temperature (°C, optional):", min_value=0, max_value=50, step=1)

# Step 2: Generate recommendation
if st.button("Get Fertilizer Recommendation"):
    options = df[df["Crop"] == crop].copy()  # avoid SettingWithCopyWarning
    
    # Select best (least toxic) fertilizer
    options["tox"] = options["Toxicity Score (1=low,5=high)"].astype(int)
    best_option = options.loc[options["tox"].idxmin()]
    
    # Calculate fertilizer requirement
    qty, unit = best_option["Recommended Amount per Acre"].split()
    total_qty = int(qty) * acres
    
    # Show result
    st.subheader("🌾 Fertilizer Recommendation")
    st.write(f"✅ Crop: *{crop}* | Acres: *{acres}*")
    st.write(f"🥇 Recommended Fertilizer: *{best_option['Fertilizer Name']} ({best_option['Type']})*")
    st.write(f"📦 Amount Needed: *{total_qty} {unit}*")
    st.write(f"📝 Notes: {best_option['Notes']}")
    st.write(f"🌡 Ideal Temp: {best_option['Ideal Temp (°C)']} | 🌱 Ideal pH: {best_option['Ideal pH']}")
    
    # Check conditions
    if soil_ph > 0:
        ideal_ph = best_option["Ideal pH"]
        st.warning(f"⚠ Your soil pH: {soil_ph}. Recommended range: {ideal_ph}.")
    if temp > 0:
        ideal_temp = best_option["Ideal Temp (°C)"]
        st.warning(f"⚠ Your temp: {temp}°C. Recommended range: {ideal_temp}.")
