import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Convert cost and efficiency to numeric safely
    df['Cost_per_kg'] = pd.to_numeric(df['Cost_per_kg'], errors='coerce')
    df['Efficiency'] = pd.to_numeric(df['Efficiency'], errors='coerce')
    # Handle missing values only on numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean(numeric_only=True))
    return df

# Compute fertilizer score based on efficiency and cost
def compute_score(df, weight_eff=0.6, weight_cost=0.4):
    df = df.copy()
    # Normalize cost and efficiency
    df['norm_eff'] = (df['Efficiency'] - df['Efficiency'].min()) / (df['Efficiency'].max() - df['Efficiency'].min() + 1e-9)
    df['norm_cost'] = (df['Cost_per_kg'] - df['Cost_per_kg'].min()) / (df['Cost_per_kg'].max() - df['Cost_per_kg'].min() + 1e-9)
    # Higher efficiency is better, lower cost is better
    df['computed_score'] = (weight_eff * df['norm_eff'] + weight_cost * (1 - df['norm_cost'])) / (weight_eff + weight_cost)
    return df

# Train ML model to predict score
def train_model(df):
    features = df[['Cost_per_kg', 'Efficiency']]
    target = df['computed_score']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print("Model performance:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return model

# Recommend fertilizers
def recommend_fertilizers(df, crop_name, model=None, top_n=5, weight_eff=0.6, weight_cost=0.4, prefer_type=None):
    df_scored = compute_score(df, weight_eff, weight_cost)
    crop_df = df_scored[df_scored['Crop'] == crop_name].copy()

    if prefer_type:
        crop_df = crop_df[crop_df['Fertiliser_Type'].str.lower() == prefer_type.lower()]

    if model:
        crop_df['model_score'] = model.predict(crop_df[['Cost_per_kg', 'Efficiency']])
    else:
        crop_df['model_score'] = crop_df['computed_score']

    crop_df = crop_df.sort_values(by='computed_score', ascending=False)
    return crop_df.head(top_n)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fertilizer_recommender.py fertilizer.csv")
        sys.exit(1)

    filepath = sys.argv[1]
    df = load_data(filepath)
    df = compute_score(df)
    model = train_model(df)

    crop = "Paddy (rice)"
    print(f"\nTop recommendations for {crop}:")
    recs = recommend_fertilizers(df, crop, model=model, top_n=5)
    print(recs[['Crop','Fertiliser_Type','Fertiliser_Name','Brand','Cost_per_kg','Efficiency','computed_score','model_score']])
