import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load Models
model_pts = pickle.load(open("model_pts.pkl", "rb"))
model_reb = pickle.load(open("model_reb.pkl", "rb"))
model_ast = pickle.load(open("model_ast.pkl", "rb"))

# Load Metadata (used to populate dropdowns and structure features)
player_meta = pd.read_csv("player_metadata.csv")

# Unique dropdown values
players = sorted(player_meta.index.astype(str).tolist())
teams = sorted(player_meta['Tm'].unique().tolist())
opponents = sorted(player_meta['Opp'].unique().tolist())

st.title("🏀 HoopsPredictor")
st.write("Predict NBA Player Props: **Points**, **Rebounds**, and **Assists**")

# Sidebar Inputs
st.sidebar.header("Game Inputs")

# Select Player
player = st.sidebar.selectbox("Select Player ID (Index)", players)
player_idx = int(player)

# Game Inputs
mp = st.sidebar.slider("Minutes Played", 10, 45, 32)
fg = st.sidebar.slider("FG Made", 0, 20, 8)
fga = st.sidebar.slider("FG Attempts", 0, 30, 16)
p3 = st.sidebar.slider("3PM", 0, 12, 3)
p3a = st.sidebar.slider("3PA", 0, 15, 8)
ft = st.sidebar.slider("FT Made", 0, 20, 5)
fta = st.sidebar.slider("FTA", 0, 20, 6)
home = st.sidebar.radio("Home or Away", ("Home", "Away"))
home = 1 if home == "Home" else 0

opp = st.sidebar.selectbox("Opponent (numeric ID)", opponents)
opp = int(opp)

# Build input row matching model features
columns = player_meta.columns  # ensures correct order
input_data = np.zeros((1, len(columns)))

input_data[0, :] = [
    (mp if col == "MP" else
     fg if col == "FG" else
     fga if col == "FGA" else
     p3 if col == "3P" else
     p3a if col == "3PA" else
     ft if col == "FT" else
     fta if col == "FTA" else
     home if col == "Home" else
     opp if col == "Opp" else
     player_meta.iloc[player_idx][col])
    for col in columns
]

# Prediction Button
if st.sidebar.button("Predict"):
    pred_points = model_pts.predict(input_data)[0]
    pred_rebounds = model_reb.predict(input_data)[0]
    pred_assists = model_ast.predict(input_data)[0]

    st.subheader("📊 Predictions")
    st.write(f"**Projected Points:** {pred_points:.1f}")
    st.write(f"**Projected Rebounds:** {pred_rebounds:.1f}")
    st.write(f"**Projected Assists:** {pred_assists:.1f}")
