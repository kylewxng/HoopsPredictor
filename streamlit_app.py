import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained models & metadata
model_pts = pickle.load(open("model_pts.pkl", "rb"))
model_reb = pickle.load(open("model_reb.pkl", "rb"))
model_ast = pickle.load(open("model_ast.pkl", "rb"))

player_metadata = pd.read_csv("player_metadata.csv")

# Sidebar user selections
st.sidebar.title("HoopsPredictor")
players = sorted(player_metadata["Player"].unique())

player_name = st.sidebar.selectbox("Select Player", players)
home = st.sidebar.radio("Game Location", ("Home", "Away"))
home = 1 if home == "Home" else 0

opponents = sorted(player_metadata["Opp_name"].unique())
opp_name = st.sidebar.selectbox("Opponent", opponents)

st.title("HoopsPredictor: NBA Prop Projection")

# Prepare model input automatically
if st.sidebar.button("Predict"):

    # Most recent row for player
    p_data = player_metadata[player_metadata["Player"] == player_name].iloc[-1].copy()

    # Set home/away and opponent numeric id
    p_data["Home"] = home
    p_data["Opp"] = player_metadata[player_metadata["Opp_name"] == opp_name]["Opp"].iloc[0]

    # Build model input vector — drop non-feature columns
    model_input = p_data.drop(["Player", "Tm_name", "Opp_name"]).values.reshape(1, -1)

    # ML model predictions
    pred_pts = model_pts.predict(model_input)[0]
    pred_reb = model_reb.predict(model_input)[0]
    pred_ast = model_ast.predict(model_input)[0]

    # Show results
    st.subheader(f"Prediction for {player_name}")
    st.write(f"**Points:** {pred_pts:.1f}")
    st.write(f"**Rebounds:** {pred_reb:.1f}")
    st.write(f"**Assists:** {pred_ast:.1f}")

    st.success("Prediction complete!")

st.write("---")
st.caption("👨‍💻 Powered by Machine Learning | Built by Brandon")
