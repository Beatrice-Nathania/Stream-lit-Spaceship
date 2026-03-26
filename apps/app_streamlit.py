"""
Session 04 – Streamlit App
Serve the trained Iris Random Forest classifier via a web UI.
Run with: streamlit run app_streamlit.py
"""

import sys
import pathlib as path
import sys
import os
from pathlib import Path

curr_dir = Path(__file__).resolve().parent
root_dir = curr_dir.parent

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
import streamlit as st
import pandas as pd

from config.config import ARTIFACT_PIPELINE
from src.utils.io import load_artifact
from src.feature_engineering.feat_eng import feature_engineering

@st.cache_resource
def load_pipeline():
    try:
        return load_artifact(ARTIFACT_PIPELINE)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


def main():
    st.title("ASG 04 MD - Beatrice - Spaceship Titanic Model Deployment")

    st.subheader("Passenger Information")

    model = load_pipeline()
    
    Name = st.text_input("Name", "Palis")
    id_num = st.text_input("PassengerId", "0001_01")
    cabin = st.text_input("Cabin", "B/10/P")
    HomePlanet = st.selectbox("Home Planet", ["Europa", "Earth", "Mars"])
    cryosleep = st.checkbox("Cryo Sleep")
    destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
    age = st.number_input("Age", 0, 100, 20)
    vip = st.checkbox("VIP")
    room_service = st.number_input("Room service", 0.0, 20000.0,5.0)
    FoodCourt = st.number_input("Food Court", 0.0, 20000.0, 10.0)
    ShoppingMall = st.number_input("Shopping Mall", 0.0, 20000.0, 5.0)
    Spa = st.number_input("Spa", 0.0, 20000.0, 5.0)
    VRDeck = st.number_input("VR Deck", 0.0, 20000.0, 8.0)

    if st.button("Make Prediction"):
        features = [[Name, id_num, HomePlanet, cryosleep, destination, vip, age, room_service, FoodCourt, ShoppingMall, Spa, VRDeck, cabin]]
        features = pd.DataFrame(features, columns= ['Name','PassengerId','HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin'])
        features = feature_engineering(features)
        result = model.predict(features)[0]
        if result == 1:
            st.success("Passenger Transported :D ✅")
        else:
            st.error("Passenger Not Transported :( ❌")

if __name__ == "__main__":
    main()
