"""
A little Streamlit app to try the classifier.

Drop in a CSV with x, y, z columns and it'll show you the points in 3D,
measure the same four things we trained on, and ask the model
(model.joblib) whether it's looking at a car or a pedestrian.

Start it with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# The model that train.py saved
model = joblib.load("model.joblib")


def get_features(points):
    """Same measurements as in training, otherwise the model gets confused."""
    return pd.DataFrame([{
        "length": points["x"].max() - points["x"].min(),
        "width": points["y"].max() - points["y"].min(),
        "height": points["z"].max() - points["z"].min(),
        "num_points": len(points),
    }])


# Title and a quick hint on what to do
st.title("Point Cloud Classifier")
st.write("Upload a CSV file with x, y, z columns to see the object in 3D.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    points = pd.read_csv(uploaded_file)

    # Wrong file? Say so nicely instead of crashing
    if not {"x", "y", "z"}.issubset(points.columns):
        st.error("This file must have x, y and z columns.")
        st.stop()

    st.write("Number of points:", len(points))
    st.dataframe(points.head())

    # 3D plot you can drag around
    fig = px.scatter_3d(points, x="x", y="y", z="z")
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(scene_aspectmode="data")
    st.plotly_chart(fig)

    # Show what we measured
    features = get_features(points)
    st.write("Measured features:")
    st.dataframe(features)

    # And see what the model thinks
    prediction = model.predict(features)[0]

    if prediction == "car":
        st.success("Prediction: Car")
    else:
        st.success("Prediction: Pedestrian")
