import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
from model import FRA_CNN  # <-- import your CNN model class
from parser import parse_fra_file  # <-- your universal parser

# Load trained model
model = FRA_CNN(n_classes=8)
model.load_state_dict(torch.load(r"C:\Users\divya\Downloads\fra_cnn_model.pth", map_location="cpu"),strict=False)
model.eval()

# Label map (must match training order)
label_map = {
    0: "healthy",
    1: "winding_fault",
    2: "core_fault",
    3: "insulation_fault",
    4: "loose_connection",
    5: "bushing_fault",
    6: "partial_discharge",
    7: "shorted_turns"
}

st.title("⚡ AI-Driven Transformer FRA Diagnostics")
st.write("Upload an FRA file (CSV / Excel / XML from Omicron, Megger, Doble).")

uploaded_file = st.file_uploader("Upload FRA file", type=["csv", "xls", "xlsx", "xml"])

if uploaded_file is not None:
    # Parse the uploaded FRA file
    df = parse_fra_file(uploaded_file)

    # Plot FRA curve
    st.subheader("FRA Curve")
    fig, ax = plt.subplots()
    ax.semilogx(df["frequency"], df["response"])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Response (dB)")
    ax.grid(True, which="both", ls="--")
    st.pyplot(fig)

    # Prepare data for model
    response = df["response"].values.astype("float32")
    response = (response - response.mean()) / (response.std() + 1e-6)  # normalize
    x = torch.tensor(response).unsqueeze(0).unsqueeze(0)  # (1,1,500)

    # Run inference
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_class = probs.argmax()
    
    st.subheader("Prediction Result")
    st.write(f"**Predicted Fault:** {label_map[pred_class]}")
    st.write("Confidence Scores:")
    for i, prob in enumerate(probs):
        st.write(f"- {label_map[i]}: {prob*100:.2f}%")
