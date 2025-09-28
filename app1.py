import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

st.set_page_config(page_title="FRA Digital Twin", layout="wide")

st.title("⚡ FRA Digital Twin Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload FRA Test File", type=["csv", "xlsx"])

# Load healthy reference
df_healthy = pd.read_csv("healthy_fra.csv")
df_healthy.columns = [c.strip().lower() for c in df_healthy.columns]

if uploaded_file:
    # Load test file
    if uploaded_file.name.endswith("xlsx"):
        df_test = pd.read_excel(uploaded_file)
    else:
        df_test = pd.read_csv(uploaded_file)

    # Normalize column names
    df_test.columns = [c.strip().lower() for c in df_test.columns]

    # --- Detect frequency/response columns separately for both ---
    freq_col_h = [c for c in df_healthy.columns if "freq" in c][0]
    resp_col_h = [c for c in df_healthy.columns if "resp" in c][0]

    freq_col_t = [c for c in df_test.columns if "freq" in c][0]
    resp_col_t = [c for c in df_test.columns if "resp" in c or "mag" in c or "amp" in c][0]

    # --- Plot overlay ---
    st.subheader("FRA Overlay: Healthy vs Test")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(df_healthy[freq_col_h], df_healthy[resp_col_h], label="Healthy", color="green")
    ax.semilogx(df_test[freq_col_t], df_test[resp_col_t], label="Test", color="red")

    # --- Simple AI fault detection (difference threshold) ---
    diff = np.abs(np.interp(df_test[freq_col_t], df_healthy[freq_col_h], df_healthy[resp_col_h]) - df_test[resp_col_t])
    fault_indices = np.where(diff > 5)[0]  # mark where deviation > 5 dB

    if len(fault_indices) > 0:
        f_start = df_test[freq_col_t].iloc[fault_indices[0]]
        f_end = df_test[freq_col_t].iloc[fault_indices[-1]]
        ax.axvspan(f_start, f_end, color="orange", alpha=0.3, label="Suspected Fault")
        fault_band = (f_start, f_end)
    else:
        fault_band = None

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Response (dB)")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    st.pyplot(fig)

    # --- Fault interpretation ---
    st.subheader("AI Diagnostic")
    if fault_band:
        st.warning(f"⚠️ Suspected anomaly between **{fault_band[0]:.1f} Hz – {fault_band[1]:.1f} Hz**")
    else:
        st.success("✅ No major fault detected. Curve matches healthy profile.")

    # --- Percentages for 7 categories (placeholder logic) ---
    st.subheader("Category Confidence Scores")

    categories = [
        "Healthy",
        "Winding Fault",
        "Core Fault",
        "Insulation Fault",
        "Loose Connection",
        "Bushing Fault",
        "Partial Discharge",
        "Shorted Turns"
    ]

    # Heuristic scores (replace later with real CNN model output)
    scores = np.random.rand(len(categories))  # random for demo
    if fault_band:
        if fault_band[1] < 1e3:
            scores[2] += 3  # core fault bias
        elif fault_band[1] < 1e5:
            scores[1] += 3  # winding fault bias
        else:
            scores[5] += 3  # bushing fault bias
    else:
        scores[0] += 5  # healthy bias

    # Normalize to 100%
    probs = scores / scores.sum() * 100

    # Display
    for cat, p in zip(categories, probs):
        st.write(f"- {cat}: {p:.2f}%")

    # --- Digital Twin ---
    st.subheader("Digital Twin (3D)")
    with open("twin.html", "r") as f:
        st.components.v1.html(f.read(), height=500)
else:
    st.info("Please upload an FRA test file (CSV/XLSX) to begin.")
