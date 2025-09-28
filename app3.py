import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="FRA Digital Twin", layout="wide")

st.title("⚡ FRA Digital Twin Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload FRA Test File", type=["csv", "xlsx"])

# Load healthy reference (must exist in same folder)
df_healthy = pd.read_csv("healthy_fra.csv")
df_healthy.columns = [c.strip().lower() for c in df_healthy.columns]

fault_region = None
fault_component = None

if uploaded_file:
    if uploaded_file.name.endswith("xlsx"):
        df_test = pd.read_excel(uploaded_file)
    else:
        df_test = pd.read_csv(uploaded_file)

    df_test.columns = [c.strip().lower() for c in df_test.columns]

    freq_col_h = [c for c in df_healthy.columns if "freq" in c][0]
    resp_col_h = [c for c in df_healthy.columns if "resp" in c][0]

    freq_col_t = [c for c in df_test.columns if "freq" in c][0]
    resp_col_t = [c for c in df_test.columns if any(k in c for k in ["resp","mag","amp"])][0]

    # Overlay FRA curves
    st.subheader("FRA Overlay: Healthy vs Test")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(df_healthy[freq_col_h], df_healthy[resp_col_h], label="Healthy", color="green")
    ax.semilogx(df_test[freq_col_t], df_test[resp_col_t], label="Test", color="red")

    # Detect anomaly (simple rule: >5 dB difference)
    diff = np.abs(np.interp(df_test[freq_col_t], df_healthy[freq_col_h], df_healthy[resp_col_h]) - df_test[resp_col_t])
    fault_indices = np.where(diff > 5)[0]

    if len(fault_indices) > 0:
        f_start = df_test[freq_col_t].iloc[fault_indices[0]]
        f_end = df_test[freq_col_t].iloc[fault_indices[-1]]
        ax.axvspan(f_start, f_end, color="orange", alpha=0.3, label="Suspected Fault")
        fault_region = (f_start, f_end)

        # Map fault to component
        if f_end < 1e3:
            fault_component = "core"
        elif f_end < 1e5:
            fault_component = "winding"
        else:
            fault_component = "bushing"

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Response (dB)")
    ax.grid(True, which="both", ls="--")
    ax.legend()
    st.pyplot(fig)

    # Diagnostic message
    st.subheader("AI Diagnostic")
    if fault_region:
        st.warning(f"⚠️ Anomaly detected between {fault_region[0]:.1f} Hz – {fault_region[1]:.1f} Hz → Likely {fault_component.upper()} fault")
    else:
        st.success("✅ No major fault detected. Curve matches healthy profile.")

# --------- Digital Twin (Plotly 3D) ----------
st.subheader("Digital Twin (3D)")

# Default colors
core_color = "gray"
winding_color = "green"
bushing_color = "blue"

# Highlight faulty component
if fault_component == "core":
    core_color = "red"
elif fault_component == "winding":
    winding_color = "red"
elif fault_component == "bushing":
    bushing_color = "red"

fig3d = go.Figure()

# Core (cube)
fig3d.add_trace(go.Mesh3d(
    x=[-1,1,1,-1,-1,1,1,-1],
    y=[-2,-2,2,2,-2,-2,2,2],
    z=[-1,-1,-1,-1,1,1,1,1],
    i=[0,0,0,1,1,2,2,3,4,4,5,6],
    j=[1,2,3,2,5,3,6,0,5,6,6,7],
    k=[2,3,0,5,6,6,7,7,6,7,7,4],
    color=core_color,
    opacity=0.6,
    name="Core"
))

# Winding (outer shell as cube)
fig3d.add_trace(go.Mesh3d(
    x=[-1.8,1.8,1.8,-1.8,-1.8,1.8,1.8,-1.8],
    y=[-2.2,-2.2,2.2,2.2,-2.2,-2.2,2.2,2.2],
    z=[-0.5,-0.5,-0.5,-0.5,0.5,0.5,0.5,0.5],
    i=[0,0,0,1,1,2,2,3,4,4,5,6],
    j=[1,2,3,2,5,3,6,0,5,6,6,7],
    k=[2,3,0,5,6,6,7,7,6,7,7,4],
    color=winding_color,
    opacity=0.4,
    name="Winding"
))

# Bushings (small cubes)
fig3d.add_trace(go.Mesh3d(
    x=[0.8,1.2,1.2,0.8,0.8,1.2,1.2,0.8],
    y=[2.2,2.2,2.6,2.6,2.2,2.2,2.6,2.6],
    z=[0,0,0,0,1,1,1,1],
    i=[0,0,0,1,1,2,2,3,4,4,5,6],
    j=[1,2,3,2,5,3,6,0,5,6,6,7],
    k=[2,3,0,5,6,6,7,7,6,7,7,4],
    color=bushing_color,
    opacity=1,
    name="Bushing"
))

# Animation: rotate camera
frames = []
for angle in np.linspace(0, 360, 60):  # 60 frames per spin
    rad = np.deg2rad(angle)
    eye = dict(x=4*np.cos(rad), y=4*np.sin(rad), z=2)
    frames.append(go.Frame(layout=dict(scene_camera=dict(eye=eye))))

fig3d.update(frames=frames)

# Layout with play/pause buttons
fig3d.update_layout(
    scene=dict(
        xaxis=dict(visible=False, range=[-3,3]),
        yaxis=dict(visible=False, range=[-3,3]),
        zaxis=dict(visible=False, range=[-2,3])
    ),
    margin=dict(l=0,r=0,t=0,b=0),
    showlegend=False,
    updatemenus=[
        dict(type="buttons",
             showactive=False,
             buttons=[
                 dict(label="▶ Play",
                      method="animate",
                      args=[None, dict(frame=dict(duration=100, redraw=True),
                                       fromcurrent=True, loop=True)]),
                 dict(label="⏸ Pause",
                      method="animate",
                      args=[[None], dict(mode="immediate",
                                         frame=dict(duration=0, redraw=False),
                                         transition=dict(duration=0))])
             ])
    ]
)

st.plotly_chart(fig3d, use_container_width=True)
