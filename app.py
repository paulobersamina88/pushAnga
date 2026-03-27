import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

st.set_page_config(page_title="ML Pushover Research Starter", layout="wide")


def load_model(model_name: str):
    path = MODELS / model_name
    if path.exists():
        return joblib.load(path), True
    return None, False


def make_trilinear_curve(vy, dy, vmax, dmax, vc, dc, points=220):
    xs1 = np.linspace(0, dy, max(2, points // 3))
    ys1 = vy / max(dy, 1e-9) * xs1

    xs2 = np.linspace(dy, dmax, max(2, points // 3))
    m2 = (vmax - vy) / max((dmax - dy), 1e-9)
    ys2 = vy + m2 * (xs2 - dy)

    xs3 = np.linspace(dmax, dc, max(2, points // 3))
    m3 = (vc - vmax) / max((dc - dmax), 1e-9)
    ys3 = vmax + m3 * (xs3 - dmax)

    x = np.concatenate([xs1, xs2[1:], xs3[1:]])
    y = np.concatenate([ys1, ys2[1:], ys3[1:]])
    return x, y


def make_bilinear_curve(k0, dy, alpha, dc=0.12, points=220):
    vy = k0 * dy
    xs1 = np.linspace(0, dy, max(2, points // 2))
    ys1 = k0 * xs1
    xs2 = np.linspace(dy, dc, max(2, points // 2))
    ys2 = vy + alpha * k0 * (xs2 - dy)
    x = np.concatenate([xs1, xs2[1:]])
    y = np.concatenate([ys1, ys2[1:]])
    return x, y


def plot_curve(x, y, title, xlabel="Roof displacement (m)", ylabel="Base shear / lateral force"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig


def demo_rc_frame_predict(features: dict):
    # Demonstration-only heuristic coefficients for UI testing.
    n_storey = features["n_storey"]
    n_bays = features["n_bays"]
    bay_length = features["bay_length_m"]
    storey_height = features["storey_height_m"]
    fc = features["fc_mpa"]
    fy = features["fy_mpa"]
    beam = features["beam_depth_mm"]
    col = features["column_dim_mm"]
    rho_b = features["beam_rho"]
    rho_c = features["column_rho"]
    gk = features["gravity_kn_m2"]

    total_height = n_storey * storey_height
    plan_factor = n_bays * bay_length
    size_factor = 0.55 * beam + 0.95 * col
    strength_factor = 0.7 * fc + 0.12 * fy
    steel_factor = 180 * rho_b + 120 * rho_c

    vy = max(300, 0.7 * total_height * plan_factor + 0.18 * size_factor + 7.0 * strength_factor + steel_factor)
    dy = max(0.008, 0.0022 * total_height + 0.00018 * gk * n_storey)
    vmax = 1.22 * vy
    dmax = 1.9 * dy
    vc = 0.78 * vmax
    dc = 3.9 * dy
    return {
        "Vy": float(vy),
        "dy": float(dy),
        "Vmax": float(vmax),
        "dmax": float(dmax),
        "Vc": float(vc),
        "dc": float(dc),
        "source": "demo",
    }


def demo_rc_wall_predict(features: dict):
    h = features["wall_height_m"]
    l = features["wall_length_m"]
    t = features["wall_thickness_m"]
    fc = features["fc_mpa"]
    fy = features["fy_mpa"]
    rho_v = features["rho_v"]
    rho_h = features["rho_h"]
    axial = features["axial_ratio"]

    k0 = 15000 * t * l / max(h, 0.1) * (1 + 0.8 * axial) * (1 + 15 * (rho_v + rho_h))
    k0 *= (0.6 + fc / 50)
    dy = 0.0025 * h * (1 + 0.15 * axial)
    alpha = 0.03 + 0.02 * min(axial, 0.35) + 2.5 * min(rho_v, 0.01)
    return {"K0": float(k0), "dy": float(dy), "alpha": float(alpha), "source": "demo"}


st.title("ML Pushover Research Starter")
st.caption(
    "Starter Streamlit package inspired by published ML pushover studies. "
    "Included demo predictors are placeholders for workflow testing; replace them with trained models before research or design use."
)

with st.expander("Research basis used in this starter"):
    st.markdown(
        """
- **Angarita et al. (2024)**: RF and ANN models for low-rise RC frame pushover prediction.
- **Pushover-ML (2025)**: GUI predicting a **trilinear** approximation of low-rise RC frame pushover curves.
- **Kuria & Kegyes-Brassai (2025)**: RC shear wall ML framework using **bilinear** features \(K_0\), \(\delta_y\), and \(\alpha\).
- **Samadian et al. (2024)**: surrogate modelling framework and database for steel SMRF seismic/pushover response.

This package gives you a **research-referenced scaffold**: UI, curve plotting, upload hooks, and training template.
"""
    )

mode = st.sidebar.selectbox(
    "Model family",
    [
        "RC frame trilinear (Angarita / Pushover-ML style)",
        "RC shear wall bilinear (Kuria style)",
    ],
)

use_uploaded = st.sidebar.checkbox("Use uploaded trained model if available", value=True)

if mode.startswith("RC frame"):
    st.subheader("Low-rise RC frame trilinear predictor")
    c1, c2, c3 = st.columns(3)
    features = {
        "n_storey": c1.number_input("No. of storeys", 2, 10, 4),
        "n_bays": c2.number_input("No. of bays", 1, 8, 3),
        "bay_length_m": c3.number_input("Bay length (m)", 2.0, 12.0, 5.0, 0.1),
        "storey_height_m": c1.number_input("Storey height (m)", 2.4, 5.0, 3.0, 0.1),
        "fc_mpa": c2.number_input("f'c (MPa)", 14.0, 60.0, 28.0, 0.5),
        "fy_mpa": c3.number_input("fy (MPa)", 275.0, 550.0, 420.0, 5.0),
        "beam_depth_mm": c1.number_input("Beam depth (mm)", 250, 900, 500),
        "column_dim_mm": c2.number_input("Column size (mm)", 250, 1000, 450),
        "beam_rho": c3.number_input("Beam steel ratio", 0.005, 0.04, 0.015, 0.001, format="%.3f"),
        "column_rho": c1.number_input("Column steel ratio", 0.005, 0.06, 0.020, 0.001, format="%.3f"),
        "gravity_kn_m2": c2.number_input("Gravity load (kN/m²)", 1.0, 20.0, 6.0, 0.1),
    }

    model, found = load_model("rc_frame_trilinear.joblib") if use_uploaded else (None, False)
    if found:
        X = pd.DataFrame([features])
        pred = model.predict(X)[0]
        result = {"Vy": pred[0], "dy": pred[1], "Vmax": pred[2], "dmax": pred[3], "Vc": pred[4], "dc": pred[5], "source": "uploaded model"}
    else:
        result = demo_rc_frame_predict(features)

    x, y = make_trilinear_curve(result["Vy"], result["dy"], result["Vmax"], result["dmax"], result["Vc"], result["dc"])
    st.pyplot(plot_curve(x, y, f"RC Frame Capacity Curve ({result['source']})"))

    m1, m2, m3 = st.columns(3)
    m1.metric("Yield base shear, Vy", f"{result['Vy']:.2f}")
    m2.metric("Peak base shear, Vmax", f"{result['Vmax']:.2f}")
    m3.metric("Collapse-point displacement, dc", f"{result['dc']:.4f} m")

    st.dataframe(pd.DataFrame([result]))

else:
    st.subheader("RC shear wall bilinear predictor")
    c1, c2, c3 = st.columns(3)
    features = {
        "wall_height_m": c1.number_input("Wall height (m)", 2.5, 20.0, 9.0, 0.1),
        "wall_length_m": c2.number_input("Wall length (m)", 1.0, 10.0, 4.0, 0.1),
        "wall_thickness_m": c3.number_input("Wall thickness (m)", 0.10, 0.60, 0.20, 0.01),
        "fc_mpa": c1.number_input("f'c (MPa)", 14.0, 70.0, 30.0, 0.5),
        "fy_mpa": c2.number_input("fy (MPa)", 275.0, 600.0, 420.0, 5.0),
        "rho_v": c3.number_input("Vertical steel ratio", 0.001, 0.04, 0.006, 0.001, format="%.3f"),
        "rho_h": c1.number_input("Horizontal steel ratio", 0.001, 0.04, 0.004, 0.001, format="%.3f"),
        "axial_ratio": c2.number_input("Axial load ratio", 0.00, 0.60, 0.10, 0.01, format="%.2f"),
    }

    model, found = load_model("rc_wall_bilinear.joblib") if use_uploaded else (None, False)
    if found:
        X = pd.DataFrame([features])
        pred = model.predict(X)[0]
        result = {"K0": pred[0], "dy": pred[1], "alpha": pred[2], "source": "uploaded model"}
    else:
        result = demo_rc_wall_predict(features)

    x, y = make_bilinear_curve(result["K0"], result["dy"], result["alpha"])
    st.pyplot(plot_curve(x, y, f"RC Shear Wall Capacity Curve ({result['source']})", ylabel="Lateral resistance"))

    m1, m2, m3 = st.columns(3)
    m1.metric("Initial stiffness, K0", f"{result['K0']:.2f}")
    m2.metric("Yield displacement, dy", f"{result['dy']:.4f} m")
    m3.metric("Post-yield ratio, alpha", f"{result['alpha']:.4f}")

    st.dataframe(pd.DataFrame([result]))

st.markdown("---")
st.subheader("Upload your trained model later")
st.markdown(
    "Expected outputs: **RC frame** -> `[Vy, dy, Vmax, dmax, Vc, dc]`; "
    "**RC wall** -> `[K0, dy, alpha]`. Use the included training script as your starting point."
)

st.code(
    json.dumps(
        {
            "rc_frame_input_columns": [
                "n_storey",
                "n_bays",
                "bay_length_m",
                "storey_height_m",
                "fc_mpa",
                "fy_mpa",
                "beam_depth_mm",
                "column_dim_mm",
                "beam_rho",
                "column_rho",
                "gravity_kn_m2",
            ],
            "rc_frame_output_columns": ["Vy", "dy", "Vmax", "dmax", "Vc", "dc"],
            "rc_wall_input_columns": [
                "wall_height_m",
                "wall_length_m",
                "wall_thickness_m",
                "fc_mpa",
                "fy_mpa",
                "rho_v",
                "rho_h",
                "axial_ratio",
            ],
            "rc_wall_output_columns": ["K0", "dy", "alpha"],
        },
        indent=2,
    ),
    language="json",
)
