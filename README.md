# ML Pushover Research Starter Package

This is a **Streamlit starter package** for a research-backed pushover predictor workflow.

Important: this package is **not** the original published software nor the original paper-trained weights. It is a **clean starter scaffold** so you can build your own app using the published study structures, inputs, outputs, and workflow.

## What is included

- `app.py` — Streamlit UI for two study-inspired workflows:
  - **RC frame trilinear predictor** inspired by Angarita / Pushover-ML
  - **RC shear wall bilinear predictor** inspired by Kuria et al.
- `train_models.py` — training script for your own CSV dataset
- `make_demo_data.py` — makes synthetic demo CSV files so the app can be tested immediately
- `models/` — target folder for trained `.joblib` models
- `data/` — demo CSVs will be placed here

## Research references used for this scaffold

1. **Angarita, Montes, Arroyo (2024)** — ML prediction of pushover curves of low-rise RC frame buildings.
   - Key idea used here: **predicting low-rise RC frame pushover response with RF/ANN from easily obtainable design parameters**.
2. **Angarita, Montes, Arroyo (2025), SoftwareX — Pushover-ML**
   - Key idea used here: **predicting a trilinear approximation of the pushover curve with yielding, maximum-capacity, and collapse points**.
3. **Kuria & Kegyes-Brassai (2025), Scientific Reports**
   - Key idea used here: **predicting bilinear wall features**: initial stiffness `K0`, yield displacement `dy`, and post-yield stiffness ratio `alpha`.
4. **Samadian et al. (2024)**
   - Key idea used here: surrogate modelling direction for **steel SMRF** pushover/seismic response. This package does not yet include a steel tab, but the training pattern is ready for extension.

## Quick start

```bash
pip install -r requirements.txt
python make_demo_data.py
streamlit run app.py
```

## Train your own model

### RC frame

Expected input columns:

- `n_storey`
- `n_bays`
- `bay_length_m`
- `storey_height_m`
- `fc_mpa`
- `fy_mpa`
- `beam_depth_mm`
- `column_dim_mm`
- `beam_rho`
- `column_rho`
- `gravity_kn_m2`

Expected output columns:

- `Vy`
- `dy`
- `Vmax`
- `dmax`
- `Vc`
- `dc`

Train:

```bash
python train_models.py --csv data/demo_rc_frame.csv --mode rc_frame --family rf
```

### RC wall

Expected input columns:

- `wall_height_m`
- `wall_length_m`
- `wall_thickness_m`
- `fc_mpa`
- `fy_mpa`
- `rho_v`
- `rho_h`
- `axial_ratio`

Expected output columns:

- `K0`
- `dy`
- `alpha`

Train:

```bash
python train_models.py --csv data/demo_rc_wall.csv --mode rc_wall --family rf
```

## Recommended next upgrade for your research app

1. Replace synthetic demo data with **OpenSeesPy-generated pushover results**.
2. Match the **input variables** to the paper you are following.
3. Validate with a holdout dataset and report **MAE / R² / parity plots**.
4. Add a **steel SMRF tab** once your steel surrogate dataset is ready.
5. Add ADRS / performance-point plotting after your curve prediction is validated.

## Honesty note

The included demo predictors in `app.py` are **engineering-style placeholders for UI testing only**. They are there so the package runs immediately. They are **not paper coefficients** and should **not** be used for design, publication, or code compliance checks.
