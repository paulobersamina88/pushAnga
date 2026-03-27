from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# Demo RC frame dataset
rows = []
for _ in range(500):
    n_storey = int(rng.integers(2, 6))
    n_bays = int(rng.integers(2, 5))
    bay = float(rng.uniform(4.0, 7.0))
    sh = float(rng.uniform(2.8, 3.5))
    fc = float(rng.uniform(21, 40))
    fy = float(rng.uniform(400, 500))
    bd = float(rng.uniform(350, 700))
    cd = float(rng.uniform(350, 800))
    rb = float(rng.uniform(0.010, 0.025))
    rc = float(rng.uniform(0.012, 0.035))
    gk = float(rng.uniform(4, 9))

    total_h = n_storey * sh
    vy = 0.7 * total_h * n_bays * bay + 0.18 * (0.55 * bd + 0.95 * cd) + 7.0 * (0.7 * fc + 0.12 * fy) + 180 * rb + 120 * rc
    dy = 0.0022 * total_h + 0.00018 * gk * n_storey
    vmax = 1.22 * vy
    dmax = 1.9 * dy
    vc = 0.78 * vmax
    dc = 3.9 * dy

    vy *= rng.normal(1.0, 0.05)
    dy *= rng.normal(1.0, 0.04)
    vmax *= rng.normal(1.0, 0.05)
    dmax *= rng.normal(1.0, 0.05)
    vc *= rng.normal(1.0, 0.05)
    dc *= rng.normal(1.0, 0.05)

    rows.append([n_storey, n_bays, bay, sh, fc, fy, bd, cd, rb, rc, gk, vy, dy, vmax, dmax, vc, dc])

frame_cols = [
    "n_storey","n_bays","bay_length_m","storey_height_m","fc_mpa","fy_mpa","beam_depth_mm",
    "column_dim_mm","beam_rho","column_rho","gravity_kn_m2","Vy","dy","Vmax","dmax","Vc","dc"
]
pd.DataFrame(rows, columns=frame_cols).to_csv(DATA / "demo_rc_frame.csv", index=False)

# Demo RC wall dataset
rows = []
for _ in range(350):
    h = float(rng.uniform(3.0, 18.0))
    l = float(rng.uniform(2.0, 8.0))
    t = float(rng.uniform(0.15, 0.35))
    fc = float(rng.uniform(21, 45))
    fy = float(rng.uniform(400, 500))
    rv = float(rng.uniform(0.002, 0.012))
    rh = float(rng.uniform(0.002, 0.010))
    axial = float(rng.uniform(0.02, 0.30))

    k0 = 15000 * t * l / h * (1 + 0.8 * axial) * (1 + 15 * (rv + rh)) * (0.6 + fc / 50)
    dy = 0.0025 * h * (1 + 0.15 * axial)
    alpha = 0.03 + 0.02 * axial + 2.5 * rv

    k0 *= rng.normal(1.0, 0.05)
    dy *= rng.normal(1.0, 0.04)
    alpha *= rng.normal(1.0, 0.06)

    rows.append([h, l, t, fc, fy, rv, rh, axial, k0, dy, alpha])

wall_cols = ["wall_height_m","wall_length_m","wall_thickness_m","fc_mpa","fy_mpa","rho_v","rho_h","axial_ratio","K0","dy","alpha"]
pd.DataFrame(rows, columns=wall_cols).to_csv(DATA / "demo_rc_wall.csv", index=False)
print("Demo CSV files created in ./data")
