from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

RC_FRAME_IN = [
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
]
RC_FRAME_OUT = ["Vy", "dy", "Vmax", "dmax", "Vc", "dc"]

RC_WALL_IN = [
    "wall_height_m",
    "wall_length_m",
    "wall_thickness_m",
    "fc_mpa",
    "fy_mpa",
    "rho_v",
    "rho_h",
    "axial_ratio",
]
RC_WALL_OUT = ["K0", "dy", "alpha"]


def fit_and_save(df: pd.DataFrame, x_cols: list[str], y_cols: list[str], out_path: Path, family: str):
    X = df[x_cols]
    y = df[y_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if family == "rf":
        model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    else:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1200, random_state=42)),
            ]
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae_mean": float(mean_absolute_error(y_test, pred)),
        "r2_mean": float(r2_score(y_test, pred, multioutput="uniform_average")),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    joblib.dump(model, out_path)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Training CSV path")
    parser.add_argument("--mode", choices=["rc_frame", "rc_wall"], required=True)
    parser.add_argument("--family", choices=["rf", "mlp"], default="rf")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.mode == "rc_frame":
        metrics = fit_and_save(df, RC_FRAME_IN, RC_FRAME_OUT, MODELS / "rc_frame_trilinear.joblib", args.family)
    else:
        metrics = fit_and_save(df, RC_WALL_IN, RC_WALL_OUT, MODELS / "rc_wall_bilinear.joblib", args.family)

    print("Saved model.")
    print(metrics)


if __name__ == "__main__":
    main()
