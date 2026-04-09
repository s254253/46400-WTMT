import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# READ CSV
# -----------------------------

df = pd.read_csv("assignment4/AS04_Strain_lab_data.csv", sep=";")
df.columns = df.columns.str.strip()
# -----------------------------
# CONSTANTS
# -----------------------------

g = 9.80665

# Lever arms
C = 1.032
D = 1.079
E = 1.000

# -----------------------------
# COMPUTE FORCE
# -----------------------------

df["Force_N"] = df["Load [kg]"] * g

# -----------------------------
# SPLIT DATASETS
# -----------------------------

df_Mx = df[df["Load Case"].str.contains("Mx")].copy()
df_My = df[df["Load Case"].str.contains("My")].copy()
df_Mz = df[df["Load Case"].str.contains("Mz")].copy()

# -----------------------------
# MOMENTS (INDIVIDUAL GAUGES)
# -----------------------------

# Mx → S3 uses C, S4 uses D
df_Mx_S3 = df_Mx.copy()
df_Mx_S3["Moment_Nm"] = df_Mx_S3["Force_N"] * C

df_Mx_S4 = df_Mx.copy()
df_Mx_S4["Moment_Nm"] = df_Mx_S4["Force_N"] * D

# My → S5 uses D
df_My["Moment_Nm"] = df_My["Force_N"] * D

# Mz → BOTH S1 and S2 use same torsion formula
df_Mz["Moment_Nm"] = 2 * df_Mz["Force_N"] * E

# -----------------------------
# REGRESSION FUNCTION
# -----------------------------

def fit_regression(df, signal_col, label):
    X = df[[signal_col]].values
    y = df["Moment_Nm"].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    print(f"\n--- {label} ---")
    print(f"Gain (Nm/mV): {slope:.4f}")
    print(f"Offset (Nm): {intercept:.4f}")
    print(f"R^2: {r2:.5f}")

    return model

# Torsion
model_S1 = fit_regression(df_Mz, "S1", "Mz using S1 only")
model_S2 = fit_regression(df_Mz, "S2", "Mz using S2 only")

# Bending Mx
model_S3 = fit_regression(df_Mx_S3, "S3", "Mx using S3 only (lever=C)")
model_S4 = fit_regression(df_Mx_S4, "S4", "Mx using S4 only (lever=D)")

# Bending My
model_My = fit_regression(df_My, "S5", "My using S5")

# -----------------------------
# PLOTTING FUNCTION
# -----------------------------

def plot_fit(df, signal_col, model, title):
    plt.figure()
    plt.scatter(df[signal_col], df["Moment_Nm"])

    x = np.linspace(df[signal_col].min(), df[signal_col].max(), 100).reshape(-1,1)
    y = model.predict(x)

    plt.plot(x, y)
    plt.xlabel("Signal (mV)")
    plt.ylabel("Moment (Nm)")
    plt.title(title)
    plt.grid()
    plt.show()

# -----------------------------
# PLOTS
# -----------------------------

# Individual
plot_fit(df_Mz, "S1", model_S1, "Mz Calibration (S1 only)")
plot_fit(df_Mz, "S2", model_S2, "Mz Calibration (S2 only)")
plot_fit(df_Mx_S3, "S3", model_S3, "Mx Calibration (S3 only)")
plot_fit(df_Mx_S4, "S4", model_S4, "Mx Calibration (S4 only)")
plot_fit(df_My, "S5", model_My, "My Calibration")