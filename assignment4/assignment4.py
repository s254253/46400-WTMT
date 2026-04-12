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
print(df.head())
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
    plt.text(0.05, 0.95, f"y = {model.coef_[0]:.3f}x + ({model.intercept_:.3f})\nR² = {model.score(df[[signal_col]], df['Moment_Nm']):.3f}", 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    #if flag_show == True:
     #   plt.show()


# -----------------------------
# PLOTS
# -----------------------------

# Individual
plot_fit(df_Mz, "S1", model_S1, "Mz Calibration (S1 only)")
plot_fit(df_Mz, "S2", model_S2, "Mz Calibration (S2 only)")
plot_fit(df_Mx_S3, "S3", model_S3, "Mx Calibration (S3 only)")
plot_fit(df_Mx_S4, "S4", model_S4, "Mx Calibration (S4 only)")
plot_fit(df_My, "S5", model_My, "My Calibration")


# -----------------------------
# Crosstalk Analysis (Bending → Torsion)
# -----------------------------

# Split datasets
df_Mx_ct = df[df["Load Case"].str.contains("Mx")].copy()
df_My_ct = df[df["Load Case"].str.contains("My")].copy()

# -----------------------------
# REGRESSION + PLOTTING FUNCTION
# -----------------------------

def crosstalk_fit_and_plot(df, signal, title):
    X = df[["Load [kg]"]].values
    y = df[signal].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"{title}")
    print(f"Slope (mV/kg): {slope:.4f}")
    print(f"Offset (mV): {intercept:.4f}\n")

    # Plot
    plt.figure()
    plt.scatter(df["Load [kg]"], y, label="Data")

    x_line = np.linspace(df["Load [kg]"].min(), df["Load [kg]"].max(), 100).reshape(-1,1)
    y_line = model.predict(x_line)

    plt.plot(x_line, y_line, label="Linear fit")
    plt.xlabel("Load (kg)")
    plt.ylabel(f"{signal} (mV)")
    plt.title(title)
    plt.legend()
    plt.grid()

    textstr = f"Slope = {slope:.3f} mV/kg"

    plt.text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round", facecolor="white")
    )

    plt.show()

    return model

# -----------------------------
# RUN ALL 4 CASES
# -----------------------------

print("\n--- Crosstalk: Mx → Torsion Gauges ---\n")

model_Mx_S1 = crosstalk_fit_and_plot(
    df_Mx_ct, "S1", "Crosstalk: Mx loading → S1 response"
)

model_Mx_S2 = crosstalk_fit_and_plot(
    df_Mx_ct, "S2", "Crosstalk: Mx loading → S2 response"
)

print("\n--- Crosstalk: My → Torsion Gauges ---\n")

model_My_S1 = crosstalk_fit_and_plot(
    df_My_ct, "S1", "Crosstalk: My loading → S1 response"
)

model_My_S2 = crosstalk_fit_and_plot(
    df_My_ct, "S2", "Crosstalk: My loading → S2 response"
)