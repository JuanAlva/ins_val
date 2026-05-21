import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation as R


# ============================================================
# 1. Leer archivo de cuaterniones en formato CSV directo
# ============================================================

def read_est_quaternion_file(filepath, only_valid=True):
    """
    Lee un archivo CSV que ya tiene cabecera directa.

    Formato esperado:
        Time, reference_time, gps_timestamp,
        estOrientQuaternion[0-0],
        estOrientQuaternion[0-1],
        estOrientQuaternion[0-2],
        estOrientQuaternion[0-3],
        estOrientQuaternion:valid,
        flagStop

    Retorna un DataFrame con:
        Time
        reference_time
        gps_timestamp
        q0, q1, q2, q3
        qValid
        flagStop
        t
    """

    df = pd.read_csv(filepath)

    # Limpiar nombres de columnas
    df.columns = [c.strip() for c in df.columns]

    quat_cols = [
        "estOrientQuaternion[0]",
        "estOrientQuaternion[1]",
        "estOrientQuaternion[2]",
        "estOrientQuaternion[3]",
        # "estOrientQuaternion[0-0]",
        # "estOrientQuaternion[0-1]",
        # "estOrientQuaternion[0-2]",
        # "estOrientQuaternion[0-3]",
    ]

    required_cols = ["Time"] + quat_cols

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    rename_map = {
        "estOrientQuaternion[0]": "q0",
        "estOrientQuaternion[1]": "q1",
        "estOrientQuaternion[2]": "q2",
        "estOrientQuaternion[3]": "q3",
    }

    if "estOrientQuaternion:valid" in df.columns:
        rename_map["estOrientQuaternion:valid"] = "qValid"

    df = df.rename(columns=rename_map)

    # Convertir columnas principales a numérico
    numeric_cols = ["Time", "q0", "q1", "q2", "q3"]

    optional_cols = ["reference_time", "gps_timestamp", "qValid", "flagStop"]
    for c in optional_cols:
        if c in df.columns:
            numeric_cols.append(c)

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Eliminar filas inválidas numéricamente
    df = df.dropna(subset=["Time", "q0", "q1", "q2", "q3"]).reset_index(drop=True)

    # Filtrar solo cuaterniones válidos
    if only_valid and "qValid" in df.columns:
        df = df[df["qValid"] == 1].reset_index(drop=True)

    # Eliminar cuaterniones nulos
    q = df[["q0", "q1", "q2", "q3"]].to_numpy(dtype=float)
    q_norm = np.linalg.norm(q, axis=1)

    valid_norm = q_norm > 0
    df = df.loc[valid_norm].reset_index(drop=True)

    q = q[valid_norm]
    q_norm = q_norm[valid_norm]

    # Normalizar cuaterniones por seguridad
    q = q / q_norm[:, None]

    df["q0"] = q[:, 0]
    df["q1"] = q[:, 1]
    df["q2"] = q[:, 2]
    df["q3"] = q[:, 3]

    # Tiempo relativo en segundos usando Time en nanosegundos Unix
    df["t"] = (df["Time"] - df["Time"].iloc[0]) / 1000.0 #1e9

    # Delta tiempo, útil para revisar frecuencia real
    df["dt"] = df["t"].diff()

    return df


# ============================================================
# 2. Convertir cuaterniones a Roll, Pitch, Yaw
# ============================================================

def add_euler_angles(df, degrees=True):
    """
    Agrega roll, pitch, yaw al DataFrame.

    Convención:
        Entrada IMU:
            q = [q0, q1, q2, q3] = [w, x, y, z]

        scipy:
            q = [x, y, z, w]

        Euler:
            'xyz'
            roll  = rotación sobre X
            pitch = rotación sobre Y
            yaw   = rotación sobre Z
    """

    df = df.copy()

    q_wxyz = df[["q0", "q1", "q2", "q3"]].to_numpy(dtype=float)

    q_xyzw = np.column_stack([
        q_wxyz[:, 1],
        q_wxyz[:, 2],
        q_wxyz[:, 3],
        q_wxyz[:, 0],
    ])

    rot = R.from_quat(q_xyzw)

    euler = rot.as_euler("xyz", degrees=degrees)

    df["roll"] = euler[:, 0]
    df["pitch"] = euler[:, 1]
    df["yaw"] = euler[:, 2]

    return df, rot


# ============================================================
# 3. Gráfico Roll, Pitch, Yaw vs tiempo
# ============================================================

def plot_euler_vs_time(df, show_stop_flags=True):
    plt.figure(figsize=(11, 5))

    plt.plot(df["t"], df["roll"], label="Roll X")
    plt.plot(df["t"], df["pitch"], label="Pitch Y")
    plt.plot(df["t"], df["yaw"], label="Yaw Z")

    if show_stop_flags and "flagStop" in df.columns:
        stop_df = df[df["flagStop"] == 1]

        if len(stop_df) > 0:
            plt.scatter(
                stop_df["t"],
                stop_df["yaw"],
                s=12,
                label="flagStop = 1"
            )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Ángulo [deg]")
    plt.title("Orientación del IMU: Roll, Pitch, Yaw")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. Gráfico de dt para revisar frecuencia real
# ============================================================

def plot_dt(df):
    plt.figure(figsize=(10, 4))

    plt.plot(df["t"], df["dt"] * 1000)

    plt.xlabel("Tiempo [s]")
    plt.ylabel("dt [ms]")
    plt.title("Delta tiempo entre muestras")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Resumen dt:")
    print(df["dt"].describe())

    dt_mean = df["dt"].mean()

    if pd.notna(dt_mean) and dt_mean > 0:
        fs_mean = 1.0 / dt_mean
        print(f"\nFrecuencia media aproximada: {fs_mean:.2f} Hz")


# ============================================================
# 5. Visualización 3D interactiva con slider
# ============================================================

def plot_orientation_slider(df, rot):
    """
    Muestra los ejes del IMU en 3D.
    La barra permite moverse muestra por muestra.
    """

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    plt.subplots_adjust(bottom=0.22)

    body_axes = np.eye(3)
    origin = np.zeros(3)

    idx0 = 0
    R_nb = rot[idx0].as_matrix().T
    rotated_axes = R_nb @ body_axes

    quivers = []

    for i in range(3):
        qv = ax.quiver(
            origin[0], origin[1], origin[2],
            rotated_axes[0, i],
            rotated_axes[1, i],
            rotated_axes[2, i],
            length=1.0,
            normalize=True
        )
        quivers.append(qv)

    title = ax.set_title(
        f"Orientación IMU | muestra = {idx0} | t = {df['t'].iloc[idx0]:.3f} s\n"
        f"Roll={df['roll'].iloc[idx0]:.2f}°, "
        f"Pitch={df['pitch'].iloc[idx0]:.2f}°, "
        f"Yaw={df['yaw'].iloc[idx0]:.2f}°"
    )

    ax.set_xlabel("X nav")
    ax.set_ylabel("Y nav")
    ax.set_zlabel("Z nav")

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])

    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

    ax_slider = plt.axes([0.18, 0.08, 0.65, 0.04])

    slider = Slider(
        ax=ax_slider,
        label="Muestra",
        valmin=0,
        valmax=len(df) - 1,
        valinit=0,
        valstep=1
    )

    def update(val):
        nonlocal quivers

        idx = int(slider.val)

        for qv in quivers:
            qv.remove()

        R_nb = rot[idx].as_matrix().T
        rotated_axes = R_nb @ body_axes

        quivers = []

        for i in range(3):
            qv = ax.quiver(
                origin[0], origin[1], origin[2],
                rotated_axes[0, i],
                rotated_axes[1, i],
                rotated_axes[2, i],
                length=1.0,
                normalize=True
            )
            quivers.append(qv)

        extra_info = ""

        if "flagStop" in df.columns:
            extra_info += f" | flagStop={int(df['flagStop'].iloc[idx])}"

        if "qValid" in df.columns:
            extra_info += f" | qValid={int(df['qValid'].iloc[idx])}"

        title.set_text(
            f"Orientación IMU | muestra = {idx} | t = {df['t'].iloc[idx]:.3f} s{extra_info}\n"
            f"Roll={df['roll'].iloc[idx]:.2f}°, "
            f"Pitch={df['pitch'].iloc[idx]:.2f}°, "
            f"Yaw={df['yaw'].iloc[idx]:.2f}°"
        )

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


# ============================================================
# 6. Programa principal
# ============================================================

if __name__ == "__main__":

    filepath = r"08_05_2026\08_05_2026\06\QuaternionEst__1.csv"  # Cambia por tu archivo real

    df = read_est_quaternion_file(filepath, only_valid=True)

    df, rot = add_euler_angles(df, degrees=True)

    print("\nPrimeras filas:")
    print(df[[
        "t",
        "dt",
        "q0",
        "q1",
        "q2",
        "q3",
        "roll",
        "pitch",
        "yaw"
    ]].head())

    print("\nÚltimas filas:")
    print(df[[
        "t",
        "dt",
        "roll",
        "pitch",
        "yaw"
    ]].tail())

    plot_euler_vs_time(df, show_stop_flags=True)

    plot_dt(df)

    plot_orientation_slider(df, rot)