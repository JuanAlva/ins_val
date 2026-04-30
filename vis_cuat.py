import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.transform import Rotation as R


# ============================================================
# 1. Leer archivo SensorConnect con cabecera DATA_START
# ============================================================

def read_quaternion_file(filepath):
    """
    Lee un archivo CSV exportado desde SensorConnect con bloque DATA_START.

    Retorna un DataFrame con columnas:
        Time, q0, q1, q2, q3, t
    donde:
        q0 = w
        q1 = x
        q2 = y
        q3 = z
        t  = tiempo relativo en segundos
    """

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Buscar línea DATA_START
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "DATA_START":
            data_start_idx = i
            break

    if data_start_idx is None:
        raise ValueError("No se encontró DATA_START en el archivo.")

    # La línea siguiente contiene los nombres de columnas
    header_idx = data_start_idx + 1

    df = pd.read_csv(filepath, skiprows=header_idx)

    # Limpiar nombres de columnas
    df.columns = [c.strip() for c in df.columns]

    # Detectar columnas de cuaternión
    quat_cols = [c for c in df.columns if "orientQuaternion" in c]

    if len(quat_cols) != 4:
        raise ValueError(f"Se esperaban 4 columnas de cuaternión, se encontraron {len(quat_cols)}.")

    # Renombrar columnas
    df = df.rename(columns={
        quat_cols[0]: "q0",
        quat_cols[1]: "q1",
        quat_cols[2]: "q2",
        quat_cols[3]: "q3",
    })

    # Convertir a numérico
    for c in ["Time", "q0", "q1", "q2", "q3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Time", "q0", "q1", "q2", "q3"]).reset_index(drop=True)

    # Tiempo relativo en segundos
    # Time está en nanosegundos Unix
    df["t"] = (df["Time"] - df["Time"].iloc[0]) / 1e9

    # Normalizar cuaterniones por seguridad
    q = df[["q0", "q1", "q2", "q3"]].to_numpy(dtype=float)
    q_norm = np.linalg.norm(q, axis=1)

    valid = q_norm > 0
    df = df.loc[valid].reset_index(drop=True)
    q = q[valid]
    q_norm = q_norm[valid]

    q = q / q_norm[:, None]

    df["q0"] = q[:, 0]
    df["q1"] = q[:, 1]
    df["q2"] = q[:, 2]
    df["q3"] = q[:, 3]

    return df


# ============================================================
# 2. Convertir cuaterniones a Euler Roll, Pitch, Yaw
# ============================================================

def add_euler_angles(df, degrees=True):
    """
    Agrega roll, pitch, yaw al DataFrame.

    Convención usada:
        scipy: Rotation.from_quat([x, y, z, w])
        Euler: xyz

    Resultado:
        roll  -> rotación alrededor de X
        pitch -> rotación alrededor de Y
        yaw   -> rotación alrededor de Z
    """

    df = df.copy()

    # Sensor: [w, x, y, z]
    q_wxyz = df[["q0", "q1", "q2", "q3"]].to_numpy(dtype=float)

    # scipy: [x, y, z, w]
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
# 3. Graficar Roll, Pitch, Yaw vs tiempo
# ============================================================

def plot_euler_vs_time(df):
    plt.figure(figsize=(10, 5))

    plt.plot(df["t"], df["roll"], label="Roll X")
    plt.plot(df["t"], df["pitch"], label="Pitch Y")
    plt.plot(df["t"], df["yaw"], label="Yaw Z")

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Ángulo [deg]")
    plt.title("Orientación del IMU: Roll, Pitch, Yaw")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. Visualización 3D interactiva con slider
# ============================================================

def plot_orientation_slider(df, rot):
    """
    Muestra los ejes del IMU en 3D.
    La barra permite moverse en el tiempo.

    Ejes:
        Rojo conceptual   -> X body
        Verde conceptual  -> Y body
        Azul conceptual   -> Z body

    Nota:
        No se fuerzan colores específicos para mantenerlo simple.
    """

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    plt.subplots_adjust(bottom=0.22)

    # Ejes base del cuerpo
    body_axes = np.eye(3)

    # Estado inicial
    idx0 = 0
    R_nb = rot[idx0].as_matrix()

    rotated_axes = R_nb @ body_axes

    origin = np.zeros(3)

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
        f"Orientación IMU | t = {df['t'].iloc[idx0]:.3f} s\n"
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

    # Slider
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

        # Eliminar ejes anteriores
        for qv in quivers:
            qv.remove()

        R_nb = rot[idx].as_matrix()
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

        title.set_text(
            f"Orientación IMU | t = {df['t'].iloc[idx]:.3f} s\n"
            f"Roll={df['roll'].iloc[idx]:.2f}°, "
            f"Pitch={df['pitch'].iloc[idx]:.2f}°, "
            f"Yaw={df['yaw'].iloc[idx]:.2f}°"
        )

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


# ============================================================
# 5. Programa principal
# ============================================================

if __name__ == "__main__":

    filepath = "pruebas_oficina_29-04\cuate_1.csv"  # Cambia esto por el nombre de tu archivo

    df = read_quaternion_file(filepath)

    df, rot = add_euler_angles(df, degrees=True)

    print(df[["t", "q0", "q1", "q2", "q3", "roll", "pitch", "yaw"]].head())

    plot_euler_vs_time(df)

    plot_orientation_slider(df, rot)