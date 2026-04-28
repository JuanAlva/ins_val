import numpy as np

def estimate_mounting_angles(df):
    """
    df debe contener:
    t, q0, q1, q2, q3, x_ref, y_ref, z_ref, ds
    
    x_ref, y_ref, z_ref deben estar en un frame local tipo NED:
    x_ref -> Norte
    y_ref -> Este
    z_ref -> Down
    """

    n = len(df)

    # Estado EKF
    x = np.zeros(9)

    # Estimación acumulada de montaje
    pitch_mount = 0.0
    heading_mount = 0.0

    # Covarianza inicial
    P = np.diag([
        1.0**2, 1.0**2, 1.0**2,              # posición DR: m²
        np.deg2rad(1.0)**2,                  # pitch montaje
        np.deg2rad(1.0)**2,                  # heading montaje
        np.deg2rad(1.0)**2,                  # actitud roll
        np.deg2rad(1.0)**2,                  # actitud pitch
        np.deg2rad(1.0)**2,                  # actitud heading
        (1000e-6)**2                         # escala distancia
    ])

    # Ruido de proceso
    ARW_deg_sqrt_h = 0.14
    ARW_rad_sqrt_s = np.deg2rad(ARW_deg_sqrt_h) / np.sqrt(3600)
    q_att = ARW_rad_sqrt_s**2

    q_mount = q_att / 100
    q_sf = (10e-6)**2

    # Medición: ruido posición de referencia
    R_meas = np.diag([0.20**2, 0.20**2, 0.50**2])  # ajustar

    # H: mide directamente error de posición
    H = np.zeros((3, 9))
    H[:, 0:3] = np.eye(3)

    # Posición DR inicial igual a referencia inicial
    r_dr = np.array([
        df["x_ref"].iloc[0],
        df["y_ref"].iloc[0],
        df["z_ref"].iloc[0],
    ], dtype=float)

    pitch_hist = []
    heading_hist = []
    r_dr_hist = []

    for k in range(1, n):
        dt = df["t"].iloc[k] - df["t"].iloc[k - 1]
        ds = df["ds"].iloc[k]

        q0, q1, q2, q3 = df.loc[df.index[k], ["q0", "q1", "q2", "q3"]]

        C_n_b = quat_to_C_n_b(q0, q1, q2, q3)

        C_v_b = C_v_b_from_mount(
            pitch=pitch_mount,
            heading=heading_mount,
            roll=0.0
        )

        C_b_v = C_v_b.T

        ds_v = np.array([ds, 0.0, 0.0])

        # Dead reckoning
        ds_n = C_n_b @ C_b_v @ ds_v
        r_dr = r_dr + ds_n

        # Matriz M del paper
        M = np.array([
            [0.0, 0.0],
            [0.0, -ds],
            [ds, 0.0]
        ])

        # Phi del EKF
        Phi = np.eye(9)

        # Bloque posición respecto a errores de montaje
        Phi[0:3, 3:5] = -C_n_b @ C_b_v @ M

        # Bloque posición respecto a errores de actitud
        Phi[0:3, 5:8] = skew(ds_n)

        # Bloque posición respecto a error de escala
        Phi[0:3, 8] = ds_n

        # Q discreta
        Q6 = np.diag([
            q_mount,
            q_mount,
            q_att,
            q_att,
            q_att,
            q_sf
        ]) * max(dt, 1e-3)

        G = np.zeros((9, 6))
        G[3:9, :] = np.eye(6)

        Q = G @ Q6 @ G.T

        # Predicción
        x = Phi @ x
        P = Phi @ P @ Phi.T + Q

        # Medición: error entre DR y referencia
        r_ref = np.array([
            df["x_ref"].iloc[k],
            df["y_ref"].iloc[k],
            df["z_ref"].iloc[k],
        ])

        z = r_dr - r_ref

        # Update EKF
        y = z - H @ x
        S = H @ P @ H.T + R_meas
        K = P @ H.T @ np.linalg.inv(S)

        dx = K @ y
        x = x + dx
        P = (np.eye(9) - K @ H) @ P

        # Feedback
        # Se corrige la posición DR y se acumulan ángulos estimados.
        r_dr = r_dr - x[0:3]

        pitch_mount = pitch_mount + x[3]
        heading_mount = heading_mount + x[4]

        # Reset de errores corregidos
        x[0:5] = 0.0

        pitch_hist.append(pitch_mount)
        heading_hist.append(heading_mount)
        r_dr_hist.append(r_dr.copy())

    return {
        "pitch_mount_rad": pitch_mount,
        "heading_mount_rad": heading_mount,
        "pitch_hist_rad": np.array(pitch_hist),
        "heading_hist_rad": np.array(heading_hist),
        "r_dr_hist": np.array(r_dr_hist),
    }