
# -------------------------------------------------
# For plotting
# -------------------------------------------------
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import pyvista as pv

from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "figure.titlesize": 16,
})



def load_qois(qois_filename):
    qois_vals = np.loadtxt(qois_filename)
    with open(qois_filename, "r") as f:
        qois_names = f.readline().split()[1:]
    return qois_vals, qois_names

def get(qois_vals, qois_names, key):
    return qois_vals[:, qois_names.index(key)]

def plot_linear_homogenization_hollowbox(
    res_folder="results_linear_homogenization",
    shape_list=("round", "hex"),
    r0_list=None,
    res_basename_prefix="linear",
    Km=1.0,
    save_name="plots/linear_homogenization_K_vs_phi.png",
    show_plot=False,
    eps=1e-12,
):

    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

    if r0_list is None:
        r0_list = [0.02, 0.05, 0.072, 0.1, 0.2, 0.315, 0.41, 0.5]

    def fmt_file(x):
        return f"{x:.6g}"

    def shape_label(shape):
        if shape == "round":
            return "Microscopic model - circular inclusions"
        if shape == "hex":
            return "Microscopic model - hexagonal inclusions"
        return f"Microscopic model - {shape} inclusions"

    def build_basename(shape, r0, probe):
        return f"{res_folder}/{res_basename_prefix}-{shape}-r0={fmt_file(r0)}-{probe}"

    def build_qois_filename(shape, r0, probe):
        return build_basename(shape, r0, probe) + "-qois.dat"

    def build_metadata_filename(shape, r0, probe):
        return build_basename(shape, r0, probe) + "-metadata.json"

    def read_phi_from_metadata(shape, r0, probe="gx"):
        filename = build_metadata_filename(shape, r0, probe)

        if os.path.exists(filename):
            with open(filename, "r") as f:
                metadata = json.load(f)

            for key in ["mesh_porosity", "porosity", "phi"]:
                if key in metadata:
                    return float(metadata[key])

        return None

    def read_probe(filename):
        qois_vals, names = load_qois(filename)

        data = {
            "Qx": np.asarray(get(qois_vals, names, "Q_l_avg_x"), dtype=float),
            "Qy": np.asarray(get(qois_vals, names, "Q_l_avg_y"), dtype=float),
            "gx": np.asarray(get(qois_vals, names, "grad_p_bar_avg_x"), dtype=float),
            "gy": np.asarray(get(qois_vals, names, "grad_p_bar_avg_y"), dtype=float),
        }

        npts = len(data["Qx"])

        for key, val in data.items():
            if len(val) != npts:
                raise ValueError(f"Inconsistent length for {key} in {filename}")

        return data

    def principal_values(K):
        Ksym = 0.5 * (K + K.T)

        a = Ksym[0, 0]
        b = Ksym[0, 1]
        c = Ksym[1, 1]

        tr = a + c
        delta = np.sqrt((a - c) ** 2 + 4.0 * b ** 2)

        K1 = 0.5 * (tr + delta)
        K2 = 0.5 * (tr - delta)

        return K1, K2

    rows = []

    for shape in shape_list:
        for r0 in r0_list:
            file_gx = build_qois_filename(shape, r0, "gx")
            file_gy = build_qois_filename(shape, r0, "gy")

            if not os.path.exists(file_gx):
                print(f"[WARNING] Missing file: {file_gx}")
                continue

            if not os.path.exists(file_gy):
                print(f"[WARNING] Missing file: {file_gy}")
                continue

            data_gx = read_probe(file_gx)
            data_gy = read_probe(file_gy)

            qx_gx = data_gx["Qx"][-1]
            qy_gx = data_gx["Qy"][-1]
            gx = data_gx["gx"][-1]

            qx_gy = data_gy["Qx"][-1]
            qy_gy = data_gy["Qy"][-1]
            gy = data_gy["gy"][-1]

            if abs(gx) < eps or abs(gy) < eps:
                print(f"[WARNING] Gradient too small for shape={shape}, r0={r0}")
                continue

            Kxx = -qx_gx / (gx + eps)
            Kyx = -qy_gx / (gx + eps)
            Kxy = -qx_gy / (gy + eps)
            Kyy = -qy_gy / (gy + eps)

            K = np.array(
                [
                    [Kxx, Kxy],
                    [Kyx, Kyy],
                ],
                dtype=float,
            )

            K1, K2 = principal_values(K)
            Keq = 0.5 * (K1 + K2)

            phi = read_phi_from_metadata(shape, r0, "gx")

            if phi is None:
                print(f"[WARNING] phi not found for shape={shape}, r0={r0}")
                continue

            rows.append(
                {
                    "shape": shape,
                    "r0": r0,
                    "phi": phi,
                    "Kxx": Kxx,
                    "Kxy": Kxy,
                    "Kyx": Kyx,
                    "Kyy": Kyy,
                    "K1": K1,
                    "K2": K2,
                    "Keq": Keq,
                }
            )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No valid permeability results found.")

    df = df.sort_values(["shape", "phi"]).reset_index(drop=True)

    csv_name = save_name.replace(".png", ".csv")
    df.to_csv(csv_name, index=False)
    print("Saved:", csv_name)

    phi_min = float(df["phi"].min())
    phi_max = float(df["phi"].max())
    phi_grid = np.linspace(phi_min, phi_max, 300)

    K_dilute = Km * (1.0 - 2.0 * phi_grid)
    K_diff = Km * (1.0 - phi_grid) ** 2
    mask_dilute = K_dilute > 0.0

    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    markers = {
        "round": "o",
        "hex": "s",
    }

    linestyles = {
        "round": "-",
        "hex": "--",
    }

    for shape in shape_list:
        sub = df[df["shape"] == shape]

        if len(sub) == 0:
            continue

        ax.plot(
            sub["phi"],
            sub["Keq"] / Km,
            marker=markers.get(shape, "o"),
            linestyle=linestyles.get(shape, "-"),
            linewidth=2.0,
            markerfacecolor="white",
            label=shape_label(shape),
        )

    ax.plot(
        phi_grid[mask_dilute],
        K_dilute[mask_dilute] / Km,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label="Dilute scheme",
    )

    ax.plot(
        phi_grid,
        K_diff / Km,
        color="black",
        linestyle="-.",
        linewidth=2.0,
        label="Differential scheme",
    )

    ax.set_xlabel(r"$\tilde{\Phi}_{g0}$", fontsize=14)
    ax.set_ylabel(r"$K_i/K_m$", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(False)
    ax.legend(fontsize=9.5, frameon=True)

    plt.tight_layout()
    plt.savefig(save_name, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()

    plt.close()

    print("Saved:", save_name)


def plot_principal_K_vs_U(
    res_folder,
    res_basename_prefix,
    r0_list,
    pf_list=(0.0,),
    x_component="xx",
    phi=None,
    slice_start=5,
    eps=1e-12,
    add_prediction=True,
    save_name="plots/principal_K_vs_U.png",
    show_plot=False,
    xdmf_folder=None,
    xdmf_basename_prefix=None,
    stream_pf=None,
    stream_probe="gx",
    stream_density=1.0,
    stream_scale=1.0,
    stream_grid_n=400,
    add_stream_colorbar=True,
    theta_in_degrees=True,
    theta_lift_threshold_deg=-85.0,
):
    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

    x_components = {
        "xx": "Uxx",
        "yy": "Uyy",
        "xy": "Uxy",
        "yx": "Uyx",
    }

    x_labels = {
        "xx": r"$E_x$",
        "yy": r"$E_y$",
        "xy": r"$E_{xy}$",
        "yx": r"$E_{yx}$",
    }

    if x_component not in x_components:
        raise ValueError(f"x_component must be one of {list(x_components.keys())}")

    if xdmf_folder is None:
        xdmf_folder = res_folder

    if xdmf_basename_prefix is None:
        xdmf_basename_prefix = res_basename_prefix

    if stream_pf is None:
        stream_pf = pf_list[0]

    def build_basename(folder, prefix, r0, pf, probe):
        filename = f"{folder}/{prefix}-r0={r0}-pf={pf}-{probe}"

        if phi is not None:
            phi_str = f"{phi:.3f}".replace(".", "p") if isinstance(phi, float) else str(phi)
            filename += f"-phi={phi_str}"

        return filename

    def build_filename(r0, pf, probe):
        return build_basename(
            res_folder,
            res_basename_prefix,
            r0,
            pf,
            probe,
        ) + "-qois.dat"

    def build_xdmf_filename(r0, pf, probe):
        return build_basename(
            xdmf_folder,
            xdmf_basename_prefix,
            r0,
            pf,
            probe,
        ) + ".xdmf"

    def read_phi_from_metadata(r0, pf, probe="gx"):
        basename = build_basename(
            res_folder,
            res_basename_prefix,
            r0,
            pf,
            probe,
        )

        metadata_file = basename + "-metadata.json"

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for key in ["mesh_porosity", "porosity", "phi"]:
                if key in metadata:
                    return float(metadata[key])

        return None

    def format_phi_value(value):
        if value is None or not np.isfinite(value):
            return "?"

        if abs(value) < 1e-12:
            return "0"

        return f"{value:.2f}"

    def read_probe(filename):
        qois_vals, names = load_qois(filename)

        data = {
            "Uxx": np.asarray(
                get(qois_vals, names, "U_bar_XX")[slice_start:],
                dtype=float,
            ),
            "Uyy": np.asarray(
                get(qois_vals, names, "U_bar_YY")[slice_start:],
                dtype=float,
            ),
            "Uxy": np.asarray(
                get(qois_vals, names, "U_bar_XY")[slice_start:],
                dtype=float,
            ),
            "Uyx": np.asarray(
                get(qois_vals, names, "U_bar_YX")[slice_start:],
                dtype=float,
            ),
            "Qx": np.asarray(
                get(qois_vals, names, "Q_l_avg_x")[slice_start:],
                dtype=float,
            ),
            "Qy": np.asarray(
                get(qois_vals, names, "Q_l_avg_y")[slice_start:],
                dtype=float,
            ),
            "gx": np.asarray(
                get(qois_vals, names, "grad_p_bar_avg_x")[slice_start:],
                dtype=float,
            ),
            "gy": np.asarray(
                get(qois_vals, names, "grad_p_bar_avg_y")[slice_start:],
                dtype=float,
            ),
        }

        npts = len(data["Uxx"])

        for key, val in data.items():
            if len(val) != npts:
                raise ValueError(f"Inconsistent length for {key} in {filename}")

        return data

    def continuous_axis_angle_deg(theta_raw):
        theta_raw = np.asarray(theta_raw, dtype=float)
        theta_cont = np.empty_like(theta_raw)

        if len(theta_raw) == 0:
            return theta_cont

        theta_cont[0] = theta_raw[0]

        for i in range(1, len(theta_raw)):
            delta = (
                theta_raw[i]
                - theta_cont[i - 1]
                + 90.0
            ) % 180.0 - 90.0

            theta_cont[i] = theta_cont[i - 1] + delta

        return theta_cont

    def continuous_axis_angle_rad(theta_raw):
        theta_raw = np.asarray(theta_raw, dtype=float)
        theta_cont = np.empty_like(theta_raw)

        if len(theta_raw) == 0:
            return theta_cont

        theta_cont[0] = theta_raw[0]

        for i in range(1, len(theta_raw)):
            delta = (
                theta_raw[i]
                - theta_cont[i - 1]
                + 0.5 * np.pi
            ) % np.pi - 0.5 * np.pi

            theta_cont[i] = theta_cont[i - 1] + delta

        return theta_cont

    def lift_negative_vertical_deg(theta):
        theta = np.asarray(theta, dtype=float).copy()
        theta[theta <= theta_lift_threshold_deg] += 180.0
        return theta

    def lift_negative_vertical_rad(theta):
        theta = np.asarray(theta, dtype=float).copy()
        threshold = np.deg2rad(theta_lift_threshold_deg)
        theta[theta <= threshold] += np.pi
        return theta

    def principal_quantities(K_list):
        K1 = []
        K2 = []
        theta = []

        for K in K_list:
            Ksym = 0.5 * (K + K.T)

            a = Ksym[0, 0]
            b = Ksym[0, 1]
            c = Ksym[1, 1]

            tr = a + c
            delta = np.sqrt((a - c) ** 2 + 4.0 * b ** 2)

            lam1 = 0.5 * (tr + delta)
            lam2 = 0.5 * (tr - delta)

            angle = 0.5 * np.arctan2(2.0 * b, a - c)

            K1.append(lam1)
            K2.append(lam2)
            theta.append(angle)

        K1 = np.asarray(K1, dtype=float)
        K2 = np.asarray(K2, dtype=float)
        theta = np.asarray(theta, dtype=float)

        if theta_in_degrees:
            theta = np.rad2deg(theta)
            theta = continuous_axis_angle_deg(theta)
            theta = lift_negative_vertical_deg(theta)
        else:
            theta = continuous_axis_angle_rad(theta)
            theta = lift_negative_vertical_rad(theta)

        return K1, K2, theta

    def plot_stream_subplot(ax, xdmf_file):
        if not os.path.exists(xdmf_file):
            ax.text(
                0.5,
                0.5,
                "missing xdmf",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            return None

        reader0 = pv.get_reader(xdmf_file)

        if (
            hasattr(reader0, "number_time_points")
            and reader0.number_time_points > 0
        ):
            reader0.set_active_time_point(0)

        mesh0 = reader0.read().cell_data_to_point_data()
        surf0 = mesh0.extract_surface().triangulate()

        reader = pv.get_reader(xdmf_file)

        if (
            hasattr(reader, "number_time_points")
            and reader.number_time_points > 0
        ):
            reader.set_active_time_point(reader.number_time_points - 1)

        mesh = reader.read().cell_data_to_point_data()

        warped = mesh.warp_by_vector(
            "U_tot",
            factor=stream_scale,
        )

        surf = warped.extract_surface().triangulate()

        pts = surf.points[:, :2]
        faces = surf.faces.reshape(-1, 4)[:, 1:4]

        pts0 = surf0.points[:, :2]
        faces0 = surf0.faces.reshape(-1, 4)[:, 1:4]

        c0 = pts0.mean(axis=0)
        c = pts.mean(axis=0)
        pts0_shift = pts0 + (c - c0)

        p = np.asarray(
            surf.point_data["pl_tot"],
            dtype=float,
        )

        q = np.asarray(
            surf.point_data["q_l"][:, :2],
            dtype=float,
        )

        x = pts[:, 0]
        y = pts[:, 1]
        qx = q[:, 0]
        qy = q[:, 1]

        triang = mtri.Triangulation(
            x,
            y,
            triangles=faces,
        )

        triang0 = mtri.Triangulation(
            pts0_shift[:, 0],
            pts0_shift[:, 1],
            triangles=faces0,
        )

        interp_p = mtri.LinearTriInterpolator(triang, p)
        interp_qx = mtri.LinearTriInterpolator(triang, qx)
        interp_qy = mtri.LinearTriInterpolator(triang, qy)

        xi = np.linspace(x.min(), x.max(), stream_grid_n)
        yi = np.linspace(y.min(), y.max(), stream_grid_n)
        X, Y = np.meshgrid(xi, yi)

        P = interp_p(X, Y)
        QX = interp_qx(X, Y)
        QY = interp_qy(X, Y)

        finder = triang.get_trifinder()
        inside = finder(X, Y) != -1

        P_mask = np.ma.getmaskarray(P) | (~inside)
        QX_mask = np.ma.getmaskarray(QX) | (~inside)
        QY_mask = np.ma.getmaskarray(QY) | (~inside)

        P = np.ma.array(P, mask=P_mask)
        QX = np.ma.array(QX, mask=QX_mask)
        QY = np.ma.array(QY, mask=QY_mask)

        ax.tripcolor(
            triang0,
            np.ones(len(pts0_shift)),
            shading="gouraud",
            cmap="Greys",
            vmin=0.0,
            vmax=2.0,
            alpha=0.7,
            edgecolors="none",
            zorder=0,
        )

        ax.triplot(
            triang0,
            color="0.75",
            linewidth=0.35,
            alpha=0.45,
            zorder=1,
        )

        cf = ax.contourf(
            X,
            Y,
            P,
            levels=60,
            cmap="coolwarm",
            alpha=0.65,
            zorder=2,
        )

        stream_kwargs = {
            "density": stream_density,
            "color": "#66F7FF",
            "linewidth": 1.2,
            "arrowsize": 1.2,
            "arrowstyle": "->",
            "minlength": 0.02,
            "maxlength": 10.0,
            "integration_direction": "both",
        }

        try:
            sp = ax.streamplot(
                xi,
                yi,
                QX,
                QY,
                broken_streamlines=False,
                **stream_kwargs,
            )
        except TypeError:
            sp = ax.streamplot(
                xi,
                yi,
                QX,
                QY,
                **stream_kwargs,
            )

        sp.lines.set_zorder(4)
        sp.arrows.set_zorder(5)

        pad_x = 0.01 * (x.max() - x.min())
        pad_y = 0.01 * (y.max() - y.min())

        xmin = min(x.min(), pts0_shift[:, 0].min()) - pad_x
        xmax = max(x.max(), pts0_shift[:, 0].max()) + pad_x
        ymin = min(y.min(), pts0_shift[:, 1].min()) - pad_y
        ymax = max(y.max(), pts0_shift[:, 1].max()) + pad_y

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.margins(0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        return cf

    colors = {
        "K1": "#0072B2",
        "K2": "#D55E00",
        "theta": "#009E73",
        "undef": "#009E73",
    }

    markers = {
        "K1": "o",
        "K2": "s",
        "theta": "^",
    }

    n_cols = len(r0_list)
    fig_width = max(4.3 * n_cols, 7.0)

    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(fig_width, 10.8),
        sharex=False,
        sharey=False,
        squeeze=False,
        gridspec_kw={
            "height_ratios": [1.0, 0.85, 1.25],
            "hspace": 0.22,
        },
    )

    eig_axes = axes[0]
    theta_axes = axes[1]
    stream_axes = axes[2]

    cf_last = None
    all_theta_values = []
    phi_values_by_column = []

    for i_r0, r0 in enumerate(r0_list):
        eig_ax = eig_axes[i_r0]
        theta_ax = theta_axes[i_r0]
        stream_ax = stream_axes[i_r0]

        phi_val = None
        theta_vline_added = False

        for pf in pf_list:
            filename_gx = build_filename(r0, pf, "gx")
            filename_gy = build_filename(r0, pf, "gy")

            if not os.path.exists(filename_gx):
                print(f"[WARNING] File missing: {filename_gx}")
                continue

            if not os.path.exists(filename_gy):
                print(f"[WARNING] File missing: {filename_gy}")
                continue

            if phi_val is None:
                phi_val = read_phi_from_metadata(
                    r0,
                    pf,
                    "gx",
                )

            data_gx = read_probe(filename_gx)
            data_gy = read_probe(filename_gy)

            Uxx = data_gx["Uxx"]
            Uyy = data_gx["Uyy"]
            Uxy = data_gx["Uxy"]
            Uyx = data_gx["Uyx"]

            for key in ["Uxx", "Uyy", "Uxy", "Uyx"]:
                if not np.allclose(
                    data_gx[key],
                    data_gy[key],
                    rtol=1e-6,
                    atol=1e-10,
                ):
                    print(
                        f"[WARNING] {key} differs between gx and gy "
                        f"probes for r0={r0}, pf={pf}"
                    )

            gx = data_gx["gx"]
            gy = data_gy["gy"]

            Kxx = -data_gx["Qx"] / (gx + eps)
            Kyx = -data_gx["Qy"] / (gx + eps)
            Kxy = -data_gy["Qx"] / (gy + eps)
            Kyy = -data_gy["Qy"] / (gy + eps)

            xvals = data_gx[x_components[x_component]]
            npts = len(xvals)
            markevery = max(1, npts // 8)

            if not theta_vline_added and len(xvals) > 0:
                if np.nanmin(xvals) <= 0.0 <= np.nanmax(xvals):
                    theta_x_ref = 0.0
                else:
                    theta_x_ref = xvals[0]

                theta_ax.axvline(
                    x=theta_x_ref,
                    color=colors["theta"],
                    linestyle="--",
                    linewidth=2.0,
                    alpha=1.0,
                    zorder=0,
                )

                theta_vline_added = True

            K_list = []
            F_list = []

            for n in range(npts):
                K_list.append(
                    np.array(
                        [
                            [Kxx[n], Kxy[n]],
                            [Kyx[n], Kyy[n]],
                        ],
                        dtype=float,
                    )
                )

                F_list.append(
                    np.array(
                        [
                            [1.0 + Uxx[n], Uxy[n]],
                            [Uyx[n], 1.0 + Uyy[n]],
                        ],
                        dtype=float,
                    )
                )

            K1, K2, theta = principal_quantities(K_list)

            K1_0 = K1[0]
            K2_0 = K2[0]

            if abs(K1_0) < eps:
                raise ValueError(
                    f"K1_0 is too small for r0={r0}, pf={pf}"
                )

            if abs(K2_0) < eps:
                raise ValueError(
                    f"K2_0 is too small for r0={r0}, pf={pf}"
                )

            eig_ax.plot(
                xvals,
                K1 / K1_0,
                color=colors["K1"],
                linestyle="--",
                linewidth=2.0,
                marker=markers["K1"],
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor=colors["K1"],
                markeredgewidth=1.0,
                markevery=markevery,
            )

            eig_ax.plot(
                xvals,
                K2 / K2_0,
                color=colors["K2"],
                linestyle="--",
                linewidth=2.0,
                marker=markers["K2"],
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor=colors["K2"],
                markeredgewidth=1.0,
                markevery=markevery,
            )

            theta_plot = theta.copy()

            if len(theta_plot) > 0:
                theta_plot[0] = np.nan

            theta_ax.plot(
                xvals,
                theta_plot,
                color=colors["theta"],
                linestyle="--",
                linewidth=2.0,
                marker=markers["theta"],
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor=colors["theta"],
                markeredgewidth=1.0,
                markevery=markevery,
            )

            all_theta_values.extend(
                theta_plot[np.isfinite(theta_plot)].tolist()
            )

            if add_prediction:
                K0 = K_list[0]
                K_pred_list = []

                for n in range(npts):
                    F = F_list[n]
                    J = float(np.linalg.det(F))

                    if abs(J) < 1e-14:
                        K_pred_list.append(
                            np.full((2, 2), np.nan)
                        )
                        continue

                    try:
                        Finv = np.linalg.inv(F)
                    except np.linalg.LinAlgError:
                        K_pred_list.append(
                            np.full((2, 2), np.nan)
                        )
                        continue

                    K_pred_list.append(
                        J * (Finv @ K0 @ Finv.T)
                    )

                K1_pred, K2_pred, theta_pred = principal_quantities(
                    K_pred_list
                )

                eig_ax.plot(
                    xvals,
                    K1_pred / K1_0,
                    color=colors["K1"],
                    linestyle="-",
                    linewidth=2.0,
                )

                eig_ax.plot(
                    xvals,
                    K2_pred / K2_0,
                    color=colors["K2"],
                    linestyle="-",
                    linewidth=2.0,
                )

                theta_pred_plot = theta_pred.copy()

                if len(theta_pred_plot) > 0:
                    theta_pred_plot[0] = np.nan

                theta_ax.plot(
                    xvals,
                    theta_pred_plot,
                    color=colors["theta"],
                    linestyle="-",
                    linewidth=2.0,
                )

                all_theta_values.extend(
                    theta_pred_plot[
                        np.isfinite(theta_pred_plot)
                    ].tolist()
                )

            asym = []

            for K in K_list:
                denom = np.linalg.norm(K, ord="fro") + eps
                asym.append(
                    np.linalg.norm(K - K.T, ord="fro") / denom
                )

            print(f"r0 = {r0}, pf = {pf}")
            print(f"K1_0 = {K1_0}")
            print(f"K2_0 = {K2_0}")
            print(f"max asymmetry = {np.nanmax(asym)}")

        phi_values_by_column.append(phi_val)

        if phi_val is not None:
            eig_ax.set_title(
                rf"$\tilde{{\Phi}}_{{g0}} = {phi_val:.2f}$"
            )
        else:
            eig_ax.set_title(
                rf"$r_0={r0}$"
            )

        eig_ax.grid(False)
        theta_ax.grid(False)

        eig_ax.tick_params(
            axis="x",
            labelbottom=False,
        )

        eig_ax.tick_params(
            axis="y",
            labelsize=11,
        )

        theta_ax.set_xlabel(
            x_labels[x_component]
        )

        theta_ax.tick_params(
            axis="x",
            labelsize=11,
        )

        theta_ax.tick_params(
            axis="y",
            labelsize=11,
        )

        if i_r0 > 0:
            eig_ax.tick_params(
                axis="y",
                left=False,
                labelleft=False,
            )

            theta_ax.tick_params(
                axis="y",
                left=False,
                labelleft=False,
            )

        xdmf_file = build_xdmf_filename(
            r0,
            stream_pf,
            stream_probe,
        )

        cf_last = plot_stream_subplot(
            stream_ax,
            xdmf_file,
        )

    eig_ymins = []
    eig_ymaxs = []

    for ax in eig_axes:
        ymin, ymax = ax.get_ylim()
        eig_ymins.append(ymin)
        eig_ymaxs.append(ymax)

    if eig_ymins and eig_ymaxs:
        ymin = min(eig_ymins)
        ymax = max(eig_ymaxs)
        dy = ymax - ymin

        if dy > 0:
            ymin -= 0.05 * dy
            ymax += 0.05 * dy

        for ax in eig_axes:
            ax.set_ylim(ymin, ymax)

    theta_valid = [
        value
        for value in all_theta_values
        if np.isfinite(value)
    ]

    if theta_in_degrees:
        if theta_valid:
            theta_min = min(theta_valid)
            theta_max = max(theta_valid)
            dtheta = theta_max - theta_min

            if dtheta < 1e-8:
                pad = 2.0
            else:
                pad = max(2.0, 0.15 * dtheta)

            theta_low = theta_min - pad
            theta_high = theta_max + pad
        else:
            theta_low = -5.0
            theta_high = 5.0

        for ax in theta_axes:
            ax.set_ylim(theta_low, theta_high)
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=4)
            )

    else:
        if theta_valid:
            theta_min = min(theta_valid)
            theta_max = max(theta_valid)
            dtheta = theta_max - theta_min

            if dtheta < 1e-8:
                pad = np.deg2rad(2.0)
            else:
                pad = max(
                    np.deg2rad(2.0),
                    0.15 * dtheta,
                )

            theta_low = theta_min - pad
            theta_high = theta_max + pad
        else:
            theta_low = -np.deg2rad(5.0)
            theta_high = np.deg2rad(5.0)

        for ax in theta_axes:
            ax.set_ylim(theta_low, theta_high)
            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=4)
            )

    eig_axes[0].set_ylabel(
        r"$K_i/K_{i,0}$",
        labelpad=4,
    )

    theta_axes[0].set_ylabel(
        r"$\theta$ (deg)"
        if theta_in_degrees
        else r"$\theta$ (rad)",
        labelpad=4,
    )

    eig_axes[0].yaxis.set_label_coords(
        -0.15,
        0.5,
    )

    theta_axes[0].yaxis.set_label_coords(
        -0.18,
        0.5,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors["K1"],
            linestyle="--",
            linewidth=2.0,
            marker=markers["K1"],
            markerfacecolor="white",
            markeredgecolor=colors["K1"],
            label=r"$K_1/K_{1,0}$",
        ),
    ]

    if add_prediction:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=colors["K1"],
                linestyle="-",
                linewidth=2.0,
                label=r"$K_1^{kin}/K_{1,0}$",
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=colors["K2"],
            linestyle="--",
            linewidth=2.0,
            marker=markers["K2"],
            markerfacecolor="white",
            markeredgecolor=colors["K2"],
            label=r"$K_2/K_{2,0}$",
        )
    )

    if add_prediction:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=colors["K2"],
                linestyle="-",
                linewidth=2.0,
                label=r"$K_2^{kin}/K_{2,0}$",
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=colors["theta"],
            linestyle="--",
            linewidth=2.0,
            marker=markers["theta"],
            markerfacecolor="white",
            markeredgecolor=colors["theta"],
            label=r"$\theta$",
        )
    )

    if add_prediction:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=colors["theta"],
                linestyle="-",
                linewidth=2.0,
                label=r"$\theta^{kin}$",
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=colors["theta"],
            linestyle="--",
            linewidth=2.0,
            label=r"$\theta$ undefined at reference",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(legend_handles), 4),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        labelspacing=0.7,
    )

    plt.tight_layout(
        rect=(0.055, 0.17, 1.0, 0.89)
    )

    fig.canvas.draw()

    stream_positions = [
        ax.get_position()
        for ax in stream_axes
    ]

    if add_stream_colorbar and cf_last is not None:
        left = min(
            pos.x0
            for pos in stream_positions
        )

        bottom = min(
            pos.y0
            for pos in stream_positions
        )

        top = max(
            pos.y1
            for pos in stream_positions
        )

        cbar_width = 0.012
        cbar_pad = 0.050

        cax = fig.add_axes(
            [
                left - cbar_pad - cbar_width,
                bottom,
                cbar_width,
                top - bottom,
            ]
        )

        cbar = fig.colorbar(
            cf_last,
            cax=cax,
            orientation="vertical",
        )

        def sci_fmt(x, pos):
            if abs(x) < 1e-14:
                return "0"

            s = f"{x:.1e}"
            s = s.replace("e-0", "e-")
            s = s.replace("e+0", "e")
            s = s.replace("e+", "e")

            return s

        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(sci_fmt)
        )

        cbar.ax.tick_params(
            labelsize=9
        )

        cbar.set_label(
            r"$p_\ell$ (kPa)"
        )

    if len(stream_positions) > 0:
        first_pos = stream_positions[0]
        last_pos = stream_positions[-1]

        arrow_y = min(
            pos.y0
            for pos in stream_positions
        ) - 0.055

        arrow_x0 = first_pos.x0
        arrow_x1 = last_pos.x1

        porosity_arrow = FancyArrowPatch(
            (arrow_x0, arrow_y),
            (arrow_x1, arrow_y),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=15,
            linewidth=1.5,
            color="black",
            clip_on=False,
        )

        fig.add_artist(porosity_arrow)

        for ax, phi_value in zip(
            stream_axes,
            phi_values_by_column,
        ):
            pos = ax.get_position()
            x_center = 0.5 * (pos.x0 + pos.x1)

            fig.text(
                x_center,
                arrow_y + 0.010,
                rf"${format_phi_value(phi_value)}$",
                ha="center",
                va="bottom",
            )

        fig.text(
            0.5 * (arrow_x0 + arrow_x1),
            arrow_y - 0.030,
            r"$\tilde{\Phi}_{g0}$",
            ha="center",
            va="top",
        )

    plt.savefig(
        save_name,
        bbox_inches="tight",
        dpi=300,
    )

    if show_plot:
        plt.show()

    plt.close()

    print(f"Saved: {save_name}")



def plot_gas_pressure_loading_summary(
    res_folder,
    res_basename_prefix,
    r0,
    mode_list=("stretch-x", "volumic", "shear"),
    pf_list=(0.0, 0.2),
    probe_list=("gx", "gy"),
    phi=None,
    slice_start=5,
    eps=1e-12,
    save_name="plots/Figure8_pg_compare.png",
    show_plot=False,
    stream_probe="gx",
    stream_density=0.8,
    stream_scale=1.0,
    stream_grid_n=500,
    add_stream_colorbar=True,
    theta_lift_threshold_deg=-85.0,
):
    def fmt_pg(x):
        if abs(float(x)) < 1e-14:
            return "0"
        return f"{float(x):.3g}"

    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

    if len(pf_list) != 2:
        raise ValueError("pf_list should contain exactly two values, e.g. (0.0, 0.2).")

    if set(probe_list) != {"gx", "gy"}:
        raise ValueError("probe_list must contain gx and gy.")

    mode_to_xkey = {
        "stretch-x": "Uxx",
        "volumic": "Uxx",
        "shear": "Uxy",
    }

    mode_to_xlabel = {
        "stretch-x": r"$E_x$",
        "volumic": r"$E_x=E_y$",
        "shear": r"$E_{xy}$",
    }

    mode_to_label = {
        "stretch-x": r"Uniaxial loading",
        "volumic": r"Pure volumetric loading",
        "shear": r"Simple shear loading",
    }

    colors = {
        "pg0": "#204a9a",
        "pg1": "#b22222",
    }

    markers = {
        "K1": "o",
        "K2": "s",
        "theta": "^",
    }

    linestyles = {
        "K1": "-",
        "K2": "--",
        "theta": "-.",
    }

    def mode_filename_candidates(mode):
        return [mode]

    def build_basename(mode, pf, probe, suffix=None):
        first_filename = None

        for mode_file in mode_filename_candidates(mode):
            filename = f"{res_folder}/{res_basename_prefix}-{mode_file}-r0={r0}-pf={pf}-{probe}"

            if phi is not None:
                phi_str = f"{phi:.3f}".replace(".", "p") if isinstance(phi, float) else str(phi)
                filename += f"-phi={phi_str}"

            if first_filename is None:
                first_filename = filename

            if suffix is not None and os.path.exists(filename + suffix):
                return filename

        return first_filename

    def build_qois_filename(mode, pf, probe):
        return build_basename(mode, pf, probe, suffix="-qois.dat") + "-qois.dat"

    def build_xdmf_filename(mode, pf, probe):
        return build_basename(mode, pf, probe, suffix=".xdmf") + ".xdmf"

    def read_phi_from_metadata(mode, pf, probe="gx"):
        metadata_file = build_basename(mode, pf, probe, suffix="-metadata.json") + "-metadata.json"

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for key in ["mesh_porosity", "porosity", "phi"]:
                if key in metadata:
                    return float(metadata[key])

        return None

    def read_probe(filename):
        qois_vals, names = load_qois(filename)

        data = {
            "Uxx": np.asarray(get(qois_vals, names, "U_bar_XX")[slice_start:], dtype=float),
            "Uyy": np.asarray(get(qois_vals, names, "U_bar_YY")[slice_start:], dtype=float),
            "Uxy": np.asarray(get(qois_vals, names, "U_bar_XY")[slice_start:], dtype=float),
            "Uyx": np.asarray(get(qois_vals, names, "U_bar_YX")[slice_start:], dtype=float),
            "Qx": np.asarray(get(qois_vals, names, "Q_l_avg_x")[slice_start:], dtype=float),
            "Qy": np.asarray(get(qois_vals, names, "Q_l_avg_y")[slice_start:], dtype=float),
            "gx": np.asarray(get(qois_vals, names, "grad_p_bar_avg_x")[slice_start:], dtype=float),
            "gy": np.asarray(get(qois_vals, names, "grad_p_bar_avg_y")[slice_start:], dtype=float),
        }

        npts = len(data["Uxx"])

        for key, val in data.items():
            if len(val) != npts:
                raise ValueError(f"Inconsistent length for {key} in {filename}")

        return data

    def read_K_and_U(mode, pf):
        file_gx = build_qois_filename(mode, pf, "gx")
        file_gy = build_qois_filename(mode, pf, "gy")

        if not os.path.exists(file_gx):
            raise FileNotFoundError(file_gx)

        if not os.path.exists(file_gy):
            raise FileNotFoundError(file_gy)

        data_gx = read_probe(file_gx)
        data_gy = read_probe(file_gy)

        gx = data_gx["gx"]
        gy = data_gy["gy"]

        Kxx = -data_gx["Qx"] / (gx + eps)
        Kyx = -data_gx["Qy"] / (gx + eps)
        Kxy = -data_gy["Qx"] / (gy + eps)
        Kyy = -data_gy["Qy"] / (gy + eps)

        K_list = []

        for n in range(len(Kxx)):
            K_list.append(
                np.array(
                    [
                        [Kxx[n], Kxy[n]],
                        [Kyx[n], Kyy[n]],
                    ],
                    dtype=float,
                )
            )

        U = {
            "Uxx": data_gx["Uxx"],
            "Uyy": data_gx["Uyy"],
            "Uxy": data_gx["Uxy"],
            "Uyx": data_gx["Uyx"],
        }

        return K_list, U

    def continuous_axis_angle_deg(theta_raw):
        theta_raw = np.asarray(theta_raw, dtype=float)
        theta_cont = np.empty_like(theta_raw)

        if len(theta_raw) == 0:
            return theta_cont

        theta_cont[0] = theta_raw[0]

        for i in range(1, len(theta_raw)):
            delta = (theta_raw[i] - theta_cont[i - 1] + 90.0) % 180.0 - 90.0
            theta_cont[i] = theta_cont[i - 1] + delta

        return theta_cont

    def lift_negative_vertical_deg(theta):
        theta = np.asarray(theta, dtype=float).copy()
        theta[theta <= theta_lift_threshold_deg] += 180.0
        return theta

    def principal_quantities(K_list):
        K1 = []
        K2 = []
        theta = []

        for K in K_list:
            Ksym = 0.5 * (K + K.T)

            a = Ksym[0, 0]
            b = Ksym[0, 1]
            c = Ksym[1, 1]

            tr = a + c
            delta = np.sqrt((a - c) ** 2 + 4.0 * b ** 2)

            lam1 = 0.5 * (tr + delta)
            lam2 = 0.5 * (tr - delta)

            angle = 0.5 * np.arctan2(2.0 * b, a - c)

            K1.append(lam1)
            K2.append(lam2)
            theta.append(angle)

        K1 = np.asarray(K1, dtype=float)
        K2 = np.asarray(K2, dtype=float)
        theta = np.rad2deg(np.asarray(theta, dtype=float))
        theta = continuous_axis_angle_deg(theta)
        theta = lift_negative_vertical_deg(theta)

        return K1, K2, theta

    def make_xvals(mode, U):
        return U[mode_to_xkey[mode]]

    def weighted_kde(values, weights, xgrid):
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)

        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        values = values[valid]
        weights = weights[valid]

        if len(values) == 0:
            return np.zeros_like(xgrid)

        weights = weights / (np.sum(weights) + eps)

        mu = np.sum(weights * values)
        var = np.sum(weights * (values - mu) ** 2)
        sigma = np.sqrt(max(var, eps))

        n_eff = 1.0 / (np.sum(weights ** 2) + eps)
        bw = 1.06 * sigma * max(n_eff, 2.0) ** (-1.0 / 5.0)

        span = max(np.max(values) - np.min(values), eps)
        bw = max(bw, 0.04 * span)

        z = (xgrid[:, None] - values[None, :]) / bw
        density = np.sum(
            weights[None, :] * np.exp(-0.5 * z ** 2),
            axis=1,
        ) / (np.sqrt(2.0 * np.pi) * bw)

        return density

    def read_q_distribution(mode, pf, probe):
        xdmf_file = build_xdmf_filename(mode, pf, probe)

        if not os.path.exists(xdmf_file):
            raise FileNotFoundError(xdmf_file)

        reader = pv.get_reader(xdmf_file)

        if hasattr(reader, "number_time_points") and reader.number_time_points > 0:
            reader.set_active_time_point(reader.number_time_points - 1)

        mesh = reader.read()
        mesh = mesh.cell_data_to_point_data()

        warped = mesh.warp_by_vector("U_tot", factor=stream_scale)
        surf = warped.extract_surface().triangulate()
        surf = surf.point_data_to_cell_data()

        if "q_l" not in surf.cell_data:
            raise ValueError(f"'q_l' not found in cell data of {xdmf_file}")

        q = np.asarray(surf.cell_data["q_l"], dtype=float)[:, :2]
        qmag = np.linalg.norm(q, axis=1)

        valid = np.isfinite(qmag)
        qmag = qmag[valid]

        if len(qmag) == 0:
            return {
                "values": np.array([]),
                "mean": np.nan,
                "std": np.nan,
                "n_cells": 0,
            }

        return {
            "values": qmag,
            "mean": float(np.mean(qmag)),
            "std": float(np.std(qmag)),
            "n_cells": int(len(qmag)),
        }

    def plot_distribution_subplot(ax, data0, data1, xlim, ymax):
        vals0 = data0["values"]
        vals1 = data1["values"]

        bins = np.linspace(xlim[0], xlim[1], 26)
        xgrid = np.linspace(xlim[0], xlim[1], 400)
        bin_width = bins[1] - bins[0]

        if len(vals0) > 0:
            weights0 = np.ones_like(vals0, dtype=float) * 100.0 / len(vals0)

            ax.hist(
                vals0,
                bins=bins,
                weights=weights0,
                density=False,
                color=colors["pg0"],
                alpha=0.32,
                edgecolor="white",
                linewidth=0.8,
            )

        if len(vals1) > 0:
            weights1 = np.ones_like(vals1, dtype=float) * 100.0 / len(vals1)

            ax.hist(
                vals1,
                bins=bins,
                weights=weights1,
                density=False,
                color=colors["pg1"],
                alpha=0.25,
                edgecolor="white",
                linewidth=0.8,
            )

        if len(vals0) > 1:
            w0 = np.ones_like(vals0, dtype=float) / len(vals0)
            kde0 = weighted_kde(vals0, w0, xgrid)

            ax.plot(
                xgrid,
                kde0 * bin_width * 100.0,
                color=colors["pg0"],
                linewidth=2.0,
            )

        if len(vals1) > 1:
            w1 = np.ones_like(vals1, dtype=float) / len(vals1)
            kde1 = weighted_kde(vals1, w1, xgrid)

            ax.plot(
                xgrid,
                kde1 * bin_width * 100.0,
                color=colors["pg1"],
                linewidth=2.0,
            )

        if np.isfinite(data0["mean"]):
            ax.axvline(
                data0["mean"],
                color=colors["pg0"],
                linewidth=1.8,
            )

        if np.isfinite(data1["mean"]):
            ax.axvline(
                data1["mean"],
                color=colors["pg1"],
                linewidth=1.8,
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, ymax)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlabel(r"$|\mathbf{q}_{\ell}|$", fontsize=11)

    def plot_stream_subplot(ax, xdmf_file):
        if not os.path.exists(xdmf_file):
            ax.text(0.5, 0.5, "missing xdmf", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            return None

        reader = pv.get_reader(xdmf_file)

        if hasattr(reader, "number_time_points") and reader.number_time_points > 0:
            reader.set_active_time_point(0)
            mesh0 = reader.read()
            reader.set_active_time_point(reader.number_time_points - 1)
            meshf = reader.read()
        else:
            mesh0 = reader.read()
            meshf = mesh0.copy()

        mesh0 = mesh0.cell_data_to_point_data()
        meshf = meshf.cell_data_to_point_data()

        surf0 = mesh0.extract_surface().triangulate()
        warped = meshf.warp_by_vector("U_tot", factor=stream_scale)
        surf = warped.extract_surface().triangulate()

        pts0 = surf0.points[:, :2]
        faces0 = surf0.faces.reshape(-1, 4)[:, 1:4]

        pts = surf.points[:, :2]
        faces = surf.faces.reshape(-1, 4)[:, 1:4]

        center0 = np.array(
            [
                0.5 * (pts0[:, 0].min() + pts0[:, 0].max()),
                0.5 * (pts0[:, 1].min() + pts0[:, 1].max()),
            ]
        )
        center = np.array(
            [
                0.5 * (pts[:, 0].min() + pts[:, 0].max()),
                0.5 * (pts[:, 1].min() + pts[:, 1].max()),
            ]
        )
        pts0_shift = pts0 + (center - center0)

        triang0 = mtri.Triangulation(
            pts0_shift[:, 0],
            pts0_shift[:, 1],
            triangles=faces0,
        )

        p = np.asarray(surf.point_data["pl_tot"], dtype=float)
        q = np.asarray(surf.point_data["q_l"][:, :2], dtype=float)

        x = pts[:, 0]
        y = pts[:, 1]
        qx = q[:, 0]
        qy = q[:, 1]

        triang = mtri.Triangulation(x, y, triangles=faces)

        interp_p = mtri.LinearTriInterpolator(triang, p)
        interp_qx = mtri.LinearTriInterpolator(triang, qx)
        interp_qy = mtri.LinearTriInterpolator(triang, qy)

        xi = np.linspace(x.min(), x.max(), stream_grid_n)
        yi = np.linspace(y.min(), y.max(), stream_grid_n)
        X, Y = np.meshgrid(xi, yi)

        P = interp_p(X, Y)
        QX = interp_qx(X, Y)
        QY = interp_qy(X, Y)

        finder = triang.get_trifinder()
        inside = finder(X, Y) != -1

        P_mask = np.ma.getmaskarray(P) | (~inside)
        QX_mask = np.ma.getmaskarray(QX) | (~inside)
        QY_mask = np.ma.getmaskarray(QY) | (~inside)

        P = np.ma.array(P, mask=P_mask)
        QX = np.ma.array(QX, mask=QX_mask)
        QY = np.ma.array(QY, mask=QY_mask)

        ax.tripcolor(
            triang0,
            np.ones(len(pts0_shift)),
            shading="gouraud",
            cmap="Greys",
            vmin=0.0,
            vmax=2.0,
            alpha=0.7,
            edgecolors="none",
            zorder=0,
        )

        ax.triplot(
            triang0,
            color="0.75",
            linewidth=0.35,
            alpha=0.45,
            zorder=1,
        )

        cf = ax.contourf(
            X,
            Y,
            P,
            levels=60,
            cmap="coolwarm",
            alpha=0.65,
            zorder=2,
        )

        stream_kwargs = dict(
            density=stream_density,
            color="#66F7FF",
            linewidth=1.1,
            arrowsize=1.1,
            arrowstyle="->",
            minlength=0.02,
            maxlength=10.0,
            integration_direction="both",
        )

        try:
            sp = ax.streamplot(
                xi,
                yi,
                QX,
                QY,
                broken_streamlines=False,
                **stream_kwargs,
            )
        except TypeError:
            sp = ax.streamplot(
                xi,
                yi,
                QX,
                QY,
                **stream_kwargs,
            )

        sp.lines.set_zorder(4)
        sp.arrows.set_zorder(5)

        xmin = min(pts0_shift[:, 0].min(), x.min())
        xmax = max(pts0_shift[:, 0].max(), x.max())
        ymin = min(pts0_shift[:, 1].min(), y.min())
        ymax = max(pts0_shift[:, 1].max(), y.max())

        pad_x = 0.02 * (xmax - xmin)
        pad_y = 0.02 * (ymax - ymin)

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.margins(0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        return cf

    dist_cache = {}
    all_q = []

    for mode in mode_list:
        for pf in pf_list:
            for probe in probe_list:
                try:
                    data = read_q_distribution(mode, pf, probe)
                    dist_cache[(mode, pf, probe)] = data

                    if len(data["values"]) > 0:
                        all_q.extend(data["values"].tolist())

                except FileNotFoundError:
                    dist_cache[(mode, pf, probe)] = {
                        "values": np.array([]),
                        "mean": np.nan,
                        "std": np.nan,
                        "n_cells": 0,
                    }

    if len(all_q) > 0:
        all_q = np.asarray(all_q, dtype=float)
        qmin = 0.0
        qmax = np.nanpercentile(all_q, 99.2)

        if not np.isfinite(qmax) or qmax <= qmin:
            qmax = np.nanmax(all_q)

        qpad = 0.06 * max(qmax - qmin, eps)
        dist_xlim = (qmin, qmax + qpad)

    else:
        dist_xlim = (0.0, 1.0)

    dist_ymax = 1.0
    bins_tmp = np.linspace(dist_xlim[0], dist_xlim[1], 26)
    xgrid_tmp = np.linspace(dist_xlim[0], dist_xlim[1], 400)
    bin_width_tmp = bins_tmp[1] - bins_tmp[0]

    for key, data in dist_cache.items():
        vals = data["values"]

        if len(vals) == 0:
            continue

        weights = np.ones_like(vals, dtype=float) * 100.0 / len(vals)
        hist, _ = np.histogram(vals, bins=bins_tmp, weights=weights, density=False)
        local_max = np.nanmax(hist)

        if len(vals) > 1:
            ww = np.ones_like(vals, dtype=float) / len(vals)
            kde = weighted_kde(vals, ww, xgrid_tmp)
            kde_percent = kde * bin_width_tmp * 100.0
            local_max = max(local_max, np.nanmax(kde_percent))

        if np.isfinite(local_max):
            dist_ymax = max(dist_ymax, 1.12 * local_max)

    n_rows = len(mode_list)

    fig, axes = plt.subplots(
        n_rows,
        6,
        figsize=(22.0, 3.25 * n_rows),
        squeeze=False,
        gridspec_kw={
            "width_ratios": [1.15, 0.95, 1.00, 1.00, 1, 0.9],
            "wspace": 0.30,
            "hspace": 0.36,
        },
    )

    cf_last = None
    all_K_values = []
    phi_val = None

    for i_mode, mode in enumerate(mode_list):
        if mode not in mode_to_xkey:
            raise ValueError(f"Unknown mode: {mode}")

        ax_K = axes[i_mode, 0]
        ax_theta = axes[i_mode, 1]
        ax_dist_gx = axes[i_mode, 2]
        ax_dist_gy = axes[i_mode, 3]
        ax_pf0 = axes[i_mode, 4]
        ax_pf1 = axes[i_mode, 5]

        local_theta_values = []
        theta_vline_added = False

        ref_K_list, ref_U = read_K_and_U(mode, pf_list[0])
        ref_K1, ref_K2, ref_theta = principal_quantities(ref_K_list)

        K1_ref = ref_K1[0]
        K2_ref = ref_K2[0]

        if abs(K1_ref) < eps:
            raise ValueError(f"K1_ref is too small for mode={mode}, r0={r0}.")

        if abs(K2_ref) < eps:
            raise ValueError(f"K2_ref is too small for mode={mode}, r0={r0}.")

        for i_pf, pf in enumerate(pf_list):
            if phi_val is None:
                phi_val = read_phi_from_metadata(mode, pf, "gx")

            K_list, U = read_K_and_U(mode, pf)
            xvals = make_xvals(mode, U)
            K1, K2, theta = principal_quantities(K_list)

            markevery = max(1, len(xvals) // 7)
            curve_color = colors["pg0"] if i_pf == 0 else colors["pg1"]

            ax_K.plot(
                xvals,
                K1 / K1_ref,
                color=curve_color,
                linestyle=linestyles["K1"],
                linewidth=2.0,
                marker=markers["K1"],
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=curve_color,
                markeredgewidth=1.0,
                markevery=markevery,
            )

            ax_K.plot(
                xvals,
                K2 / K2_ref,
                color=curve_color,
                linestyle=linestyles["K2"],
                linewidth=2.0,
                marker=markers["K2"],
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=curve_color,
                markeredgewidth=1.0,
                markevery=markevery,
            )

            theta_plot = theta.copy()
            if len(theta_plot) > 0:
                theta_plot[0] = np.nan

            if not theta_vline_added and len(xvals) > 0:
                if np.nanmin(xvals) <= 0.0 <= np.nanmax(xvals):
                    theta_x_ref = 0.0
                else:
                    theta_x_ref = xvals[0]

                ax_theta.axvline(
                    x=theta_x_ref,
                    color="0.55",
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.9,
                    zorder=0,
                )

                theta_vline_added = True

            ax_theta.plot(
                xvals,
                theta_plot,
                color=curve_color,
                linestyle=linestyles["theta"],
                linewidth=2.0,
                marker=markers["theta"],
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=curve_color,
                markeredgewidth=1.0,
                markevery=markevery,
            )

            all_K_values.extend((K1 / K1_ref)[np.isfinite(K1 / K1_ref)].tolist())
            all_K_values.extend((K2 / K2_ref)[np.isfinite(K2 / K2_ref)].tolist())

            theta_valid = theta_plot[np.isfinite(theta_plot)]
            local_theta_values.extend(theta_valid.tolist())

        data_gx_0 = dist_cache[(mode, pf_list[0], "gx")]
        data_gx_1 = dist_cache[(mode, pf_list[1], "gx")]
        data_gy_0 = dist_cache[(mode, pf_list[0], "gy")]
        data_gy_1 = dist_cache[(mode, pf_list[1], "gy")]

        plot_distribution_subplot(
            ax_dist_gx,
            data_gx_0,
            data_gx_1,
            dist_xlim,
            dist_ymax,
        )

        plot_distribution_subplot(
            ax_dist_gy,
            data_gy_0,
            data_gy_1,
            dist_xlim,
            dist_ymax,
        )

        xdmf_pg0 = build_xdmf_filename(mode, pf_list[0], stream_probe)
        xdmf_pg1 = build_xdmf_filename(mode, pf_list[1], stream_probe)

        cf_last = plot_stream_subplot(ax_pf0, xdmf_pg0)
        cf_last = plot_stream_subplot(ax_pf1, xdmf_pg1)

        ax_K.tick_params(axis="both", labelsize=10)
        ax_theta.tick_params(axis="both", labelsize=10)

        ax_K.grid(False)
        ax_theta.grid(False)

        ax_K.set_xlabel(mode_to_xlabel[mode])
        ax_theta.set_xlabel(mode_to_xlabel[mode])

        if len(local_theta_values) > 0:
            theta_min = min(local_theta_values)
            theta_max = max(local_theta_values)
            dtheta = theta_max - theta_min

            if dtheta < 1e-12:
                pad = 2.0
            else:
                pad = max(2.0, 0.15 * dtheta)

            ax_theta.set_ylim(theta_min - pad, theta_max + pad)
        else:
            ax_theta.set_ylim(-5.0, 5.0)

        ax_theta.yaxis.set_major_locator(MaxNLocator(nbins=4))

        if i_mode == 0:
            ax_K.set_title(r"Principal permeabilities")
            ax_theta.set_title(r"Principal direction")
            ax_dist_gx.set_title(r"$|\mathbf{q}_{\ell}|$ distribution ($g_x$ probe)")
            ax_dist_gy.set_title(r"$|\mathbf{q}_{\ell}|$ distribution ($g_y$ probe)")
            ax_pf0.set_title(rf"$p_g={fmt_pg(pf_list[0])}~\mathrm{{kPa}}$")
            ax_pf1.set_title(rf"$p_g={fmt_pg(pf_list[1])}~\mathrm{{kPa}}$")

        if i_mode == 0:
            ax_dist_gx.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=colors["pg0"],
                        linewidth=2.0,
                        label=rf"$p_g={pf_list[0]}~\mathrm{{kPa}}$",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=colors["pg1"],
                        linewidth=2.0,
                        label=rf"$p_g={pf_list[1]}~\mathrm{{kPa}}$",
                    ),
                ],
                loc="upper left",
                frameon=True,
            )

        if i_mode == 0:
            ax_dist_gx.set_ylabel("Frequency (%)")
        else:
            ax_dist_gx.set_ylabel("")

        ax_dist_gy.set_ylabel("")

    if all_K_values:
        K_min = min(all_K_values)
        K_max = max(all_K_values)
        dK = K_max - K_min

        if dK > 0:
            K_min -= 0.06 * dK
            K_max += 0.06 * dK

        for ax in axes[:, 0]:
            ax.set_ylim(K_min, K_max)

    axes[0, 0].set_ylabel(r"$K_i/K_{i,\mathrm{ref}}$")
    axes[0, 1].set_ylabel(r"$\theta$ (deg)")

    for i in range(1, n_rows):
        axes[i, 1].set_ylabel("")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors["pg0"],
            linestyle=linestyles["K1"],
            linewidth=2.0,
            marker=markers["K1"],
            markerfacecolor="white",
            markeredgecolor=colors["pg0"],
            label=rf"$K_1/K_{{1,\mathrm{{ref}}}},\ p_g={pf_list[0]}~\mathrm{{kPa}}$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["pg1"],
            linestyle=linestyles["K1"],
            linewidth=2.0,
            marker=markers["K1"],
            markerfacecolor="white",
            markeredgecolor=colors["pg1"],
            label=rf"$K_1/K_{{1,\mathrm{{ref}}}},\ p_g={pf_list[1]}~\mathrm{{kPa}}$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["pg0"],
            linestyle=linestyles["K2"],
            linewidth=2.0,
            marker=markers["K2"],
            markerfacecolor="white",
            markeredgecolor=colors["pg0"],
            label=rf"$K_2/K_{{2,\mathrm{{ref}}}},\ p_g={pf_list[0]}~\mathrm{{kPa}}$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["pg1"],
            linestyle=linestyles["K2"],
            linewidth=2.0,
            marker=markers["K2"],
            markerfacecolor="white",
            markeredgecolor=colors["pg1"],
            label=rf"$K_2/K_{{2,\mathrm{{ref}}}},\ p_g={pf_list[1]}~\mathrm{{kPa}}$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["pg0"],
            linestyle=linestyles["theta"],
            linewidth=2.0,
            marker=markers["theta"],
            markerfacecolor="white",
            markeredgecolor=colors["pg0"],
            label=rf"$\theta,\ p_g={pf_list[0]}~\mathrm{{kPa}}$",
        ),
        Line2D(
            [0],
            [0],
            color=colors["pg1"],
            linestyle=linestyles["theta"],
            linewidth=2.0,
            marker=markers["theta"],
            markerfacecolor="white",
            markeredgecolor=colors["pg1"],
            label=rf"$\theta,\ p_g={pf_list[1]}~\mathrm{{kPa}}$",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.1,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.10,
        top=0.84,
        wspace=0.30,
        hspace=0.36,
    )

    shift = -0.055

    for i in range(n_rows):
        pos = axes[i, 5].get_position()
        axes[i, 5].set_position([
            pos.x0 + shift,
            pos.y0,
            pos.width,
            pos.height,
        ])

    if add_stream_colorbar and cf_last is not None:
        fig.canvas.draw()

        mesh_ax_pos = axes[0, 4].get_position()

        cbar_width = 0.008
        cbar_pad = 0.035

        cax = fig.add_axes(
            [
                mesh_ax_pos.x0 - cbar_pad - cbar_width,
                mesh_ax_pos.y0,
                cbar_width,
                mesh_ax_pos.height,
            ]
        )

        cbar = fig.colorbar(
            cf_last,
            cax=cax,
            orientation="vertical",
        )

        def sci_fmt(x, pos):
            if abs(x) < 1e-14:
                return "0"
            s = f"{x:.1e}"
            s = s.replace("e-0", "e-")
            s = s.replace("e+0", "e")
            s = s.replace("e+", "e")
            return s

        cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(r"$p_\ell$ (kPa)")

    plt.savefig(save_name, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()

    plt.close()

    print(f"Saved: {save_name}")

def plot_pg_after_stretch_x_summary(
    res_folder="results_multi_pg_after_stretch_x",
    res_basename_prefix="pg-after-Ex",
    r0_list=(0.072, 0.41, 0.5),
    Ex_list=(0.0, 0.1, 0.2),
    probe_list=("gx", "gy"),
    stream_r0=0.41,
    stream_Ex=0.2,
    stream_probe="gx",
    n_stream_states=6,
    slice_start=5,
    eps=1e-12,
    theta_lift_threshold_deg=-85.0,
    stream_density=0.8,
    stream_scale=1.0,
    stream_grid_n=500,
    save_name="plots/pg_after_stretch_x_summary.png",
    show_plot=False,
):
    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

    if set(probe_list) != {"gx", "gy"}:
        raise ValueError("probe_list must contain gx and gy.")

    def fmt_file(x):
        return f"{x:.6g}"

    def fmt_disp(x):
        if abs(x) < 1e-12:
            return "0"
        return f"{x:.3g}"

    def build_basename(r0, Ex, probe):
        return f"{res_folder}/{res_basename_prefix}-r0={fmt_file(r0)}-Ex={fmt_file(Ex)}-{probe}"

    def build_qois_filename(r0, Ex, probe):
        return build_basename(r0, Ex, probe) + "-qois.dat"

    def build_xdmf_filename(r0, Ex, probe):
        return build_basename(r0, Ex, probe) + ".xdmf"

    def build_metadata_filename(r0, Ex, probe):
        return build_basename(r0, Ex, probe) + "-metadata.json"

    def read_phi_from_metadata(r0, Ex, probe="gx"):
        filename = build_metadata_filename(r0, Ex, probe)
        if os.path.exists(filename):
            with open(filename, "r") as f:
                metadata = json.load(f)
            for key in ["mesh_porosity", "porosity", "phi"]:
                if key in metadata:
                    return float(metadata[key])
        return None

    def get_first_existing(qois_vals, names, keys, default=None):
        for key in keys:
            if key in names:
                return np.asarray(get(qois_vals, names, key), dtype=float)
        return default

    def read_probe(filename):
        qois_vals, names = load_qois(filename)

        pg = get_first_existing(
            qois_vals,
            names,
            ["p_f", "p_g", "pf", "pg"],
            default=None,
        )

        if pg is None:
            raise ValueError(f"Gas pressure series not found in {filename}")

        data = {
            "pg": np.asarray(pg, dtype=float),
            "Uxx": np.asarray(get(qois_vals, names, "U_bar_XX"), dtype=float),
            "Uyy": np.asarray(get(qois_vals, names, "U_bar_YY"), dtype=float),
            "Uxy": np.asarray(get(qois_vals, names, "U_bar_XY"), dtype=float),
            "Uyx": np.asarray(get(qois_vals, names, "U_bar_YX"), dtype=float),
            "Qx": np.asarray(get(qois_vals, names, "Q_l_avg_x"), dtype=float),
            "Qy": np.asarray(get(qois_vals, names, "Q_l_avg_y"), dtype=float),
            "gx": np.asarray(get(qois_vals, names, "grad_p_bar_avg_x"), dtype=float),
            "gy": np.asarray(get(qois_vals, names, "grad_p_bar_avg_y"), dtype=float),
        }

        npts = len(data["pg"])
        for key, val in data.items():
            if len(val) != npts:
                raise ValueError(f"Inconsistent length for {key} in {filename}")

        return data

    def continuous_axis_angle_deg(theta_raw):
        theta_raw = np.asarray(theta_raw, dtype=float)
        theta_cont = np.empty_like(theta_raw)

        if len(theta_raw) == 0:
            return theta_cont

        theta_cont[0] = theta_raw[0]

        for i in range(1, len(theta_raw)):
            delta = (theta_raw[i] - theta_cont[i - 1] + 90.0) % 180.0 - 90.0
            theta_cont[i] = theta_cont[i - 1] + delta

        return theta_cont

    def lift_negative_vertical_deg(theta):
        theta = np.asarray(theta, dtype=float).copy()
        theta[theta <= theta_lift_threshold_deg] += 180.0
        return theta

    def principal_quantities(K_list):
        k1 = []
        k2 = []
        theta = []

        for K in K_list:
            Ksym = 0.5 * (K + K.T)

            a = Ksym[0, 0]
            b = Ksym[0, 1]
            c = Ksym[1, 1]

            tr = a + c
            delta = np.sqrt((a - c) ** 2 + 4.0 * b ** 2)

            lam1 = 0.5 * (tr + delta)
            lam2 = 0.5 * (tr - delta)

            angle = 0.5 * np.arctan2(2.0 * b, a - c)

            k1.append(lam1)
            k2.append(lam2)
            theta.append(angle)

        k1 = np.asarray(k1, dtype=float)
        k2 = np.asarray(k2, dtype=float)
        theta = np.rad2deg(np.asarray(theta, dtype=float))
        theta = continuous_axis_angle_deg(theta)
        theta = lift_negative_vertical_deg(theta)

        return k1, k2, theta

    def read_series(r0, Ex):
        file_gx = build_qois_filename(r0, Ex, "gx")
        file_gy = build_qois_filename(r0, Ex, "gy")

        if not os.path.exists(file_gx):
            raise FileNotFoundError(file_gx)
        if not os.path.exists(file_gy):
            raise FileNotFoundError(file_gy)

        data_gx = read_probe(file_gx)
        data_gy = read_probe(file_gy)

        for key in ["pg", "Uxx", "Uyy", "Uxy", "Uyx"]:
            if not np.allclose(data_gx[key], data_gy[key], rtol=1e-5, atol=1e-8):
                print(f"Warning: {key} differs between gx and gy for r0={r0}, Ex={Ex}")

        gx = data_gx["gx"]
        gy = data_gy["gy"]

        Kxx = -data_gx["Qx"] / (gx + eps)
        Kyx = -data_gx["Qy"] / (gx + eps)
        Kxy = -data_gy["Qx"] / (gy + eps)
        Kyy = -data_gy["Qy"] / (gy + eps)

        K_list = []
        for n in range(len(Kxx)):
            K_list.append(
                np.array(
                    [
                        [Kxx[n], Kxy[n]],
                        [Kyx[n], Kyy[n]],
                    ],
                    dtype=float,
                )
            )

        start = max(0, min(slice_start, len(data_gx["pg"]) - 1))

        k1, k2, theta = principal_quantities(K_list)

        k1_ref = k1[start]
        k2_ref = k2[start]

        if abs(k1_ref) < eps or abs(k2_ref) < eps:
            raise ValueError(f"Reference principal permeability too small for r0={r0}, Ex={Ex}")

        return {
            "pg": data_gx["pg"][start:],
            "k1": (k1 / k1_ref)[start:],
            "k2": (k2 / k2_ref)[start:],
            "phi": read_phi_from_metadata(r0, Ex, "gx"),
        }

    def pick_stream_indices(pg_vals, n_pick):
        pg_vals = np.asarray(pg_vals, dtype=float)
        pos = np.where(pg_vals > eps)[0]

        if len(pos) > 0:
            start = max(pos[0] - 1, 0)
        else:
            start = 0

        candidates = np.arange(start, len(pg_vals), dtype=int)

        if len(candidates) == 0:
            return np.array([], dtype=int)

        if len(candidates) <= n_pick:
            return candidates

        idx = np.linspace(candidates[0], candidates[-1], n_pick)
        idx = np.round(idx).astype(int)
        idx = np.unique(idx)

        if len(idx) < n_pick:
            full = list(idx)
            for k in candidates:
                if k not in full:
                    full.append(k)
                if len(full) == n_pick:
                    break
            idx = np.array(sorted(full), dtype=int)

        return idx[:n_pick]

    def build_snapshot_data(xdmf_file, time_idx):
        reader = pv.get_reader(xdmf_file)

        if hasattr(reader, "number_time_points") and reader.number_time_points > 0:
            reader.set_active_time_point(0)
            mesh0 = reader.read()
            reader.set_active_time_point(int(time_idx))
            meshf = reader.read()
        else:
            mesh0 = reader.read()
            meshf = mesh0.copy()

        mesh0 = mesh0.cell_data_to_point_data()
        meshf = meshf.cell_data_to_point_data()

        surf0 = mesh0.extract_surface().triangulate()
        surf = meshf.warp_by_vector("U_tot", factor=stream_scale).extract_surface().triangulate()

        pts0 = surf0.points[:, :2]
        faces0 = surf0.faces.reshape(-1, 4)[:, 1:4]

        pts = surf.points[:, :2]
        faces = surf.faces.reshape(-1, 4)[:, 1:4]

        center0 = np.array(
            [
                0.5 * (pts0[:, 0].min() + pts0[:, 0].max()),
                0.5 * (pts0[:, 1].min() + pts0[:, 1].max()),
            ]
        )
        center = np.array(
            [
                0.5 * (pts[:, 0].min() + pts[:, 0].max()),
                0.5 * (pts[:, 1].min() + pts[:, 1].max()),
            ]
        )
        pts0_shift = pts0 + (center - center0)

        triang0 = mtri.Triangulation(
            pts0_shift[:, 0],
            pts0_shift[:, 1],
            triangles=faces0,
        )

        p = np.asarray(surf.point_data["pl_tot"], dtype=float)
        q = np.asarray(surf.point_data["q_l"][:, :2], dtype=float)

        x = pts[:, 0]
        y = pts[:, 1]
        qx = q[:, 0]
        qy = q[:, 1]

        triang = mtri.Triangulation(x, y, triangles=faces)

        interp_p = mtri.LinearTriInterpolator(triang, p)
        interp_qx = mtri.LinearTriInterpolator(triang, qx)
        interp_qy = mtri.LinearTriInterpolator(triang, qy)

        xi = np.linspace(x.min(), x.max(), stream_grid_n)
        yi = np.linspace(y.min(), y.max(), stream_grid_n)
        X, Y = np.meshgrid(xi, yi)

        P = interp_p(X, Y)
        QX = interp_qx(X, Y)
        QY = interp_qy(X, Y)

        finder = triang.get_trifinder()
        inside = finder(X, Y) != -1

        P_mask = np.ma.getmaskarray(P) | (~inside)
        QX_mask = np.ma.getmaskarray(QX) | (~inside)
        QY_mask = np.ma.getmaskarray(QY) | (~inside)

        P = np.ma.array(P, mask=P_mask)
        QX = np.ma.array(QX, mask=QX_mask)
        QY = np.ma.array(QY, mask=QY_mask)

        xmin = min(pts0_shift[:, 0].min(), x.min())
        xmax = max(pts0_shift[:, 0].max(), x.max())
        ymin = min(pts0_shift[:, 1].min(), y.min())
        ymax = max(pts0_shift[:, 1].max(), y.max())

        return {
            "triang0": triang0,
            "pts0_shift": pts0_shift,
            "X": X,
            "Y": Y,
            "P": P,
            "QX": QX,
            "QY": QY,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }

    def plot_snapshot(ax, snap, xlim, ylim, vmin, vmax):
        ax.tripcolor(
            snap["triang0"],
            np.ones(len(snap["pts0_shift"])),
            shading="gouraud",
            cmap="Greys",
            vmin=0.0,
            vmax=2.0,
            alpha=0.7,
            edgecolors="none",
            zorder=0,
        )

        ax.triplot(
            snap["triang0"],
            color="0.75",
            linewidth=0.35,
            alpha=0.45,
            zorder=1,
        )

        cf = ax.contourf(
            snap["X"],
            snap["Y"],
            snap["P"],
            levels=np.linspace(vmin, vmax, 60),
            cmap="coolwarm",
            alpha=0.65,
            zorder=2,
        )

        sp = ax.streamplot(
            snap["X"][0, :],
            snap["Y"][:, 0],
            snap["QX"],
            snap["QY"],
            density=stream_density,
            color="#66F7FF",
            linewidth=1.1,
            arrowsize=1.1,
            arrowstyle="->",
            minlength=0.02,
            maxlength=10.0,
            integration_direction="both",
        )
        sp.lines.set_zorder(4)
        sp.arrows.set_zorder(5)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.margins(0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        return cf

    series_cache = {}
    phi_vals = {}
    all_k = []

    for r0 in r0_list:
        for Ex in Ex_list:
            try:
                series = read_series(r0, Ex)
                series_cache[(r0, Ex)] = series
                if series["phi"] is not None and r0 not in phi_vals:
                    phi_vals[r0] = series["phi"]
                all_k.extend(series["k1"][np.isfinite(series["k1"])].tolist())
                all_k.extend(series["k2"][np.isfinite(series["k2"])].tolist())
            except FileNotFoundError as e:
                print(f"[WARNING] Missing file for r0={r0}, Ex={Ex}: {e}")
            except Exception as e:
                print(f"[WARNING] Failed reading series for r0={r0}, Ex={Ex}: {e}")

    stream_file = build_xdmf_filename(stream_r0, stream_Ex, stream_probe)
    stream_qois_file = build_qois_filename(stream_r0, stream_Ex, stream_probe)

    stream_axes_data = []
    stream_pg_vals = []
    cf_last = None

    if os.path.exists(stream_file) and os.path.exists(stream_qois_file):
        stream_probe_data = read_probe(stream_qois_file)
        stream_indices = pick_stream_indices(stream_probe_data["pg"], n_stream_states)

        if len(stream_indices) > 0:
            for idx in stream_indices:
                snap = build_snapshot_data(stream_file, idx)
                stream_axes_data.append(snap)
                stream_pg_vals.append(stream_probe_data["pg"][idx])
    else:
        print(f"[WARNING] Missing stream file for r0={stream_r0}, Ex={stream_Ex}")

    if len(stream_axes_data) > 0:
        xmin = min(s["xmin"] for s in stream_axes_data)
        xmax = max(s["xmax"] for s in stream_axes_data)
        ymin = min(s["ymin"] for s in stream_axes_data)
        ymax = max(s["ymax"] for s in stream_axes_data)

        pad_x = 0.04 * (xmax - xmin)
        pad_y = 0.04 * (ymax - ymin)

        common_xlim = (xmin - pad_x, xmax + pad_x)
        common_ylim = (ymin - pad_y, ymax + pad_y)

        p_all = []
        for s in stream_axes_data:
            P = s["P"]

            if np.ma.isMaskedArray(P):
                vals = P.compressed()
            else:
                vals = np.asarray(P, dtype=float).ravel()
                vals = vals[np.isfinite(vals)]

            if len(vals) > 0:
                p_all.append(vals)

        if len(p_all) > 0:
            p_all = np.concatenate(p_all)
            pmin = np.nanmin(p_all)
            pmax = np.nanmax(p_all)
            if not np.isfinite(pmin) or not np.isfinite(pmax) or abs(pmax - pmin) < eps:
                pmin, pmax = -1.0, 1.0
        else:
            pmin, pmax = -1.0, 1.0
    else:
        common_xlim = (0.0, 1.0)
        common_ylim = (0.0, 1.0)
        pmin, pmax = -1.0, 1.0

    ex_colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(Ex_list)))

    fig = plt.figure(figsize=(17.0, 7.8))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.22],
        hspace=0.34,
    )

    gs_row1 = outer[0].subgridspec(1, len(r0_list), wspace=0.28)
    gs_row2 = outer[1].subgridspec(1, max(1, len(stream_axes_data)), wspace=0.14)

    ax_row1 = [fig.add_subplot(gs_row1[0, i]) for i in range(len(r0_list))]
    ax_row2 = [fig.add_subplot(gs_row2[0, i]) for i in range(max(1, len(stream_axes_data)))]

    for i_r0, r0 in enumerate(r0_list):
        axk = ax_row1[i_r0]

        for i_Ex, Ex in enumerate(Ex_list):
            if (r0, Ex) not in series_cache:
                continue

            s = series_cache[(r0, Ex)]
            color = ex_colors[i_Ex]
            pg = s["pg"]
            k1 = s["k1"]
            k2 = s["k2"]

            markevery = max(1, len(pg) // 7)

            axk.plot(
                pg,
                k1,
                color=color,
                linestyle="-",
                linewidth=2.0,
                marker="o",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                markevery=markevery,
                label=r"$K_1$",
            )

            axk.plot(
                pg,
                k2,
                color=color,
                linestyle="--",
                linewidth=2.0,
                marker="s",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                markevery=markevery,
                label=r"$K_2$",
            )

        phi_val = phi_vals.get(r0, None)
        if phi_val is None:
            axk.set_title(rf"$r_0={fmt_disp(r0)}$")
        else:
            axk.set_title(rf"$\tilde{{\Phi}}_{{g0}} = {phi_val:.2f}$")

        axk.set_xlabel(r"$p_g$ (kPa)")

        if i_r0 == 0:
            axk.set_ylabel(r"$K_i/K_{i,p_g=0}$")

        axk.grid(False)
        axk.tick_params(axis="both", labelsize=11)



    for i, ax in enumerate(ax_row2):
        if i < len(stream_axes_data):
            cf_last = plot_snapshot(
                ax,
                stream_axes_data[i],
                common_xlim,
                common_ylim,
                pmin,
                pmax,
            )
        else:
            ax.axis("off")

    ex_handles = [
        Line2D([0], [0], color=ex_colors[i], linewidth=2.0, label=rf"$E_x = {fmt_disp(Ex)}$")
        for i, Ex in enumerate(Ex_list)
    ]

    style_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle="-",
            linewidth=1.8,
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            label=r"$K_1$",
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            linewidth=1.8,
            marker="s",
            markerfacecolor="white",
            markeredgecolor="black",
            label=r"$K_2$",
        ),
    ]

    fig.legend(
        handles=ex_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(ex_handles) + len(style_handles),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )

    plt.tight_layout(rect=(0.08, 0.08, 0.985, 0.91))
    fig.canvas.draw()

    if len(stream_axes_data) > 0 and cf_last is not None:
        first_ax = ax_row2[0]
        last_ax = ax_row2[len(stream_axes_data) - 1]

        first_pos = first_ax.get_position()
        last_pos = last_ax.get_position()

        cbar_width = 0.016
        cbar_pad = 0.065
        cax = fig.add_axes(
            [
                first_pos.x0 - cbar_pad - cbar_width,
                first_pos.y0,
                cbar_width,
                first_pos.height,
            ]
        )

        cbar = fig.colorbar(cf_last, cax=cax, orientation="vertical")
        cbar.set_label(r"$p_\ell$ (kPa)")
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))

        arrow_y = first_pos.y0 - 0.045
        x0 = first_pos.x0
        x1 = last_pos.x1

        arrow = FancyArrowPatch(
            (x0, arrow_y),
            (x1, arrow_y),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.4,
            color="black",
        )
        fig.add_artist(arrow)

        fig.text(
            0.5 * (x0 + x1),
            arrow_y - 0.030,
            r"$p_g$ (kPa)",
            ha="center",
            va="top",
        )

        for ax, pg in zip(ax_row2[:len(stream_axes_data)], stream_pg_vals):
            pos = ax.get_position()
            xc = 0.5 * (pos.x0 + pos.x1)
            fig.text(
                xc,
                arrow_y + 0.008,
                rf"${fmt_disp(pg)}$",
                ha="center",
                va="bottom",
            )

    plt.savefig(save_name, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()

    plt.close()

    print(f"Saved: {save_name}")

def plot_figure9_summary(
    r0_list,
    cases=None,
    loading_text=None,
    res_folder="results_four_loading_cases",
    prefix="Fig_loading",
    pg=0.2,
    r0_to_phi=None,
    slice_start=5,
    final_index=-1,
    eps=1e-12,
    save_name="plots/Figure9_summary.png",
    show_plot=False,
):

    os.makedirs(os.path.dirname(save_name) or ".", exist_ok=True)

    if cases is None:
        cases = [
            {
                "mode": "stretch_x",
                "label": "uniaxial loading",
                "pf": 0.0,
                "use_prediction": True,
            },
            {
                "mode": "volumic",
                "label": "pure volumetric loading",
                "pf": 0.0,
                "use_prediction": True,
            },
            {
                "mode": "simple_shear",
                "label": "simple shear loading",
                "pf": 0.0,
                "use_prediction": True,
            },
            {
                "mode": "gas_pressure",
                "label": "gas pressure loading",
                "pf": pg,
                "use_prediction": True,
            },
        ]

    if loading_text is None:
        loading_text = {
            "uniaxial loading": r"$E_x=0.3$",
            "pure volumetric loading": r"$E_x=E_y=0.3$",
            "simple shear loading": r"$E_{xy}=0.3$",
            "gas pressure loading": rf"$p_g={pg}~\mathrm{{kPa}}$",
        }

    colors = {
        "uniaxial loading": "#0072B2",
        "pure volumetric loading": "#D55E00",
        "simple shear loading": "#009E73",
        "gas pressure loading": "#CC79A7",
    }

    markers = {
        "uniaxial loading": "o",
        "pure volumetric loading": "s",
        "simple shear loading": "^",
        "gas pressure loading": "D",
    }

    def fmt_candidates(x):
        x = float(x)

        vals = [
            str(x),
            f"{x:.6g}",
            f"{x:.3g}",
            f"{x:.1f}",
        ]

        if abs(x) < 1e-14:
            vals += ["0", "0.0"]

        out = []
        for v in vals:
            if v not in out:
                out.append(v)

        return out

    def build_basename(case, r0, probe):
        folder = case.get("res_folder", res_folder)
        case_prefix = case.get("prefix", prefix)
        mode = case["mode"]
        pf = case.get("pf", 0.0)

        first = None

        for r0_str in fmt_candidates(r0):
            for pf_str in fmt_candidates(pf):
                basename = f"{folder}/{case_prefix}-{mode}-r0={r0_str}-pf={pf_str}-{probe}"

                if first is None:
                    first = basename

                if os.path.exists(basename + "-qois.dat"):
                    return basename

        return first

    def read_phi(basename, r0):
        metadata_file = basename + "-metadata.json"

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for key in ["mesh_porosity", "porosity", "phi"]:
                if key in metadata:
                    return float(metadata[key])

        if r0_to_phi is not None:
            return float(r0_to_phi[r0])

        return float(r0)

    def read_probe(filename):
        qois_vals, names = load_qois(filename)

        data = {
            "Uxx": np.asarray(get(qois_vals, names, "U_bar_XX")[slice_start:], dtype=float),
            "Uyy": np.asarray(get(qois_vals, names, "U_bar_YY")[slice_start:], dtype=float),
            "Uxy": np.asarray(get(qois_vals, names, "U_bar_XY")[slice_start:], dtype=float),
            "Uyx": np.asarray(get(qois_vals, names, "U_bar_YX")[slice_start:], dtype=float),
            "Qx": np.asarray(get(qois_vals, names, "Q_l_avg_x")[slice_start:], dtype=float),
            "Qy": np.asarray(get(qois_vals, names, "Q_l_avg_y")[slice_start:], dtype=float),
            "gx": np.asarray(get(qois_vals, names, "grad_p_bar_avg_x")[slice_start:], dtype=float),
            "gy": np.asarray(get(qois_vals, names, "grad_p_bar_avg_y")[slice_start:], dtype=float),
        }

        npts = len(data["Uxx"])

        for key, val in data.items():
            if len(val) != npts:
                raise ValueError(f"Inconsistent length for {key} in {filename}")

        return data

    def read_K_and_F(case, r0):
        basename_gx = build_basename(case, r0, "gx")
        basename_gy = build_basename(case, r0, "gy")

        file_gx = basename_gx + "-qois.dat"
        file_gy = basename_gy + "-qois.dat"

        if not os.path.exists(file_gx):
            raise FileNotFoundError(file_gx)

        if not os.path.exists(file_gy):
            raise FileNotFoundError(file_gy)

        data_gx = read_probe(file_gx)
        data_gy = read_probe(file_gy)

        for key in ["Uxx", "Uyy", "Uxy", "Uyx"]:
            if not np.allclose(data_gx[key], data_gy[key], rtol=1e-5, atol=1e-8):
                print(
                    f"[WARNING] {key} differs between gx and gy "
                    f"for mode={case['mode']}, r0={r0}"
                )

        gx = data_gx["gx"]
        gy = data_gy["gy"]

        Kxx = -data_gx["Qx"] / (gx + eps)
        Kyx = -data_gx["Qy"] / (gx + eps)
        Kxy = -data_gy["Qx"] / (gy + eps)
        Kyy = -data_gy["Qy"] / (gy + eps)

        Uxx = 0.5 * (data_gx["Uxx"] + data_gy["Uxx"])
        Uyy = 0.5 * (data_gx["Uyy"] + data_gy["Uyy"])
        Uxy = 0.5 * (data_gx["Uxy"] + data_gy["Uxy"])
        Uyx = 0.5 * (data_gx["Uyx"] + data_gy["Uyx"])

        K_list = []
        F_list = []

        for n in range(len(Kxx)):
            K_list.append(
                np.array(
                    [
                        [Kxx[n], Kxy[n]],
                        [Kyx[n], Kyy[n]],
                    ],
                    dtype=float,
                )
            )

            F_list.append(
                np.array(
                    [
                        [1.0 + Uxx[n], Uxy[n]],
                        [Uyx[n], 1.0 + Uyy[n]],
                    ],
                    dtype=float,
                )
            )

        phi = read_phi(basename_gx, r0)

        return K_list, F_list, phi

    def compute_indicators(K0, Kf, Ff, use_prediction):
        C_K = np.trace(Kf) / (np.trace(K0) + eps)

        Ksym = 0.5 * (Kf + Kf.T)
        eigvals = np.linalg.eigvalsh(Ksym)

        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)

        if eig_min <= eps:
            A_K = np.nan
        else:
            A_K = eig_max / eig_min

        if use_prediction:
            J = float(np.linalg.det(Ff))

            if abs(J) < eps:
                E_K = np.nan
            else:
                Finv = np.linalg.inv(Ff)
                K_pred = J * (Finv @ K0 @ Finv.T)

                E_K = np.linalg.norm(Kf - K_pred, ord="fro") / (
                    np.linalg.norm(Kf, ord="fro") + eps
                )
        else:
            E_K = np.nan

        return C_K, A_K, E_K

    rows = []

    for case in cases:
        label = case["label"]
        use_prediction = case.get("use_prediction", True)

        for r0 in r0_list:
            try:
                K_list, F_list, phi = read_K_and_F(case, r0)

                K0 = K_list[0]
                Kf = K_list[final_index]
                Ff = F_list[final_index]

                C_K, A_K, E_K = compute_indicators(
                    K0=K0,
                    Kf=Kf,
                    Ff=Ff,
                    use_prediction=use_prediction,
                )

                rows.append(
                    {
                        "label": label,
                        "mode": case["mode"],
                        "r0": r0,
                        "phi": phi,
                        "C_K": C_K,
                        "A_K": A_K,
                        "E_K": E_K,
                    }
                )

                print(
                    f"{label}, r0={r0}, phi={phi:.6f}, "
                    f"C_K={C_K:.6f}, A_K={A_K:.6f}, E_K={E_K:.6f}"
                )

            except FileNotFoundError as e:
                print(f"[WARNING] Missing file: {e}")

            except Exception as e:
                print(f"[WARNING] Failed for {label}, r0={r0}: {e}")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    plot_settings = [
        ("C_K", r"$C_K=\mathrm{tr}(\mathbf{K})/\mathrm{tr}(\mathbf{K}_0)$"),
        ("A_K", r"$A_K=K_{\max}/K_{\min}$"),
        ("E_K", r"$E_K=\|\mathbf{K}-\mathbf{K}^{kin}\|_F/\|\mathbf{K}\|_F$"),
    ]

    for ax, (key, ylabel) in zip(axes, plot_settings):
        for case in cases:
            label = case["label"]
            case_rows = [row for row in rows if row["label"] == label]

            if not case_rows:
                continue

            case_rows = sorted(case_rows, key=lambda row: row["phi"])

            x = np.array([row["phi"] for row in case_rows], dtype=float)
            y = np.array([row[key] for row in case_rows], dtype=float)

            valid = np.isfinite(y)

            if not np.any(valid):
                continue

            ax.plot(
                x[valid],
                y[valid],
                color=colors.get(label, "black"),
                marker=markers.get(label, "o"),
                markersize=5.5,
                linewidth=2.0,
                markerfacecolor="white",
                markeredgewidth=1.2,
                label=label,
            )

        ax.set_xlabel(r"$\tilde{\Phi}_{g0}$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(False)

    axes[0].axhline(1.0, color="0.5", linewidth=1.0, linestyle="--")
    axes[1].axhline(1.0, color="0.5", linewidth=1.0, linestyle="--")
    axes[2].axhline(0.0, color="0.5", linewidth=1.0, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()

    legend_labels = []

    for label in labels:
        if label in loading_text:
            legend_labels.append(f"{label}, {loading_text[label]}")
        else:
            legend_labels.append(label)

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.51, 0.91),
    )
        

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.73])
    plt.savefig(save_name, bbox_inches="tight", dpi=300)

    if show_plot:
        plt.show()

    plt.close()

    print(f"Saved: {save_name}")

    return rows



def plot_q_vs_gradp(
    res_folder="results_gradp_linearity",
    res_basename_prefix="gradp-linearity",
    r0_list=(0.2,),
    pf=0.0,
    slice_start=0,
    eps=1e-14,
    save_name="plots/q_vs_gradp.png",
    show_plot=False,
    xdmf_folder=None,
    xdmf_basename_prefix=None,
    stream_density=1.0,
    stream_scale=1.0,
    stream_grid_n=400,
    add_stream_colorbar=True,
):
    os.makedirs(
        os.path.dirname(save_name) or ".",
        exist_ok=True,
    )

    if xdmf_folder is None:
        xdmf_folder = res_folder

    if xdmf_basename_prefix is None:
        xdmf_basename_prefix = res_basename_prefix

    def build_basename(folder, prefix, r0, probe):
        return (
            f"{folder}/"
            f"{prefix}"
            f"-r0={r0}"
            f"-pf={pf}"
            f"-{probe}"
        )

    def build_qois_filename(r0, probe):
        return (
            build_basename(
                res_folder,
                res_basename_prefix,
                r0,
                probe,
            )
            + "-qois.dat"
        )

    def build_xdmf_filename(r0, probe):
        return (
            build_basename(
                xdmf_folder,
                xdmf_basename_prefix,
                r0,
                probe,
            )
            + ".xdmf"
        )

    def build_metadata_filename(r0, probe):
        return (
            build_basename(
                res_folder,
                res_basename_prefix,
                r0,
                probe,
            )
            + "-metadata.json"
        )

    def read_phi_from_metadata(r0, probe="gx"):
        metadata_file = build_metadata_filename(
            r0,
            probe,
        )

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            for key in [
                "mesh_porosity",
                "porosity",
                "phi",
            ]:
                if key in metadata:
                    return float(metadata[key])

        return None

    def read_probe(r0, probe):
        filename = build_qois_filename(
            r0,
            probe,
        )

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        qois_vals, names = load_qois(filename)

        data = {
            "gx": np.asarray(
                get(
                    qois_vals,
                    names,
                    "grad_p_bar_avg_x",
                )[slice_start:],
                dtype=float,
            ),
            "gy": np.asarray(
                get(
                    qois_vals,
                    names,
                    "grad_p_bar_avg_y",
                )[slice_start:],
                dtype=float,
            ),
            "Qx": np.asarray(
                get(
                    qois_vals,
                    names,
                    "Q_l_avg_x",
                )[slice_start:],
                dtype=float,
            ),
            "Qy": np.asarray(
                get(
                    qois_vals,
                    names,
                    "Q_l_avg_y",
                )[slice_start:],
                dtype=float,
            ),
        }

        npts = len(data["gx"])

        for key, values in data.items():
            if len(values) != npts:
                raise ValueError(
                    f"Inconsistent length for {key} "
                    f"in {filename}"
                )

        return data

    def make_valid_triangulation(
        points,
        faces,
        area_tol=1e-14,
    ):
        points = np.asarray(
            points,
            dtype=float,
        )

        faces = np.asarray(
            faces,
            dtype=int,
        )

        finite_points = np.all(
            np.isfinite(points),
            axis=1,
        )

        valid_faces = np.all(
            finite_points[faces],
            axis=1,
        )

        faces = faces[valid_faces]

        distinct_vertices = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 2] != faces[:, 0])
        )

        faces = faces[distinct_vertices]

        p0 = points[faces[:, 0]]
        p1 = points[faces[:, 1]]
        p2 = points[faces[:, 2]]

        signed_area = 0.5 * (
            (p1[:, 0] - p0[:, 0])
            * (p2[:, 1] - p0[:, 1])
            - (p1[:, 1] - p0[:, 1])
            * (p2[:, 0] - p0[:, 0])
        )

        lx = (
            np.nanmax(points[:, 0])
            - np.nanmin(points[:, 0])
        )

        ly = (
            np.nanmax(points[:, 1])
            - np.nanmin(points[:, 1])
        )

        scale = max(
            lx * ly,
            1.0,
        )

        valid_area = (
            np.isfinite(signed_area)
            & (
                np.abs(signed_area)
                > area_tol * scale
            )
        )

        faces = faces[valid_area]
        signed_area = signed_area[valid_area]

        inverted = signed_area < 0.0

        faces[inverted] = (
            faces[inverted][:, [0, 2, 1]]
        )

        if len(faces) == 0:
            raise ValueError(
                "No valid triangles remain"
            )

        return mtri.Triangulation(
            points[:, 0],
            points[:, 1],
            triangles=faces,
        )

    def extract_triangular_faces(surface):
        faces_raw = np.asarray(
            surface.faces,
            dtype=int,
        )

        if len(faces_raw) % 4 != 0:
            raise ValueError(
                "Unexpected PyVista face array"
            )

        faces_all = faces_raw.reshape(
            -1,
            4,
        )

        if not np.all(
            faces_all[:, 0] == 3
        ):
            raise ValueError(
                "Non-triangular faces found"
            )

        return faces_all[:, 1:4]

    def read_stream_data(xdmf_file):
        if not os.path.exists(xdmf_file):
            return None

        reader = pv.get_reader(
            xdmf_file
        )

        if (
            hasattr(
                reader,
                "number_time_points",
            )
            and reader.number_time_points > 0
        ):
            reader.set_active_time_point(0)
            mesh0 = reader.read()

            reader.set_active_time_point(
                reader.number_time_points - 1
            )
            meshf = reader.read()
        else:
            mesh0 = reader.read()
            meshf = mesh0.copy()

        mesh0 = (
            mesh0
            .cell_data_to_point_data()
        )

        meshf = (
            meshf
            .cell_data_to_point_data()
        )

        surf0 = (
            mesh0
            .extract_surface()
            .triangulate()
            .clean(
                tolerance=1e-10,
                absolute=False,
            )
        )

        warped = meshf.warp_by_vector(
            "U_tot",
            factor=stream_scale,
        )

        surf = (
            warped
            .extract_surface()
            .triangulate()
            .clean(
                tolerance=1e-10,
                absolute=False,
            )
        )

        pts0 = np.asarray(
            surf0.points[:, :2],
            dtype=float,
        )

        pts = np.asarray(
            surf.points[:, :2],
            dtype=float,
        )

        faces0 = extract_triangular_faces(
            surf0
        )

        faces = extract_triangular_faces(
            surf
        )

        center0 = np.array(
            [
                0.5
                * (
                    pts0[:, 0].min()
                    + pts0[:, 0].max()
                ),
                0.5
                * (
                    pts0[:, 1].min()
                    + pts0[:, 1].max()
                ),
            ]
        )

        center = np.array(
            [
                0.5
                * (
                    pts[:, 0].min()
                    + pts[:, 0].max()
                ),
                0.5
                * (
                    pts[:, 1].min()
                    + pts[:, 1].max()
                ),
            ]
        )

        pts0_shift = (
            pts0
            + center
            - center0
        )

        triang0 = make_valid_triangulation(
            pts0_shift,
            faces0,
        )

        triang = make_valid_triangulation(
            pts,
            faces,
        )

        p = np.asarray(
            surf.point_data["pl_tot"],
            dtype=float,
        )

        q = np.asarray(
            surf.point_data["q_l"][:, :2],
            dtype=float,
        )

        qx = q[:, 0]
        qy = q[:, 1]

        interp_p = (
            mtri.LinearTriInterpolator(
                triang,
                p,
            )
        )

        interp_qx = (
            mtri.LinearTriInterpolator(
                triang,
                qx,
            )
        )

        interp_qy = (
            mtri.LinearTriInterpolator(
                triang,
                qy,
            )
        )

        xi = np.linspace(
            pts[:, 0].min(),
            pts[:, 0].max(),
            stream_grid_n,
        )

        yi = np.linspace(
            pts[:, 1].min(),
            pts[:, 1].max(),
            stream_grid_n,
        )

        X, Y = np.meshgrid(
            xi,
            yi,
        )

        P = interp_p(
            X,
            Y,
        )

        QX = interp_qx(
            X,
            Y,
        )

        QY = interp_qy(
            X,
            Y,
        )

        try:
            finder = (
                triang.get_trifinder()
            )

            inside = (
                finder(X, Y) != -1
            )
        except Exception:
            inside = ~np.ma.getmaskarray(P)

        P_mask = (
            np.ma.getmaskarray(P)
            | (~inside)
        )

        QX_mask = (
            np.ma.getmaskarray(QX)
            | (~inside)
        )

        QY_mask = (
            np.ma.getmaskarray(QY)
            | (~inside)
        )

        P = np.ma.array(
            P,
            mask=P_mask,
        )

        QX = np.ma.array(
            QX,
            mask=QX_mask,
        )

        QY = np.ma.array(
            QY,
            mask=QY_mask,
        )

        xmin = min(
            pts0_shift[:, 0].min(),
            pts[:, 0].min(),
        )

        xmax = max(
            pts0_shift[:, 0].max(),
            pts[:, 0].max(),
        )

        ymin = min(
            pts0_shift[:, 1].min(),
            pts[:, 1].min(),
        )

        ymax = max(
            pts0_shift[:, 1].max(),
            pts[:, 1].max(),
        )

        return {
            "triang0": triang0,
            "pts0_shift": pts0_shift,
            "X": X,
            "Y": Y,
            "P": P,
            "QX": QX,
            "QY": QY,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }

    def plot_stream_subplot(
        ax,
        stream_data,
        p_levels,
        xlim,
        ylim,
    ):
        if stream_data is None:
            ax.text(
                0.5,
                0.5,
                "missing xdmf",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            return None

        ax.tripcolor(
            stream_data["triang0"],
            np.ones(
                len(
                    stream_data[
                        "pts0_shift"
                    ]
                )
            ),
            shading="gouraud",
            cmap="Greys",
            vmin=0.0,
            vmax=2.0,
            alpha=0.72,
            edgecolors="none",
            zorder=0,
        )

        ax.triplot(
            stream_data["triang0"],
            color="0.72",
            linewidth=0.30,
            alpha=0.40,
            zorder=1,
        )

        cf = ax.contourf(
            stream_data["X"],
            stream_data["Y"],
            stream_data["P"],
            levels=p_levels,
            cmap="coolwarm",
            alpha=0.65,
            zorder=2,
        )

        stream_kwargs = {
            "density": stream_density,
            "color": "#66F7FF",
            "linewidth": 1.15,
            "arrowsize": 1.15,
            "arrowstyle": "->",
            "minlength": 0.02,
            "maxlength": 10.0,
            "integration_direction": "both",
        }

        try:
            sp = ax.streamplot(
                stream_data["X"][0, :],
                stream_data["Y"][:, 0],
                stream_data["QX"],
                stream_data["QY"],
                broken_streamlines=False,
                **stream_kwargs,
            )
        except TypeError:
            sp = ax.streamplot(
                stream_data["X"][0, :],
                stream_data["Y"][:, 0],
                stream_data["QX"],
                stream_data["QY"],
                **stream_kwargs,
            )

        sp.lines.set_zorder(4)
        sp.arrows.set_zorder(5)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.set_anchor("C")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.margins(0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        return cf

    stream_cache = {}
    phi_values = []
    pressure_values = []

    for r0 in r0_list:
        phi_val = read_phi_from_metadata(
            r0,
            probe="gx",
        )

        phi_values.append(
            phi_val
        )

        for probe in (
            "gx",
            "gy",
        ):
            xdmf_file = (
                build_xdmf_filename(
                    r0,
                    probe,
                )
            )

            try:
                stream_data = (
                    read_stream_data(
                        xdmf_file
                    )
                )
            except Exception as error:
                print(
                    f"[WARNING] Failed to read "
                    f"{xdmf_file}: {error}"
                )

                stream_data = None

            stream_cache[
                (r0, probe)
            ] = stream_data

            if stream_data is not None:
                P = stream_data["P"]

                if np.ma.isMaskedArray(P):
                    values = P.compressed()
                else:
                    values = np.asarray(
                        P,
                        dtype=float,
                    ).ravel()

                    values = values[
                        np.isfinite(values)
                    ]

                if len(values) > 0:
                    pressure_values.append(
                        values
                    )

    if pressure_values:
        pressure_values = np.concatenate(
            pressure_values
        )

        pmin = np.nanmin(
            pressure_values
        )

        pmax = np.nanmax(
            pressure_values
        )

        if (
            not np.isfinite(pmin)
            or not np.isfinite(pmax)
            or abs(pmax - pmin) < eps
        ):
            pmin = -1.0
            pmax = 1.0
    else:
        pmin = -1.0
        pmax = 1.0

    p_levels = np.linspace(
        pmin,
        pmax,
        61,
    )

    all_stream_data = [
        data
        for data in stream_cache.values()
        if data is not None
    ]

    if all_stream_data:
        xmin = min(
            data["xmin"]
            for data in all_stream_data
        )

        xmax = max(
            data["xmax"]
            for data in all_stream_data
        )

        ymin = min(
            data["ymin"]
            for data in all_stream_data
        )

        ymax = max(
            data["ymax"]
            for data in all_stream_data
        )

        pad_x = 0.05 * (
            xmax - xmin
        )

        pad_y = 0.05 * (
            ymax - ymin
        )

        common_xlim = (
            xmin - pad_x,
            xmax + pad_x,
        )

        common_ylim = (
            ymin - pad_y,
            ymax + pad_y,
        )
    else:
        common_xlim = (
            0.0,
            1.0,
        )

        common_ylim = (
            0.0,
            1.0,
        )

    n_groups = len(r0_list)
    n_cols = 2 * n_groups

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(
            max(
                5.4 * n_groups,
                11.0,
            ),
            9.2,
        ),
        squeeze=False,
        gridspec_kw={
            "height_ratios": [
                1.0,
                1.45,
            ],
            "wspace": 0.38,
            "hspace": 0.38,
        },
    )

    results = {}
    cf_last = None

    for i_r0, r0 in enumerate(
        r0_list
    ):
        col_gx = 2 * i_r0
        col_gy = col_gx + 1

        ax_qx = axes[
            0,
            col_gx,
        ]

        ax_qy = axes[
            0,
            col_gy,
        ]

        ax_stream_gx = axes[
            1,
            col_gx,
        ]

        ax_stream_gy = axes[
            1,
            col_gy,
        ]

        data_gx = read_probe(
            r0,
            "gx",
        )

        data_gy = read_probe(
            r0,
            "gy",
        )

        gx = data_gx["gx"]
        Qx_gx = data_gx["Qx"]

        gy = data_gy["gy"]
        Qy_gy = data_gy["Qy"]

        ax_qx.plot(
            gx,
            Qx_gx,
            linestyle="-",
            linewidth=2.0,
            color="#0072B2",
        )

        ax_qx.axhline(
            0.0,
            color="0.65",
            linewidth=1.0,
            linestyle=":",
        )

        ax_qx.set_xlabel(
            r"$\nabla_0"
            r"\tilde p_{\ell,x}$",
            fontsize=14,
        )

        if i_r0 == 0:
            ax_qx.set_ylabel(
                r"$\tilde{Q}_{\ell,x}$",
                fontsize=14,
            )



        ax_qx.tick_params(
            axis="both",
            labelsize=11,
        )

        ax_qx.grid(False)

        ax_qy.plot(
            gy,
            Qy_gy,
            linestyle="-",
            linewidth=2.0,
            color="#D55E00",
        )

        ax_qy.axhline(
            0.0,
            color="0.65",
            linewidth=1.0,
            linestyle=":",
        )

        ax_qy.set_xlabel(
            r"$\nabla_0"
            r"\tilde p_{\ell,y}$",
            fontsize=14,
        )

        ax_qy.set_ylabel(
            r"$\tilde{Q}_{\ell,y}$",
            fontsize=14,
        )



        ax_qy.tick_params(
            axis="both",
            labelsize=11,
        )

        ax_qy.grid(False)

        stream_gx = stream_cache[
            (r0, "gx")
        ]

        stream_gy = stream_cache[
            (r0, "gy")
        ]

        cf_gx = plot_stream_subplot(
            ax_stream_gx,
            stream_gx,
            p_levels,
            common_xlim,
            common_ylim,
        )

        cf_gy = plot_stream_subplot(
            ax_stream_gy,
            stream_gy,
            p_levels,
            common_xlim,
            common_ylim,
        )

        if cf_gx is not None:
            cf_last = cf_gx

        if cf_gy is not None:
            cf_last = cf_gy

        phi_val = phi_values[i_r0]

        if phi_val is not None:
            group_label = (
                rf"$\tilde{{\Phi}}_{{g0}}"
                rf"={phi_val:.2f}$"
            )
        else:
            group_label = (
                rf"$r_0={r0}$"
            )

        results[r0] = {
            "phi": phi_val,
            "gx": gx,
            "gy": gy,
            "Qx_gx": Qx_gx,
            "Qy_gy": Qy_gy,
        }

        print(f"r0 = {r0}")
        print(f"phi = {phi_val}")

    fig.subplots_adjust(
        left=0.115,
        right=0.985,
        bottom=0.075,
        top=0.86,
        wspace=0.38,
        hspace=0.40,
    )

    fig.canvas.draw()

    for i_r0, r0 in enumerate(
        r0_list
    ):
        ax_left = axes[
            0,
            2 * i_r0,
        ]

        ax_right = axes[
            0,
            2 * i_r0 + 1,
        ]

        left_pos = ax_left.get_position()
        right_pos = ax_right.get_position()

        group_center = 0.5 * (
            left_pos.x0
            + right_pos.x1
        )

        phi_val = phi_values[i_r0]

        if phi_val is not None:
            text = (
                rf"$\tilde{{\Phi}}_{{g0}}"
                rf"={phi_val:.2f}$"
            )
        else:
            text = rf"$r_0={r0}$"

        fig.text(
            group_center,
            left_pos.y1 + 0.035,
            text,
            ha="center",
            va="bottom",
            fontsize=17,
        )

    stream_axes = axes[
        1,
        :,
    ]

    stream_positions = [
        ax.get_position()
        for ax in stream_axes
    ]

    if (
        add_stream_colorbar
        and cf_last is not None
        and stream_positions
    ):
        left = min(
            pos.x0
            for pos in stream_positions
        )

        bottom = min(
            pos.y0
            for pos in stream_positions
        )

        top = max(
            pos.y1
            for pos in stream_positions
        )

        cbar_width = 0.011
        cbar_pad = 0.070

        cax = fig.add_axes(
            [
                left
                - cbar_pad
                - cbar_width,
                bottom,
                cbar_width,
                top - bottom,
            ]
        )

        cbar = fig.colorbar(
            cf_last,
            cax=cax,
            orientation="vertical",
        )

        def sci_fmt(value, position):
            if abs(value) < 1e-14:
                return "0"

            text = f"{value:.1e}"
            text = text.replace(
                "e-0",
                "e-",
            )
            text = text.replace(
                "e+0",
                "e",
            )
            text = text.replace(
                "e+",
                "e",
            )

            return text

        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(sci_fmt)
        )

        cbar.ax.tick_params(
            labelsize=10,
        )

        cbar.set_label(
            r"$p_\ell$ (kPa)",
            fontsize=14,
            labelpad=8,
        )

    plt.savefig(
        save_name,
        bbox_inches="tight",
        dpi=300,
    )

    if show_plot:
        plt.show()

    plt.close()

    print(f"Saved: {save_name}")

    return results

