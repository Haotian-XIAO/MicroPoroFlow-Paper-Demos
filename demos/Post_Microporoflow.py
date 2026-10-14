"""Post-processing for Fig. 11 and for the numbers quoted in the text (Fig3, Fig10, Fig11-kappa0,
Fig11-refined and Fig11 notebooks).

Only numpy, scipy, matplotlib and pandas are needed (no FEniCS). This module does not import
Plot_Microporoflow, which changes the global matplotlib font sizes on import: Fig. 11 is drawn with the
default matplotlib settings.

Conventions (all quantities are read from the *-qois.dat files written by dolfin_mech):
  K~      pulled-back (referential) permeability, K_ij = -Q_i/g_j (gx probe -> first column, gy probe ->
          second column), in units of k_m = 1e-6 for the loading cases;
  k~      = J^-1 F K~ F^T, the spatial permeability; y = tr(k~)/(2 k_m);
  phi_g   = 1 - vs/v, the current gas fraction;
  h(phi)  = (1 - phi)/(1 + phi), the 2D Hashin-Shtrikman upper bound for tr(k~)/(2 k_m);
  kappa0  = permeability of the undeformed hexagonal-pore RVEs, PCHIP in the mesh porosity
            (49 RVEs, r0 = 0.02 ... 0.50 in steps of 0.01);
  only frames t >= 0.01 (end of the probe ramp, unloaded reference state) are used as K samples.
"""
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

from Quiet_Microporoflow import display_path

T_REF = 0.01
K_M = 1e-6
EPS = 1e-12
MINUS = "−"
R0_LIST = (0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.41, 0.45)
MODES = ("stretch_x", "simple_shear", "volumic", "gas_pressure")
OMEGA0 = np.sqrt(3.0)


# ============================================================================================ helpers

def load(path):
    with open(path) as f:
        names = f.readline().split()[1:]
    return names, np.loadtxt(path, ndmin=2)


def h(phi):
    return (1.0 - phi) / (1.0 + phi)


def pf_tag(mode):
    return "0.5" if mode == "gas_pressure" else "0"


def fig10_basename(fig10_dir, mode, r0):
    return Path(fig10_dir) / f"Fig10-{mode}-r0={r0:.6g}-pf={pf_tag(mode)}"


def require_files(groups):
    """groups: {producer notebook: [paths]}. Raise FileNotFoundError listing every missing file by producer."""
    missing = {nb: [p for p in paths if not Path(p).exists()] for nb, paths in groups.items()}
    missing = {nb: ps for nb, ps in missing.items() if ps}
    if missing:
        lines = ["Missing input files. Run the following notebook(s) first (with their do_* switches on):"]
        for nb, ps in missing.items():
            lines.append(f"  {nb}: {len(ps)} missing file(s), e.g. {display_path(ps[0])}")
        raise FileNotFoundError("\n".join(lines))
    n = sum(len(ps) for ps in groups.values())
    print(f"All {n} input files found ({', '.join(groups)}).")


def K_series(base, kl):
    """gx and gy probes: full K~ (n, 2, 2), F, phi_g."""
    ng, dg = load(f"{base}-gx-qois.dat")
    ny, dy = load(f"{base}-gy-qois.dat")
    c = lambda n, d, k: d[:, n.index(k)]
    gx, gy = c(ng, dg, "grad_p_bar_avg_x"), c(ny, dy, "grad_p_bar_avg_y")
    K = np.empty((len(gx), 2, 2))
    K[:, 0, 0] = -c(ng, dg, "Q_l_avg_x") / (gx + EPS)
    K[:, 1, 0] = -c(ng, dg, "Q_l_avg_y") / (gx + EPS)
    K[:, 0, 1] = -c(ny, dy, "Q_l_avg_x") / (gy + EPS)
    K[:, 1, 1] = -c(ny, dy, "Q_l_avg_y") / (gy + EPS)
    F = np.empty_like(K)
    F[:, 0, 0] = 1 + c(ng, dg, "U_bar_XX")
    F[:, 0, 1] = c(ng, dg, "U_bar_XY")
    F[:, 1, 0] = c(ng, dg, "U_bar_YX")
    F[:, 1, 1] = 1 + c(ng, dg, "U_bar_YY")
    v, vs = c(ng, dg, "v"), c(ng, dg, "vs")
    return dict(t=c(ng, dg, "t"), K=K / kl, F=F, phi_g=1.0 - vs / v)


def spatial_k(s):
    return np.array([np.linalg.det(F) ** -1 * F @ K @ F.T for F, K in zip(s["F"], s["K"])])


def spatial_trace(s):
    k = spatial_k(s)
    return 0.5 * (k[:, 0, 0] + k[:, 1, 1])


def anisotropy(s):
    """a_D = (lambda_1 - lambda_2)/(lambda_1 + lambda_2), eigenvalues of the symmetric part of k~."""
    k = spatial_k(s)
    lam = np.linalg.eigvalsh(0.5 * (k + np.swapaxes(k, 1, 2)))
    return (lam[:, 1] - lam[:, 0]) / (lam[:, 1] + lam[:, 0])


def gx_series(base, kl):
    """gx probe only (isotropic symmetric loadings): K~ = K_xx I, tr(k~)/2 = K_xx tr(F F^T)/(2J)."""
    n, d = load(f"{base}-gx-qois.dat")
    c = lambda k: d[:, n.index(k)]
    Kxx = -c("Q_l_avg_x") / (c("grad_p_bar_avg_x") + EPS) / kl
    F = np.empty((len(Kxx), 2, 2))
    F[:, 0, 0] = 1 + c("U_bar_XX"); F[:, 0, 1] = c("U_bar_XY")
    F[:, 1, 0] = c("U_bar_YX"); F[:, 1, 1] = 1 + c("U_bar_YY")
    J = np.linalg.det(F)
    tr_half = Kxx * np.trace(F @ np.swapaxes(F, 1, 2), axis1=1, axis2=2) / (2 * J)
    return dict(t=c("t"), Kxx=Kxx, y=tr_half, phi_g=1.0 - c("vs") / c("v"), J=J, p_f=c("p_f"))


def valid(t, *arrays):
    ok = t >= T_REF - 1e-9
    return [a[ok] for a in arrays]


# ============================================================================================ kappa0

def kappa0_r0_list():
    return np.round(np.arange(0.02, 0.5001, 0.01), 2)


def kappa0_input_files(lin_dir, probe_only_gx, gy_r0=()):
    files = []
    for r0 in kappa0_r0_list():
        base = Path(lin_dir) / f"linear-hex-r0={r0:.6g}"
        files += [f"{base}-gx-qois.dat", f"{base}-gx-metadata.json"]
        if not probe_only_gx or r0 in gy_r0:
            files.append(f"{base}-gy-qois.dat")
    return files


def build_kappa0(lin_dir, probe_only_gx):
    """Table (phi = mesh porosity, kappa0/k_m, r0), sorted in phi, from the 49 one-step linear runs."""
    pts = []
    for r0 in kappa0_r0_list():
        base = Path(lin_dir) / f"linear-hex-r0={r0:.6g}"
        if probe_only_gx:
            s = gx_series(str(base), 1.0)
            k0 = s["Kxx"][-1]
        else:
            s = K_series(str(base), 1.0)
            k0 = 0.5 * np.trace(s["K"][-1])
        with open(f"{base}-gx-metadata.json") as f:
            meta = json.load(f)
        pts.append((meta["mesh_porosity"], k0, r0))
    pts.sort()
    return np.array(pts)


def kappa0_check(lin_def, lin_ref, r0_list=R0_LIST):
    """Print kappa0 of both meshes at the Fig. 10 r0, their maximum relative difference, monotonicity, and
    the isotropy of the refined undeformed RVEs (gy runs at r0 = 0.07 and 0.30)."""
    tab_def, tab_ref = build_kappa0(lin_def, False), build_kappa0(lin_ref, True)
    kappa0, kappa0_r = PchipInterpolator(tab_def[:, 0], tab_def[:, 1]), PchipInterpolator(tab_ref[:, 0], tab_ref[:, 1])
    print("| r0 | phi | kappa0 default (l=0.03) | kappa0 refined | (refined - default)/default |")
    print("|---|---|---|---|---|")
    for r0 in r0_list:
        i = int(np.argmin(np.abs(tab_ref[:, 2] - r0)))
        print(f"| {r0:g} | {tab_ref[i, 0]:.4f} | {tab_def[i, 1]:.5f} | {tab_ref[i, 1]:.5f} | "
              f"{100 * (tab_ref[i, 1] / tab_def[i, 1] - 1):+.3f}% |")
    pgrid = np.linspace(tab_def[0, 0], tab_def[-1, 0], 2000)
    dd = kappa0_r(pgrid) / kappa0(pgrid) - 1
    print(f"\nmax |refined/default - 1| over all phi: {np.abs(dd).max():.2e} at phi = {pgrid[np.abs(dd).argmax()]:.3f}")
    for name, tab in (("default", tab_def), ("refined", tab_ref)):
        print(f"{name}: {len(tab)} points, phi in [{tab[0, 0]:.4f}, {tab[-1, 0]:.4f}], "
              f"strictly decreasing: {bool(np.all(np.diff(tab[:, 1]) < 0))}")
    for r0, v in isotropy_undeformed(lin_ref).items():
        print(f"refined undeformed RVE r0 = {r0:g}: K_yy(gy)/K_xx(gx) - 1 = {v:+.2e}")


def isotropy_undeformed(lin_ref):
    iso_lin = {}
    for r0 in (0.07, 0.30):
        sx = gx_series(str(Path(lin_ref) / f"linear-hex-r0={r0:.6g}"), 1.0)
        ny, dy = load(str(Path(lin_ref) / f"linear-hex-r0={r0:.6g}-gy-qois.dat"))
        kyy = -dy[-1, ny.index("Q_l_avg_y")] / dy[-1, ny.index("grad_p_bar_avg_y")]
        iso_lin[r0] = kyy / sx["Kxx"][-1] - 1.0
    return iso_lin


# ============================================================================================ Fig. 11 data

REF_GXGY = [("volumic", 0.07), ("volumic", 0.10), ("volumic", 0.15), ("volumic", 0.20), ("gas_pressure", 0.07)]


def fig11_input_files(fig9_dir, fig10_dir, lin_def, lin_ref, ref_dir, r0_list=R0_LIST):
    """Input files of Fig11.ipynb, grouped by the notebook that produces them."""
    fig10 = [f"{fig10_basename(fig10_dir, mode, r0)}-{p}-qois.dat" for mode in MODES for r0 in r0_list for p in ("gx", "gy")]
    fig9 = [f"{Path(fig9_dir) / 'Fig9-volumic-r0=0.41-pf=0.2'}-{p}-qois.dat" for p in ("gx", "gy")]
    kap = kappa0_input_files(lin_def, False) + kappa0_input_files(lin_ref, True, gy_r0=(0.07, 0.30))
    ref = [f"{fig10_basename(ref_dir, mode, r0)}-gx-qois.dat" for mode in ("volumic", "gas_pressure") for r0 in r0_list]
    ref += [f"{fig10_basename(ref_dir, mode, r0)}-gy-qois.dat" for mode, r0 in REF_GXGY]
    ref.append(f"{Path(ref_dir) / 'Fig9-volumic-r0=0.41-pf=0.2'}-gx-qois.dat")
    return {"Fig10.ipynb": fig10, "Fig9.ipynb": fig9, "Fig11-kappa0.ipynb": kap, "Fig11-refined.ipynb": ref}


def fig11_data(fig9_dir, fig10_dir, lin_def, lin_ref, ref_dir, r0_list=R0_LIST):
    """All data of Fig. 11.

    data[(mode, r0)] = (phi_g, y, delta, t, a_D), default mesh (Fig10), delta relative to the default kappa0;
    fig9 = (phi_g, y, delta, t, a_D) of Fig9 "equibiaxial + 0.2 kPa" (r0 = 0.41), default mesh;
    ref[(mode, r0)] = dict(phi, y, dev, t, K, J), refined mesh, gx probe, dev relative to the refined kappa0.
    """
    d = {"r0_list": tuple(r0_list)}
    d["tab_def"] = build_kappa0(lin_def, False)
    d["tab_ref"] = build_kappa0(lin_ref, True)
    d["kappa0"] = kappa0 = PchipInterpolator(d["tab_def"][:, 0], d["tab_def"][:, 1])        # default mesh
    d["kappa0_r"] = kappa0_r = PchipInterpolator(d["tab_ref"][:, 0], d["tab_ref"][:, 1])    # refined mesh
    d["iso_lin"] = isotropy_undeformed(lin_ref)

    data = {}
    for mode in MODES:
        for r0 in r0_list:
            s = K_series(str(fig10_basename(fig10_dir, mode, r0)), K_M)
            phi, y, t, D = valid(s["t"], s["phi_g"], spatial_trace(s), s["t"], anisotropy(s))
            data[(mode, r0)] = (phi, y, y / kappa0(phi) - 1.0, t, D)
    s9 = K_series(str(Path(fig9_dir) / "Fig9-volumic-r0=0.41-pf=0.2"), K_M)
    phi9, y9, t9, D9 = valid(s9["t"], s9["phi_g"], spatial_trace(s9), s9["t"], anisotropy(s9))
    d["data"], d["fig9"] = data, (phi9, y9, y9 / kappa0(phi9) - 1.0, t9, D9)

    ref = {}
    for mode in ("volumic", "gas_pressure"):
        for r0 in r0_list:
            s = gx_series(str(fig10_basename(ref_dir, mode, r0)), K_M)
            phi, y, t, K, J, pfv = valid(s["t"], s["phi_g"], s["y"], s["t"], s["Kxx"], s["J"], s["p_f"])
            ref[(mode, r0)] = dict(phi=phi, y=y, dev=y / kappa0_r(phi) - 1.0, t=t, K=K, J=J)
    s = gx_series(str(Path(ref_dir) / "Fig9-volumic-r0=0.41-pf=0.2"), K_M)
    phi, y, t, K, J = valid(s["t"], s["phi_g"], s["y"], s["t"], s["Kxx"], s["J"])
    ref[("fig9", 0.41)] = dict(phi=phi, y=y, dev=y / kappa0_r(phi) - 1.0, t=t, K=K, J=J)
    d["ref"] = ref

    # isotropy check on the refined gx+gy runs (justifies the gx-only runs)
    iso_def = []
    for mode, r0 in REF_GXGY:
        sgg = K_series(str(fig10_basename(ref_dir, mode, r0)), K_M)
        ok = sgg["t"] >= T_REF - 1e-9
        iso_def.append(np.abs(sgg["K"][ok][:, 1, 1] / sgg["K"][ok][:, 0, 0] - 1).max())
        sg = gx_series(str(fig10_basename(ref_dir, mode, r0)), K_M)
        assert np.abs(sg["y"][ok] / spatial_trace(sgg)[ok] - 1).max() < 1e-4
    d["iso_def"] = iso_def
    print(f"Isotropy of the refined symmetric loadings: max |K_yy/K_xx - 1| = {max(iso_def):.2e}; "
          f"undeformed refined RVEs: " + ", ".join(f"r0 = {k:g}: {v:+.2e}" for k, v in d["iso_lin"].items()))
    return d


# ============================================================================================ Fig. 11 plot

def plot_figure11(d, save_name):
    """Fig. 11 (a) tr(k~)/2k_m, (b) deviation from kappa0, (c) anisotropy a_D, versus the current gas fraction.
    Drawing code of fig11_abc_v3.py (final labels); saved as PDF only. Returns the figure."""
    matplotlib.rcParams["axes.unicode_minus"] = True
    R0_LIST = d["r0_list"]
    data, ref, tab_def = d["data"], d["ref"], d["tab_def"]
    kappa0, kappa0_r = d["kappa0"], d["kappa0_r"]
    phi9, y9, _, _, D9 = d["fig9"]

    cmap = plt.get_cmap("viridis")
    colors = {r0: cmap(i / (len(R0_LIST) - 1)) for i, r0 in enumerate(R0_LIST)}
    fig, (axa, axb, axc) = plt.subplots(3, 1, figsize=(7.4, 12.2), sharex=True,
                                         gridspec_kw=dict(height_ratios=(1.35, 1.0, 1.0), hspace=0.08))
    # (a)
    for mode in ("stretch_x", "simple_shear"):
        for r0 in R0_LIST:
            axa.plot(data[(mode, r0)][0], data[(mode, r0)][1], "o", ms=3, color="0.8", mec="none", zorder=1)
    for r0 in R0_LIST:
        axa.plot(data[("volumic", r0)][0], data[("volumic", r0)][1], "o", ms=4.5, color=colors[r0], mec="none", zorder=3)
        axa.plot(data[("gas_pressure", r0)][0], data[("gas_pressure", r0)][1], "s", ms=4.5, color=colors[r0], mec="none", zorder=3)
    axa.plot(phi9, y9, "^", ms=5, color=colors[0.41], mec="k", mew=0.5, zorder=4)
    # (b): gray = default mesh; symmetric branches = refined mesh
    for mode in ("stretch_x", "simple_shear"):
        for r0 in R0_LIST:
            axb.plot(data[(mode, r0)][0], data[(mode, r0)][2], "o", ms=3, color="0.8", mec="none", zorder=1)
    for r0 in R0_LIST:
        axb.plot(ref[("volumic", r0)]["phi"], ref[("volumic", r0)]["dev"], "o", ms=4.5, color=colors[r0], mec="none", zorder=3)
        axb.plot(ref[("gas_pressure", r0)]["phi"], ref[("gas_pressure", r0)]["dev"], "s", ms=4.5, color=colors[r0], mec="none", zorder=3)
    axb.plot(ref[("fig9", 0.41)]["phi"], ref[("fig9", 0.41)]["dev"], "^", ms=5, color=colors[0.41], mec="k", mew=0.5, zorder=4)
    # (c)
    for key in ("volumic", "gas_pressure"):
        for r0 in R0_LIST:
            axc.plot(data[(key, r0)][0], data[(key, r0)][4], "o", ms=2.2, color="0.55", mec="none", zorder=1)
    axc.plot(phi9, D9, "o", ms=2.2, color="0.55", mec="none", zorder=1)
    for r0 in R0_LIST:
        axc.plot(data[("stretch_x", r0)][0], data[("stretch_x", r0)][4], "D", ms=4.2, color=colors[r0], mec="none", zorder=3)
        axc.plot(data[("simple_shear", r0)][0], data[("simple_shear", r0)][4], "p", ms=5.2, color=colors[r0], mec="none", zorder=3)
    axc.axhline(0.0, color="k", lw=0.6, zorder=0)

    pg = np.linspace(tab_def[0, 0], tab_def[-1, 0], 800)
    axa.plot(pg, kappa0(pg), "-", color="k", lw=1.6, zorder=5)
    axa.plot(pg, h(pg), "--", color="k", lw=0.9, zorder=5)
    axb.plot(pg, h(pg) / kappa0_r(pg) - 1.0, "--", color="k", lw=1.0, zorder=5)
    axb.axhline(0.0, color="k", lw=0.6, zorder=0)

    all_d = np.concatenate([data[k][2] for k in data if k[0] in ("stretch_x", "simple_shear")] +
                           [v["dev"] for v in ref.values()])
    lo, hi = min(all_d.min(), (h(pg) / kappa0_r(pg) - 1).min()), max(all_d.max(), (h(pg) / kappa0_r(pg) - 1).max())
    pad = 0.06 * (hi - lo)
    axb.set_ylim(lo - pad, hi + pad)
    axb.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}".replace("-", MINUS)))

    allphi = np.concatenate([data[k][0] for k in data] + [phi9])
    axc.set_xlim(-0.02, max(allphi.max(), tab_def[-1, 0]) + 0.02)
    Dmax = max(data[k][4].max() for k in data)
    axc.set_ylim(-0.03 * Dmax, 1.08 * Dmax)

    axa.set_ylabel(r"$\mathrm{tr}(\tilde{k})/(2k_m)$")
    axb.set_ylabel(r"$(\mathrm{tr}(\tilde{k})/2k_m)\,/\,\kappa_0(\tilde{\phi}_g)-1$  (%)")
    axc.set_ylabel(r"anisotropy $a_D=(\lambda_1-\lambda_2)/(\lambda_1+\lambda_2)$")
    axc.set_xlabel(r"current gas fraction $\tilde{\phi}_g$")
    for ax, lab in ((axa, "(a)"), (axb, "(b)"), (axc, "(c)")):
        ax.text(-0.13, 1.02, lab, transform=ax.transAxes, fontsize=13, fontweight="bold")

    handles = [Line2D([], [], marker="o", ls="", color="0.4", ms=5, label="equibiaxial stretching"),
               Line2D([], [], marker="s", ls="", color="0.4", ms=5, label="gas pressure"),
               Line2D([], [], marker="^", ls="", color="0.4", mec="k", ms=5, label=r"equibiaxial + 0.2 kPa ($r_0=0.41$, Fig. 9)"),
               Line2D([], [], marker="o", ls="", color="0.8", ms=4, label="uniaxial / simple shear"),
               Line2D([], [], color="k", lw=1.6, label=r"undeformed RVEs, $\kappa_0(\phi)/k_m$"),
               Line2D([], [], color="k", lw=0.9, ls="--", label=r"HS bound $h(\phi)$")]
    axa.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")
    axb.legend([Line2D([], [], color="k", lw=1.0, ls="--")], [r"$h(\phi)/\kappa_0(\phi)-1$"], fontsize=8, frameon=False,
               loc="lower left")
    axc.legend([Line2D([], [], marker="D", ls="", color="0.35", ms=5), Line2D([], [], marker="p", ls="", color="0.35", ms=6),
                Line2D([], [], marker="o", ls="", color="0.55", ms=3)],
               ["uniaxial stretching", "simple shear", "symmetric loadings (equibiaxial, gas pressure): $a_D=0$"],
               fontsize=8, frameon=False, loc="upper left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(R0_LIST[0], R0_LIST[-1]))
    cb = fig.colorbar(sm, ax=[axa, axb, axc], pad=0.02, fraction=0.05)
    cb.set_label(r"$r_0$")
    fig.savefig(save_name, bbox_inches="tight")
    print("Figure saved in:", display_path(save_name))
    return fig


# ============================================================================================ Fig. 11 tables

def _fmt8(v):
    return f"{v:.8g}" if isinstance(v, float) else v


def table2_rows(d):
    """Former Table 2 (refined mesh, final frame): C_K of the RVE and of the two gas-fraction updates."""
    rows = []
    kappa0_r = d["kappa0_r"]
    for mode in ("volumic", "gas_pressure"):
        for r0 in d["r0_list"]:
            x = d["ref"][(mode, r0)]
            phi0, phif = x["phi"][0], x["phi"][-1]
            C_rve = x["K"][-1] / x["K"][0]
            C_lh = h(phif) / h(phi0)
            C_lk = float(kappa0_r(phif) / kappa0_r(phi0))
            rows.append(dict(mode=mode, r0=r0, phi0=phi0, phi_g=phif, C_RVE=C_rve, L1_h=C_lh, L1_kappa0=C_lk,
                             err_h=C_lh - C_rve, err_k=C_lk - C_rve))
    return rows


def deviation_range_rows(d):
    """delta = (tr k~/2k_m)/kappa0(phi_g) - 1 over all loading frames, default mesh and default kappa0."""
    rows = []
    for mode in MODES:
        for r0 in d["r0_list"]:
            phi, y, dv, t, _ = d["data"][(mode, r0)]
            rows.append(dict(mode=mode, r0=r0, d_min=dv.min(), d_max=dv.max(), d_final=dv[-1],
                             t_at_min=t[dv.argmin()], t_at_max=t[dv.argmax()]))
    return rows


def anisotropy_rows(d):
    rows = []
    for mode in MODES:
        for r0 in d["r0_list"]:
            phi, y, dv, t, D = d["data"][(mode, r0)]
            rows.append(dict(mode=mode, r0=r0, D_final=D[-1], D_max=D.max(), phi_g_initial=phi[0], phi_g_final=phi[-1]))
    phi9, y9, _, _, D9 = d["fig9"]
    rows.append(dict(mode="volumic+0.2kPa (Fig9, r0=0.41)", r0=0.41, D_final=D9[-1], D_max=D9.max(),
                     phi_g_initial=phi9[0], phi_g_final=phi9[-1]))
    return rows


def path_independence_rows(d):
    """Equibiaxial path interpolated at the final phi_g of the gas-pressure run (same r0), and the Fig. 9 case
    against the equibiaxial path of r0 = 0.41. The kappa0 columns use the default kappa0."""
    data, kappa0 = d["data"], d["kappa0"]
    rows = []
    for r0 in d["r0_list"]:
        xv, yv, _, tv, _ = data[("volumic", r0)]
        xg, yg, _, _, _ = data[("gas_pressure", r0)]
        m = tv >= T_REF - 1e-9
        xv, yv = xv[m], yv[m]
        order = np.argsort(xv)
        assert np.all(np.diff(xv) > 0), f"equibiaxial phi_g not monotone for r0={r0}"
        phi_t, y_gas = xg[-1], yg[-1]
        assert xv.min() <= phi_t <= xv.max()
        y_eq = float(np.interp(phi_t, xv[order], yv[order]))
        kap = float(kappa0(phi_t))
        rows.append(dict(r0=r0, phi_g=phi_t, y_gas=y_gas, y_equi=y_eq, rel=(y_gas - y_eq) / y_eq, kappa0=kap,
                         gas_vs_kappa=(y_gas - kap) / kap, equi_vs_kappa=(y_eq - kap) / kap))
    xv, yv, _, tv, _ = data[("volumic", 0.41)]
    xv, yv = xv[tv >= T_REF - 1e-9], yv[tv >= T_REF - 1e-9]
    x9, y9 = d["fig9"][0], d["fig9"][1]
    sel = (x9 >= xv.min()) & (x9 <= xv.max())
    rel9 = (y9[sel] - np.interp(x9[sel], xv, yv)) / np.interp(x9[sel], xv, yv)
    fig9 = dict(n_frames=int(sel.sum()), max_abs_rel=float(np.abs(rel9).max()), final_rel=float(rel9[-1]))
    return rows, fig9


# ---------------------------------------------------------------------- L1(h) error decomposition (default mesh)

def _read_qois(path):
    with open(path) as f:
        names = f.readline().lstrip("#").split()
    dd = np.loadtxt(path, ndmin=2)
    return {n: dd[:, i] for i, n in enumerate(names)}


def load_state(fig10_dir, mode, r0):
    """Loading-step frames of a Fig10 case (gx and gy probes): t, F, J, phi, K, k, K0 (2x2 at t = 0.01), phi0.
    F is the mean of the two probe runs and phi = 1 - vs/(J |Omega0|), |Omega0| = sqrt(3)."""
    qx = _read_qois(f"{fig10_basename(fig10_dir, mode, r0)}-gx-qois.dat")
    qy = _read_qois(f"{fig10_basename(fig10_dir, mode, r0)}-gy-qois.dat")
    n = len(qx["t"])
    if n != len(qy["t"]) or qx["t"][-1] < 0.11 - 1e-9 or qy["t"][-1] < 0.11 - 1e-9:
        raise ValueError("incomplete run: " + str(fig10_dir))
    K = np.zeros((n, 2, 2))
    K[:, 0, 0] = -qx["Q_l_avg_x"] / qx["grad_p_bar_avg_x"]
    K[:, 1, 0] = -qx["Q_l_avg_y"] / qx["grad_p_bar_avg_x"]
    K[:, 0, 1] = -qy["Q_l_avg_x"] / qy["grad_p_bar_avg_y"]
    K[:, 1, 1] = -qy["Q_l_avg_y"] / qy["grad_p_bar_avg_y"]
    K /= K_M
    F = np.zeros((n, 2, 2))
    F[:, 0, 0] = 1 + 0.5 * (qx["U_bar_XX"] + qy["U_bar_XX"])
    F[:, 0, 1] = 0.5 * (qx["U_bar_XY"] + qy["U_bar_XY"])
    F[:, 1, 0] = 0.5 * (qx["U_bar_YX"] + qy["U_bar_YX"])
    F[:, 1, 1] = 1 + 0.5 * (qx["U_bar_YY"] + qy["U_bar_YY"])
    J = np.linalg.det(F)
    phi = 1.0 - 0.5 * (qx["vs"] + qy["vs"]) / (J * OMEGA0)
    k = F @ K @ np.swapaxes(F, 1, 2) / J[:, None, None]
    sel = qx["t"] >= T_REF - 1e-12
    i0 = int(np.where(sel)[0][0])
    return {"t": qx["t"][sel], "F": F[sel], "J": J[sel], "phi": phi[sel], "K": K[sel], "k": k[sel],
            "K0": K[i0], "phi0": phi[i0]}


def l1_decomposition_h(s):
    """L1 (gas-fraction update, h version): K_L1 = sqrt(det K0) h(phi)/h(phi0) J F^-1 F^-T.
    Returns size = sqrt(det K)/predicted - 1, anisotropy = |S_rve - S_kin|_F/|S_rve|_F,
    total = |K - K_L1|_F/|K|_F (all frames)."""
    size_ratio = h(s["phi"]) / h(s["phi0"])
    F, J = s["F"], s["J"]
    K = 0.5 * (s["K"] + np.swapaxes(s["K"], 1, 2))
    Finv = np.linalg.inv(F)
    S_kin = J[:, None, None] * Finv @ np.swapaxes(Finv, 1, 2)
    detK = np.sqrt(np.linalg.det(K))
    S_rve = K / detK[:, None, None]
    K0 = 0.5 * (s["K0"] + s["K0"].T)
    pred = np.sqrt(np.linalg.det(K0)) * size_ratio
    Kp = pred[:, None, None] * S_kin
    size = detK / pred - 1.0
    shape = np.linalg.norm(S_rve - S_kin, axis=(1, 2)) / np.linalg.norm(S_rve, axis=(1, 2))
    total = np.linalg.norm(K - Kp, axis=(1, 2)) / np.linalg.norm(K, axis=(1, 2))
    return size, shape, total


def l1_decomposition_rows(fig10_dir, r0_list=R0_LIST):
    rows = []
    for mode in ("stretch_x", "simple_shear"):
        for r0 in r0_list:
            size, aniso, total = l1_decomposition_h(load_state(fig10_dir, mode, r0))
            rows.append(dict(mode=mode, r0=r0, size=size[-1], aniso=aniso[-1], total=total[-1], total_max=total.max(),
                             aniso_over_abs_size=aniso[-1] / abs(size[-1])))
    return rows


def _write_csv(path, rows, fmt=_fmt8):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})


def fig11_report(d):
    """Isotropy checks, kappa0 refined vs default, symmetric branches against kappa0 and h (fig11_abc_v3 report)."""
    tab_def, tab_ref, kappa0, kappa0_r = d["tab_def"], d["tab_ref"], d["kappa0"], d["kappa0_r"]
    data, ref = d["data"], d["ref"]
    L = ["# Fig. 11: refined-mesh symmetric branches\n"]
    L.append("## Isotropy checks (justify the gx-only runs)\n")
    L.append(f"- refined deformed gx+gy runs (volumic r0 = 0.07, 0.10, 0.15, 0.20; gas pressure r0 = 0.07; all frames t >= 0.01): "
             f"max |K_yy/K_xx - 1| = {max(d['iso_def']):.2e} (threshold 1e-3); gx-only tr(k)/2 vs the full gx+gy trace: < 1e-4.")
    L.append("- undeformed refined hexagonal RVE, K_yy(gy)/K_xx(gx) - 1: " +
             ", ".join(f"r0 = {k:g}: {v:+.2e}" for k, v in d["iso_lin"].items()) + "\n")
    L.append("## kappa0: refined vs default mesh at the Fig. 10 r0\n")
    L.append("| r0 | phi | kappa0 default (l=0.03) | kappa0 refined | (refined - default)/default |")
    L.append("|---|---|---|---|---|")
    for r0 in d["r0_list"]:
        i = int(np.argmin(np.abs(tab_ref[:, 2] - r0)))
        L.append(f"| {r0:g} | {tab_ref[i, 0]:.4f} | {tab_def[i, 1]:.5f} | {tab_ref[i, 1]:.5f} | {100 * (tab_ref[i, 1] / tab_def[i, 1] - 1):+.3f}% |")
    pgrid = np.linspace(tab_def[0, 0], tab_def[-1, 0], 2000)
    dd = kappa0_r(pgrid) / kappa0(pgrid) - 1
    L.append(f"\nmax |refined/default - 1| over all phi: {np.abs(dd).max():.2e} at phi = {pgrid[np.abs(dd).argmax()]:.3f}\n")
    L.append("## Symmetric branches, refined mesh: final frame and maximum excess over h (all loading steps)\n")
    L.append("dev_k = (tr k/2k_m)/kappa0_refined(phi_g) - 1 ; dev_h = (tr k/2k_m)/h(phi_g) - 1 (> 0 violates HS). "
             "'max dev_h' = max over all frames t >= 0.01 [phi_g at that frame]. Percent. 'default' = l = 0.03.\n")
    L.append("| branch | phi_g final | dev_k final (refined) | dev_h final (refined) | max dev_h (refined) [phi_g] | "
             "dev_h final (default) | max dev_h (default) |")
    L.append("|---|---|---|---|---|---|---|")

    def line(name, x, dflt):
        dh = x["y"] / h(x["phi"]) - 1
        k = int(np.argmax(dh))
        dh0 = dflt[1] / h(dflt[0]) - 1
        dfs = f"{100 * dh0[-1]:+.3f}% | {100 * dh0.max():+.3f}%"
        return (f"| {name} | {x['phi'][-1]:.4f} | {100 * x['dev'][-1]:+.3f}% | {100 * dh[-1]:+.3f}% | "
                f"{100 * dh[k]:+.3f}% [{x['phi'][k]:.4f}] | {dfs} |")

    for mode in ("volumic", "gas_pressure"):
        for r0 in d["r0_list"]:
            L.append(line(f"{mode} r0={r0:g}", ref[(mode, r0)], data[(mode, r0)]))
    L.append(line("Fig9 volumic+0.2 kPa r0=0.41", ref[("fig9", 0.41)], d["fig9"]))
    return "\n".join(L) + "\n"


def write_fig11_tables(d, out_dir, fig10_dir):
    """Write the tables behind Fig. 11 and the numbers of Sec. 3.5.2 to out_dir (nothing is printed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "kappa0_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi", "kappa0_over_km_refined", "r0", "kappa0_over_km_default_same_r0"])
        for a, b in zip(d["tab_ref"], d["tab_def"]):
            w.writerow([f"{a[0]:.10g}", f"{a[1]:.10g}", f"{a[2]:g}", f"{b[1]:.10g}"])
    _write_csv(out_dir / "table2_refined.csv", table2_rows(d))
    _write_csv(out_dir / "deviation_range_default.csv", deviation_range_rows(d), fmt=lambda v: v)
    _write_csv(out_dir / "anisotropy.csv", anisotropy_rows(d), fmt=lambda v: f"{v:.6g}" if isinstance(v, float) else v)
    pi_rows, _ = path_independence_rows(d)
    _write_csv(out_dir / "path_independence.csv", pi_rows, fmt=lambda v: v)
    _write_csv(out_dir / "l1_decomposition_h.csv", l1_decomposition_rows(fig10_dir, d["r0_list"]))
    (out_dir / "fig11_report.md").write_text(fig11_report(d))
    names = ["kappa0_table.csv", "table2_refined.csv", "deviation_range_default.csv", "anisotropy.csv",
             "path_independence.csv", "l1_decomposition_h.csv", "fig11_report.md"]
    print("Tables written in", display_path(out_dir), ":", ", ".join(names))


# ============================================================================================ state-function test (Fig10)

ISO_TOL = 1e-3   # |A_K - 1| below this: principal direction is numerically undefined
SFT_MODES = ("stretch_x", "volumic", "simple_shear", "gas_pressure")


def _sft_read_case(fig10_dir, mode, r0):
    base = fig10_basename(fig10_dir, mode, r0)
    ng, dg = load(f"{base}-gx-qois.dat")
    ny, dy = load(f"{base}-gy-qois.dat")
    c = lambda n, d, k: d[:, n.index(k)]
    t = c(ng, dg, "t")
    gx, gy = c(ng, dg, "grad_p_bar_avg_x"), c(ny, dy, "grad_p_bar_avg_y")
    K = np.empty((len(t), 2, 2))
    K[:, 0, 0] = -c(ng, dg, "Q_l_avg_x") / (gx + EPS)
    K[:, 1, 0] = -c(ng, dg, "Q_l_avg_y") / (gx + EPS)
    K[:, 0, 1] = -c(ny, dy, "Q_l_avg_x") / (gy + EPS)
    K[:, 1, 1] = -c(ny, dy, "Q_l_avg_y") / (gy + EPS)
    F = np.empty_like(K)
    F[:, 0, 0] = 1 + c(ng, dg, "U_bar_XX")
    F[:, 0, 1] = c(ng, dg, "U_bar_XY")
    F[:, 1, 0] = c(ng, dg, "U_bar_YX")
    F[:, 1, 1] = 1 + c(ng, dg, "U_bar_YY")
    v, vs = c(ng, dg, "v"), c(ng, dg, "vs")
    return dict(t=t, K=K / K_M, F=F, v=v, vs=vs, V0=v[0])


def _principal(K):
    Ks = 0.5 * (K + K.T)
    a, b, c = Ks[0, 0], Ks[0, 1], Ks[1, 1]
    dd = np.sqrt((a - c) ** 2 + 4 * b ** 2)
    lam1, lam2 = 0.5 * (a + c + dd), 0.5 * (a + c - dd)
    theta = np.rad2deg(0.5 * np.arctan2(2 * b, a - c))     # (-90, 90]
    return lam1, lam2, theta


def _fold(angle):
    """angle difference folded to (-90, 90] (axis, not vector)."""
    return (angle + 90.0) % 180.0 - 90.0


def _indicators(K, K0):
    lam1, lam2, th = _principal(K)
    A = lam1 / lam2
    return dict(C=np.trace(K) / np.trace(K0), A=A, theta=th if abs(A - 1.0) > ISO_TOL else np.nan)


def state_function_rows(fig10_dir, r0_list=R0_LIST):
    """State-function test of the reduced permeability models against the RVE (Fig. 10 cases, final frame).

    RVE: K~ from the gx and gy probes, C_K = tr K~/tr K~0 (K~0 at t = 0.01), A_K = K1/K2, theta.
    L0 (kinematic update):    Kbar = J F^-1 K~0 F^-T.
    L1 (gas-fraction update): Kbar = [h(phi_g)/h(Phi_g0)] * L0.
    Errors of Kbar with respect to K~: dC_K (abs, rel), dA_K, dtheta, E_K = ||K~ - Kbar||_F/||K~||_F.
    """
    rows = []
    for mode in SFT_MODES:
        for r0 in r0_list:
            s = _sft_read_case(fig10_dir, mode, r0)
            i0 = int(np.argmin(np.abs(s["t"] - T_REF)))
            assert abs(s["t"][i0] - T_REF) < 1e-9
            K0, Kf, Ff = s["K"][i0], s["K"][-1], s["F"][-1]
            J = s["v"][-1] / s["V0"]
            J_F = np.linalg.det(Ff)
            phi_g = 1.0 - s["vs"][-1] / s["v"][-1]
            phi_g0 = 1.0 - s["vs"][i0] / s["v"][i0]
            Finv = np.linalg.inv(Ff)
            Kbar0 = J * Finv @ K0 @ Finv.T
            ratio = h(phi_g) / h(phi_g0)
            Kbar1 = ratio * Kbar0
            rve = _indicators(Kf, K0)
            row = dict(mode=mode, r0=r0, Phi_g0=phi_g0, phi_g=phi_g, J=J, J_minus_detF=J - J_F, h_ratio=ratio,
                       C_K_rve=rve["C"], A_K_rve=rve["A"], theta_rve=rve["theta"])
            for tag, Kb in (("L0", Kbar0), ("L1", Kbar1)):
                m = _indicators(Kb, K0)
                row[f"C_K_{tag}"] = m["C"]
                row[f"dC_abs_{tag}"] = m["C"] - rve["C"]
                row[f"dC_rel_{tag}"] = (m["C"] - rve["C"]) / rve["C"]
                row[f"dA_{tag}"] = m["A"] - rve["A"]
                row[f"dtheta_{tag}"] = _fold(m["theta"] - rve["theta"]) if np.isfinite(m["theta"] * rve["theta"]) else np.nan
                row[f"E_K_{tag}"] = np.linalg.norm(Kf - Kb, "fro") / np.linalg.norm(Kf, "fro")
            rows.append(row)
    return rows


def write_state_function_test(rows, out_dir):
    """state_function_test_results.csv and summary.txt (same format as the original state_function_test.py)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "state_function_test_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = []
    hdr = ("r0    Phi_g0  phi_g   J      | C_K(RVE) A_K(RVE) theta(RVE) || L0: dC_K(abs) dC_K(rel) dA_K  dtheta   E_K "
           "|| L1: dC_K(abs) dC_K(rel) dA_K  dtheta   E_K")
    fmt_th = lambda x: "  n/a " if not np.isfinite(x) else f"{x:6.2f}"
    for mode in SFT_MODES:
        lines.append(f"\n### {mode}   (signed errors = model - RVE; dtheta in degrees; n/a = near-isotropic, axis undefined)")
        lines.append(hdr)
        for r in [x for x in rows if x["mode"] == mode]:
            lines.append(
                f"{r['r0']:<5g} {r['Phi_g0']:.4f} {r['phi_g']:.4f} {r['J']:.4f} | {r['C_K_rve']:.4f}   {r['A_K_rve']:.4f}   {fmt_th(r['theta_rve'])} || "
                f"{r['dC_abs_L0']:+.4f}   {r['dC_rel_L0']:+.2%}   {r['dA_L0']:+.4f} {fmt_th(r['dtheta_L0'])} {r['E_K_L0']:.4f} || "
                f"{r['dC_abs_L1']:+.4f}   {r['dC_rel_L1']:+.2%}   {r['dA_L1']:+.4f} {fmt_th(r['dtheta_L1'])} {r['E_K_L1']:.4f}")
    lines.append(f"\nmax |J - det F| over all cases: {max(abs(r['J_minus_detF']) for r in rows):.1e}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("State-function test written in", display_path(out_dir), ": state_function_test_results.csv, summary.txt")


def _row(rows, mode, r0):
    return next(r for r in rows if r["mode"] == mode and abs(r["r0"] - r0) < 1e-12)


def _argmax(rows, mode, key, absval=False):
    rr = [r for r in rows if r["mode"] == mode]
    return max(rr, key=lambda r: abs(r[key]) if absval else r[key])


def print_state_function_numbers(rows):
    """Numbers quoted in the abstract, the introduction and Sec. 3.5 (default mesh l = 0.03, Fig. 10 data)."""
    r07 = _row(rows, "volumic", 0.07)
    rmax = _argmax(rows, "volumic", "dC_rel_L0")
    ek = _argmax(rows, "volumic", "E_K_L0")
    print("Default mesh (l = 0.03), Fig. 10 data; reference frame t = 0.01; final frame.")
    print(f"[Abstract] kinematic update, equibiaxial r0 = 0.07 (Phi_g0 = {r07['Phi_g0']:.4f}): "
          f"C_K error = {100 * r07['dC_rel_L0']:+.2f}%")
    print(f"[Introduction] kinematic update, equibiaxial, maximum over r0: {100 * rmax['dC_rel_L0']:+.2f}% "
          f"(r0 = {rmax['r0']:g})")
    print(f"[Sec. 3.5.1] E_K of the kinematic update, equibiaxial, maximum: {ek['E_K_L0']:.4f} (r0 = {ek['r0']:g})")
    sh = [r["C_K_rve"] for r in rows if r["mode"] == "simple_shear"]
    print(f"[Sec. 3.5.1] simple shear: C_K in [{min(sh):.4f}, {max(sh):.4f}], max |1 - C_K| = "
          f"{100 * max(abs(1 - c) for c in sh):.2f}%")
    for mode in ("stretch_x", "simple_shear", "volumic", "gas_pressure"):
        a = [r["A_K_rve"] for r in rows if r["mode"] == mode]
        print(f"[Sec. 3.5.1] A_K {mode}: [{min(a):.4f}, {max(a):.4f}]")
    print(f"[Sec. 3.5.2] equibiaxial r0 = 0.07: J = {r07['J']:.4f}, Phi_g0 = {r07['Phi_g0']:.4f} -> phi_g = {r07['phi_g']:.4f}")


# ============================================================================================ Fig. 11 paper numbers

def _pct(v, nd=3):
    return f"{100 * v:+.{nd}f}%"


def paper_numbers(d, fig10_dir):
    """Rows (location, quantity, data, value) of the numbers quoted in the paper that come from
    the Fig. 11 data (and, for the abstract and the introduction, from the Fig. 10 data)."""
    out = []

    def add(loc, what, data, value):
        out.append(dict(location=loc, quantity=what, data=data, value=value))

    sft = state_function_rows(fig10_dir, d["r0_list"])
    r07 = _row(sft, "volumic", 0.07)
    rmax = _argmax(sft, "volumic", "dC_rel_L0")
    add("Abstract", f"kinematic update error, equibiaxial r0 = 0.07 (Phi_g0 = {r07['Phi_g0']:.4f})",
        "default mesh l = 0.03 (Fig. 10)", _pct(r07["dC_rel_L0"], 2))
    add("Introduction", f"kinematic update error, equibiaxial, max over r0 (r0 = {rmax['r0']:g})",
        "default mesh l = 0.03 (Fig. 10)", _pct(rmax["dC_rel_L0"], 2))

    ref = d["ref"]
    for mode in ("volumic", "gas_pressure"):
        errs = []
        for r0 in d["r0_list"]:
            x = ref[(mode, r0)]
            C_rve = x["K"][-1] / x["K"][0]
            errs.append((abs(h(x["phi"][-1]) / h(x["phi"][0]) / C_rve - 1.0), r0))
        e, r0 = max(errs)
        add("Abstract; Sec. 3.5.2", f"gas-fraction update vs RVE, |C_h/C_RVE - 1|, {mode}, max over r0 (r0 = {r0:g})",
            "refined mesh (Fig. 11(b))", f"{100 * e:.2f}%")

    x = ref[("volumic", 0.07)]
    add("Sec. 3.5.2", "equibiaxial r0 = 0.07: J final, Phi_g0 -> phi_g final", "refined mesh (Fig. 11(b))",
        f"J = {x['J'][-1]:.4f}, {x['phi'][0]:.4f} -> {x['phi'][-1]:.4f}")

    pi_rows, pi9 = path_independence_rows(d)
    big = max(pi_rows, key=lambda r: abs(r["rel"]))
    add("Sec. 3.5.2", f"path independence: gas pressure vs equibiaxial at equal phi_g, max |rel| (r0 = {big['r0']:g})",
        "default mesh (Fig. 11(a))", f"{100 * big['rel']:+.4f}%")
    add("Sec. 3.5.2", f"path independence: Fig. 9 (+0.2 kPa) vs equibiaxial r0 = 0.41, max |rel| over {pi9['n_frames']} frames",
        "default mesh (Fig. 11(a))", f"{100 * pi9['max_abs_rel']:.4f}%")

    devs = [(x["dev"].max(), k) for k, x in ref.items()]
    v, k = max(devs)
    add("Sec. 3.5.2", f"max deviation above kappa0, symmetric loadings ({k[0]} r0 = {k[1]:g})",
        "refined mesh, refined kappa0 (Fig. 11(b))", _pct(v))

    exc = []
    for k, x in ref.items():
        dh = x["y"] / h(x["phi"]) - 1
        i = int(np.argmax(dh))
        exc.append((dh[i], k, x["phi"][i]))
    v, k, ph = max(exc)
    over = sorted(f"{kk[0]} r0={kk[1]:g}" for vv, kk, _ in exc if vv > 0)
    add("Sec. 3.5.2", f"max excess over the HS bound ({k[0]} r0 = {k[1]:g}, phi_g = {ph:.4f}); branches above h: "
        + ", ".join(over), "refined mesh (Fig. 11(b))", _pct(v))

    for mode in ("stretch_x", "simple_shear"):
        v, r0 = min((d["data"][(mode, r0)][2].min(), r0) for r0 in d["r0_list"])
        add("Sec. 3.5.2", f"min deviation from kappa0, {mode} (r0 = {r0:g})",
            "default mesh, default 49-point kappa0 (Fig. 11(b) gray)", _pct(v))

    for mode in ("stretch_x", "simple_shear"):
        v, r0 = max((d["data"][(mode, r0)][4].max(), r0) for r0 in d["r0_list"])
        add("Sec. 3.5.2", f"max anisotropy a_D, {mode} (r0 = {r0:g})", "default mesh (Fig. 11(c))", f"{v:.4f}")

    l1 = l1_decomposition_rows(fig10_dir, d["r0_list"])
    for mode in ("stretch_x", "simple_shear"):
        r = next(r for r in l1 if r["mode"] == mode and abs(r["r0"] - 0.30) < 1e-12)
        rr = [q["aniso_over_abs_size"] for q in l1 if q["mode"] == mode]
        add("Sec. 3.5.2", f"L1(h) error at the final frame, {mode} r0 = 0.30: anisotropy, size (ratio); ratio over all r0",
            "default mesh (Fig. 10)", f"aniso {r['aniso']:.4f}, size {r['size']:+.4f} (ratio {r['aniso_over_abs_size']:.2f}); "
            f"all r0: {min(rr):.2f}-{max(rr):.2f}")
    return out


def print_paper_numbers(rows):
    for r in rows:
        print(f"[{r['location']}] {r['quantity']}\n    = {r['value']}    [{r['data']}]")


# ============================================================================================ Sec. 2.4 (Fig11-refined)

def mesh_convergence_rows(ref_dir, fig10_dir, r0_list=(0.07, 0.10)):
    """Equibiaxial stretching (J = 1.69), refined (l = 0.015 + local) vs default (l = 0.03) mesh, gx + gy probes.
    K = tr K~/(2 k_m); C_K = K(final)/K(t = 0.01)."""
    rows = []
    for r0 in r0_list:
        row = {"r0": r0}
        for tag, folder in (("default", fig10_dir), ("refined", ref_dir)):
            s = K_series(str(fig10_basename(folder, "volumic", r0)), K_M)
            i0 = int(np.argmin(np.abs(s["t"] - T_REF)))
            assert abs(s["t"][i0] - T_REF) < 1e-9
            K = 0.5 * np.trace(s["K"], axis1=1, axis2=2)
            row[f"C_K_{tag}"] = K[-1] / K[i0]
            row[f"K_final_{tag}"] = K[-1]
            row[f"K_ref_{tag}"] = K[i0]
            row[f"J_final_{tag}"] = np.linalg.det(s["F"][-1])
        for q in ("C_K", "K_final", "K_ref"):
            row[f"d{q}_rel"] = row[f"{q}_refined"] / row[f"{q}_default"] - 1.0
        rows.append(row)
    return rows


def print_mesh_convergence(rows, out_csv):
    _write_csv(out_csv, rows)
    for r in rows:
        print(f"[Sec. 2.4] equibiaxial r0 = {r['r0']:g}, J = {r['J_final_refined']:.4f}, refined vs default mesh:")
        print(f"    metric 1, C_K = tr K(final)/tr K(t = 0.01): {r['C_K_default']:.5f} -> {r['C_K_refined']:.5f}, "
              f"change {100 * r['dC_K_rel']:+.2f}%")
        print(f"    metric 2, tr K/(2 k_m) at the final frame: {r['K_final_default']:.6f} -> {r['K_final_refined']:.6f}, "
              f"change {100 * r['dK_final_rel']:+.2f}%")
        print(f"    (reference frame t = 0.01: change {100 * r['dK_ref_rel']:+.2f}%)")
    print("Written:", display_path(out_csv))


# ============================================================================================ Sec. 3.1 (Fig3)

def hs_check_linear(res_folder, shape_list, r0_list, out_csv, eps=1e-12):
    """Linear homogenization of Fig. 3 against the 2D HS bound h(phi) = (1 - phi)/(1 + phi).

    Same K as plot_linear_homogenization_hollowbox (last frame, K1 >= K2 principal values of sym K,
    Keq = (K1 + K2)/2, phi = mesh porosity), plus h, gap_eq = h/Keq - 1 and gap_K1 = h/K1 - 1."""
    rows = []
    for shape in shape_list:
        for r0 in r0_list:
            base = Path(res_folder) / f"linear-{shape}-r0={r0:.6g}"
            Kcol = {}
            for probe in ("gx", "gy"):
                n, dd = load(f"{base}-{probe}-qois.dat")
                Kcol[probe] = (dd[-1, n.index("Q_l_avg_x")], dd[-1, n.index("Q_l_avg_y")],
                               dd[-1, n.index("grad_p_bar_avg_x")], dd[-1, n.index("grad_p_bar_avg_y")])
            qx_gx, qy_gx, gx, _ = Kcol["gx"]
            qx_gy, qy_gy, _, gy = Kcol["gy"]
            Kxx = -qx_gx / (gx + eps)
            Kyx = -qy_gx / (gx + eps)
            Kxy = -qx_gy / (gy + eps)
            Kyy = -qy_gy / (gy + eps)
            a, b, c = Kxx, 0.5 * (Kxy + Kyx), Kyy
            tr, delta = a + c, np.sqrt((a - c) ** 2 + 4.0 * b ** 2)
            K1, K2 = 0.5 * (tr + delta), 0.5 * (tr - delta)
            with open(f"{base}-gx-metadata.json") as f:
                phi = float(json.load(f)["mesh_porosity"])
            rows.append(dict(shape=shape, r0=r0, phi=phi, Kxx=Kxx, Kxy=Kxy, Kyx=Kyx, Kyy=Kyy, K1=K1, K2=K2,
                             Keq=0.5 * (K1 + K2)))
    df = pd.DataFrame(rows).sort_values(["shape", "phi"]).reset_index(drop=True)
    df["h"] = h(df["phi"])
    df["gap_eq"] = df["h"] / df["Keq"] - 1.0
    df["gap_K1"] = df["h"] / df["K1"] - 1.0
    df.to_csv(out_csv, index=False)
    print("Written:", display_path(out_csv))
    return df


def print_hs_check(df):
    for shape, g in df.groupby("shape", sort=False):
        i = g["gap_eq"].abs().idxmax()
        print(f"[Sec. 3.1] {shape}: max |h/Keq - 1| = {100 * abs(df.loc[i, 'gap_eq']):.2f}% (r0 = {df.loc[i, 'r0']:g}, "
              f"phi = {df.loc[i, 'phi']:.4f}); h/Keq - 1 in [{100 * g['gap_eq'].min():+.3f}%, {100 * g['gap_eq'].max():+.3f}%]")
