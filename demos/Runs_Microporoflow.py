"""Finite-element runs used by the Fig. 11 notebooks (Fig11-kappa0.ipynb, Fig11-refined.ipynb).

Two kinds of runs on the hexagonal-pore RVE (hex pores + central pore, periodic boundary conditions):

* run_linear_hex_case: one-step linear homogenization of the undeformed RVE (k_l = identity, probe
  gradient 0.01), as in Fig3.ipynb, used to build the reference curve kappa0(phi);
* run_refined_case: the two-step loading cases of Fig10.ipynb (first step: probe gradient ramp, second
  step: loading), and the "equibiaxial + 0.2 kPa" case of Fig9.ipynb, used on a refined mesh.

Two options are not available in dolfin_mech.run_HollowBox_MicroPoroflow and are injected here, each
only for the duration of one case:

* local mesh refinement around the pores (gmsh background field), by wrapping gmsh.model.mesh.generate;
* the LU backend (MUMPS instead of the hard-coded UMFPACK), by wrapping dmech.NonlinearSolver.

Both wrappers are context managers that restore the original objects on exit (also on errors), and
run_case_list checks after every case that the original objects are back in place.
"""
import contextlib
import math
import time
from pathlib import Path

import dolfin
import gmsh
import dolfin_mech as dmech

MATERIAL_PARAMETERS = {"alpha": 0.16, "gamma": 0.5, "c1": 0.2, "c2": 0.4, "kappa": 1e2, "eta": 1e-5}

# Original objects replaced temporarily by local_refinement() and lu_solver(). The raw class attribute
# of gmsh.model.mesh is stored (a staticmethod object), so that it can be restored exactly.
_GMSH_GENERATE_RAW = vars(gmsh.model.mesh)["generate"]
_NONLINEAR_SOLVER = dmech.NonlinearSolver


def check_restored():
    """Raise RuntimeError if gmsh.model.mesh.generate or dmech.NonlinearSolver is not the original object."""
    problems = []
    if vars(gmsh.model.mesh)["generate"] is not _GMSH_GENERATE_RAW:
        problems.append("gmsh.model.mesh.generate is still wrapped (local mesh refinement was not removed)")
    if dmech.NonlinearSolver is not _NONLINEAR_SOLVER:
        problems.append("dmech.NonlinearSolver is still wrapped (LU backend override was not removed)")
    if problems:
        raise RuntimeError("; ".join(problems) + ". The following cases would silently use these settings.")


@contextlib.contextmanager
def local_refinement(r0, l, size_near):
    """Refine the mesh near the pores while the context is active.

    Before meshing, a Threshold-of-Distance background field is added around all geometry points (hexagon
    vertices): size size_near within r0 of them (this covers the whole pore boundary), growing back to l
    at distance r0 + 4*l. run_HollowBox_Mesh itself is not modified.
    """
    orig = gmsh.model.mesh.generate

    def generate(dim=-1):
        pts = [t for (d, t) in gmsh.model.getEntities(0)]
        f = gmsh.model.mesh.field
        f.add("Distance", 1)
        f.setNumbers(1, "PointsList", pts)
        f.add("Threshold", 2)
        f.setNumber(2, "InField", 1)
        f.setNumber(2, "SizeMin", size_near)
        f.setNumber(2, "SizeMax", l)
        f.setNumber(2, "DistMin", r0)
        f.setNumber(2, "DistMax", r0 + 4.0 * l)
        f.setAsBackgroundMesh(2)
        return orig(dim)

    if vars(gmsh.model.mesh)["generate"] is not _GMSH_GENERATE_RAW:
        raise RuntimeError("gmsh.model.mesh.generate is already wrapped; refusing to wrap it twice")
    gmsh.model.mesh.generate = generate
    try:
        yield
    finally:
        setattr(gmsh.model.mesh, "generate", _GMSH_GENERATE_RAW)


@contextlib.contextmanager
def lu_solver(name):
    """Use the LU backend `name` in dmech.NonlinearSolver while the context is active ("umfpack": no change)."""
    if name == "umfpack":
        yield
        return
    if dmech.NonlinearSolver is not _NONLINEAR_SOLVER:
        raise RuntimeError("dmech.NonlinearSolver is already wrapped; refusing to wrap it twice")

    def patched(problem, parameters={}, **kw):
        parameters = dict(parameters, linear_solver_name=name)
        return _NONLINEAR_SOLVER(problem=problem, parameters=parameters, **kw)

    dmech.NonlinearSolver = patched
    try:
        yield
    finally:
        dmech.NonlinearSolver = _NONLINEAR_SOLVER


def mesh_params(r0, l, mesh_filebasename):
    return {
        "dim": 2,
        "xmin": 0.0,
        "ymin": -math.sqrt(3.0) / 2.0,
        "xmax": 1.0,
        "ymax": math.sqrt(3.0) / 2.0,
        "r0": r0,
        "l": l,
        "hole_shape": "hex",
        "add_center_hole": True,
        "mesh_filebasename": str(mesh_filebasename),
    }


def run_linear_hex_case(res_folder, r0, probe, l=0.03, amp=0.01):
    """One-step linear homogenization of the undeformed hexagonal-pore RVE (k_l = identity).

    Output: res_folder/linear-hex-r0=<r0>-<probe>.{xdmf,h5,-qois.dat,-metadata.json}
            res_folder/mesh-hex-r0=<r0>-<probe>.{xdmf,h5,vtk}
    """
    res_folder = Path(res_folder)
    res_folder.mkdir(parents=True, exist_ok=True)
    gx = ([0.0], [amp] if probe == "gx" else [0.0])
    gy = ([0.0], [amp] if probe == "gy" else [0.0])
    lp = {"solid": {f"sigma_bar_{i}{j}": 0.0 for i in range(2) for j in range(2)}, "liquid": {}, "air": {}}
    for k in ("00", "01", "10", "11"):
        lp["solid"][f"U_bar_{k}_lst"] = [0.0]
    lq = lp["liquid"]
    lq["pl_bar_ini_lst"] = [0.0]
    lq["pl_bar_fin_lst"] = [0.0]
    lq["grad_p_bar_x_ini_lst"], lq["grad_p_bar_x_fin_lst"] = gx
    lq["grad_p_bar_y_ini_lst"], lq["grad_p_bar_y_fin_lst"] = gy
    for k in ("Theta_in_ini_lst", "Theta_in_fin_lst", "Theta_out_ini_lst", "Theta_out_fin_lst"):
        lq[k] = [0.0]
    lp["air"]["pf"] = 0.0
    lp["air"]["pf_lst"] = [0.0]
    r0s = f"{r0:.6g}"
    dmech.run_HollowBox_MicroPoroflow(
        dim=2,
        mesh_params=mesh_params(r0, l, res_folder / f"mesh-hex-r0={r0s}-{probe}"),
        mat_params={k: {"parameters": dict(MATERIAL_PARAMETERS), "scaling": "no"} for k in ("skel", "bulk", "pore")},
        flow_params={"k_l": dolfin.Constant(((1.0, 0.0), (0.0, 1.0))), "use_kozeny_carman": False},
        porosity_params={"type": "constant", "val": 0.3},
        bcs="pbc",
        step_params={"n_steps": 1, "Deltat_lst": [1e-2], "dt_ini_lst": [1e-3], "dt_min_lst": [1e-4],
                     "dt_max_lst": [5e-3]},
        load_params=lp,
        res_basename=str(res_folder / f"linear-hex-r0={r0s}-{probe}"),
        verbose=0)
    return res_folder / f"linear-hex-r0={r0s}-{probe}"


def run_refined_case(res_folder, mode, r0, probe, l=0.015, gamp=0.01, pg=0.5, Umax=0.3, beta=0.3,
                     prefix="Fig10", pf_lst=None):
    """Two-step loading case of Fig10.ipynb (prefix "Fig10") or Fig9.ipynb (prefix "Fig9", explicit pf_lst).

    Step 1 (Delta t = 0.01): probe pressure gradient 0 -> gamp, no deformation.
    Step 2 (Delta t = 0.1): loading (stretch_x / volumic: U_bar = Umax; simple_shear: U_bar_01 = beta;
    gas_pressure: F = I and gas pressure 0 -> pg), probe gradient kept constant.
    If pf_lst is given, the gas pressure follows pf_lst (end of step 1, end of step 2) for any mode.

    Output: res_folder/<prefix>-<mode>-r0=<r0>-pf=<final gas pressure>-<probe>.{xdmf,h5,-qois.dat,-metadata.json}
            res_folder/mesh-<mode>-r0=<r0>-<probe>.{xdmf,h5,vtk}
    """
    res_folder = Path(res_folder)
    res_folder.mkdir(parents=True, exist_ok=True)

    solid = {}
    if mode == "stretch_x":
        solid["U_bar_00_lst"] = [0.0, Umax]
    elif mode == "volumic":
        solid["U_bar_00_lst"] = [0.0, Umax]
        solid["U_bar_11_lst"] = [0.0, Umax]
    elif mode == "simple_shear":
        solid.update(U_bar_00_lst=[0.0, 0.0], U_bar_01_lst=[0.0, beta],
                     U_bar_10_lst=[0.0, 0.0], U_bar_11_lst=[0.0, 0.0])
    elif mode == "gas_pressure":
        solid.update(U_bar_00_lst=[0.0, 0.0], U_bar_01_lst=[0.0, 0.0],
                     U_bar_10_lst=[0.0, 0.0], U_bar_11_lst=[0.0, 0.0])
    else:
        raise ValueError("mode must be 'stretch_x', 'volumic', 'simple_shear' or 'gas_pressure'")

    if probe == "gx":
        gx_ini, gx_fin, gy_ini, gy_fin = [0.0, gamp], [gamp, gamp], [0.0, 0.0], [0.0, 0.0]
    elif probe == "gy":
        gx_ini, gx_fin, gy_ini, gy_fin = [0.0, 0.0], [0.0, 0.0], [0.0, gamp], [gamp, gamp]
    else:
        raise ValueError("probe must be 'gx' or 'gy'")

    if pf_lst is not None:
        pg_lst, pg_tag = list(pf_lst), pf_lst[-1]
    elif mode == "gas_pressure":
        pg_lst, pg_tag = [0.0, pg], pg
    else:
        pg_lst, pg_tag = [0.0, 0.0], 0.0

    r0_str, pg_str = f"{r0:.6g}", f"{pg_tag:.6g}"
    result_basename = res_folder / f"{prefix}-{mode}-r0={r0_str}-pf={pg_str}-{probe}"
    mesh_filebasename = res_folder / f"mesh-{mode}-r0={r0_str}-{probe}"

    load_params = {"solid": {}, "liquid": {}, "air": {}}
    for i in range(2):
        for j in range(2):
            load_params["solid"][f"sigma_bar_{i}{j}"] = 0.0
    load_params["solid"].update(solid)
    n_steps = 2
    liq = load_params["liquid"]
    liq["pl_bar_ini_lst"] = [0.0] * n_steps
    liq["pl_bar_fin_lst"] = [0.0] * n_steps
    liq["grad_p_bar_x_ini_lst"], liq["grad_p_bar_x_fin_lst"] = gx_ini, gx_fin
    liq["grad_p_bar_y_ini_lst"], liq["grad_p_bar_y_fin_lst"] = gy_ini, gy_fin
    for k in ("Theta_in_ini_lst", "Theta_in_fin_lst", "Theta_out_ini_lst", "Theta_out_fin_lst"):
        liq[k] = [0.0] * n_steps
    load_params["air"]["pf"] = pg_tag
    load_params["air"]["pf_lst"] = pg_lst

    dmech.run_HollowBox_MicroPoroflow(
        dim=2,
        mesh_params=mesh_params(r0, l, mesh_filebasename),
        mat_params={k: {"parameters": dict(MATERIAL_PARAMETERS), "scaling": "no"} for k in ("skel", "bulk", "pore")},
        flow_params={"k_l": dolfin.Constant(((1e-6, 0.0), (0.0, 1e-6))), "use_kozeny_carman": False},
        porosity_params={"type": "constant", "val": 0.3},
        bcs="pbc",
        step_params={"n_steps": n_steps, "Deltat_lst": [1e-2, 1e-1], "dt_ini_lst": [1e-3, 1e-3],
                     "dt_min_lst": [1e-4, 1e-4], "dt_max_lst": [5e-3, 5e-3]},
        load_params=load_params,
        res_basename=str(result_basename),
        verbose=0)
    return result_basename


def run_case_list(run_case, cases, res_folder):
    """Run the cases one after the other (serially, in the given order).

    run_case : run_linear_hex_case or run_refined_case
    cases    : list of dicts; the keys "local" (bool) and "solver" select the local refinement (size
               r0 / 10 near the pores) and the LU backend; all other keys are passed to run_case.
    After every case the replaced global objects are checked to be the original ones again.
    """
    check_restored()
    n = len(cases)
    t_all = time.time()
    for i, case in enumerate(cases, start=1):
        kw = dict(case)
        local = kw.pop("local")
        solver = kw.pop("solver")
        label = ", ".join(f"{k}={v}" for k, v in case.items())
        print(f"[{i}/{n}] {label}", flush=True)
        t0 = time.time()
        with contextlib.ExitStack() as stack:
            if local:
                stack.enter_context(local_refinement(kw["r0"], kw["l"], size_near=kw["r0"] / 10.0))
            stack.enter_context(lu_solver(solver))
            run_case(res_folder=res_folder, **kw)
        check_restored()
        print(f"[{i}/{n}] done in {time.time() - t0:.1f} s", flush=True)
    print(f"{n} cases done in {(time.time() - t_all) / 60:.1f} min; results in {res_folder}")
