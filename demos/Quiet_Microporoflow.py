"""Quieter notebook output: known, harmless messages of the libraries, and paths printed relative to demos/.

Only what is printed is affected: no computation, parameter, mesh or plot setting is changed.

* import_pyvista(): imports pyvista without its two start-up messages (VTK cell-type errors, tqdm warning).
* quiet_fe_output(): hides the ufl FutureWarning and the gmsh information/progress lines of the
  finite-element runs.
* display_path(): path relative to the working directory (demos/), for printed messages only; the paths
  passed to dolfin_mech and to the plotting functions are not changed.
"""
import os
import warnings

# gmsh "General.Verbosity": 0 silent, 1 +errors, 2 +warnings, 3 +direct, 4 +information, 5 +status (gmsh default),
# 99 debug. 2 keeps the gmsh errors and warnings and drops the information lines ("Info : [ 10%] Meshing curve ...",
# "Info : Writing '<absolute path>.vtk'...").
GMSH_VERBOSITY = 2


def display_path(path):
    """Path as shown in printed messages: relative to the working directory (demos/)."""
    return os.path.relpath(path)


def import_pyvista():
    """Return the pyvista module, imported without its two start-up messages.

    - VTK: while pyvista is imported, VTK 9.2 writes (from C++, to stderr) one line
      "vtkGenericCell.cxx:439 ERR| vtkGenericCell (...): Unsupported cell type: 51 Setting to vtkEmptyCell"
      for each of the 16 cell type ids (51-56, 60-67, 74, 81) that have no vtkGenericCell implementation;
      nothing is wrong with the data. The VTK global warning display is switched off during the import only,
      and set back to its previous value afterwards, so later VTK errors are still shown.
    - tqdm: pyvista imports pooch, which does "from tqdm.auto import tqdm"; in a Jupyter kernel without
      ipywidgets, tqdm then warns "IProgress not found. Please update jupyter and ipywidgets." (TqdmWarning)
      and uses its text progress bar, which is what is used anyway. This warning is ignored during the import only.
    """
    from tqdm import TqdmWarning
    from vtkmodules.vtkCommonCore import vtkObject

    previous = vtkObject.GetGlobalWarningDisplay()
    vtkObject.GlobalWarningDisplayOff()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="IProgress not found", category=TqdmWarning)
            import pyvista
    finally:
        vtkObject.SetGlobalWarningDisplay(previous)
    return pyvista


def quiet_fe_output():
    """Hide two kinds of messages of the finite-element runs (dolfin_mech.run_HollowBox_MicroPoroflow).

    - ufl: ufl/exproperators.py:336 tests `arg in ("+", "-")` with a numpy array `arg`; numpy warns
      "FutureWarning: elementwise comparison failed; returning scalar instead" (the comparison returns False,
      which is the intended result). Ignored for this message, category and module only.
    - gmsh: dolfin_mech.run_HollowBox_Mesh calls gmsh.initialize() for every mesh, which resets all gmsh options,
      so General.Verbosity cannot be set beforehand. gmsh.initialize is replaced by a function that calls the
      original and then sets General.Verbosity = GMSH_VERBOSITY. The verbosity only controls messages, not the
      mesh. The replacement stays for the whole session (it is not restored), and is installed only once.
    """
    warnings.filterwarnings("ignore", message="elementwise comparison failed", category=FutureWarning,
                            module=r"ufl\.exproperators")

    import gmsh
    if getattr(gmsh.initialize, "_sets_gmsh_verbosity", False):
        return
    original = gmsh.initialize

    def initialize(*args, **kwargs):
        result = original(*args, **kwargs)
        gmsh.option.setNumber("General.Verbosity", GMSH_VERBOSITY)
        return result

    initialize._sets_gmsh_verbosity = True
    initialize.__wrapped__ = original
    gmsh.initialize = initialize
