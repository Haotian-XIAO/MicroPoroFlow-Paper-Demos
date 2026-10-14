# Welcome to [Xiao & Genet (2026). Effective permeability of double-scale porous media under finite deformation: a three-phase micro-poro-mechanics framework. In preparation.]'s demos!

Static and interactive demos can be found at [https://haotian-xiao.github.io/MicroPoroFlow-Paper-Demos](https://haotian-xiao.github.io/MicroPoroFlow-Paper-Demos/), or directly on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Haotian-XIAO/MicroPoroFlow-Paper-Demos/main?urlpath=lab/tree/./demos).

## Notebooks

All notebooks are in `demos/`; each one writes its results and figures to `demos/<notebook name>/`.
Running times were measured on a laptop (Apple M3, one thread per notebook); "≈" marks estimates.

| Notebook | Paper | Running time |
|---|---|---|
| `Fig3.ipynb` | Fig. 3; Sec. 3.1 (comparison with the Hashin–Shtrikman bound) | 3–5 min |
| `Fig4.ipynb` | Fig. 4 | 20–25 min |
| `Fig5-6-7.ipynb` | Figs. 5, 6 and 7 | 20–25 min |
| `Fig8.ipynb` | Fig. 8 | 15 min |
| `Fig9.ipynb` | Fig. 9 | 5–8 min |
| `Fig10.ipynb` | Fig. 10; abstract, introduction and Sec. 3.5 (state-function test) | 1–1.5 h |
| `Fig11-kappa0.ipynb` | Fig. 11: permeability κ0 of the undeformed unit cells (default and refined meshes) | ≈ 1 h |
| `Fig11-refined.ipynb` | Fig. 11(b): symmetric loadings on the refined mesh; Sec. 2.4 (mesh convergence) | ≈ 1.5–2 h |
| `Fig11.ipynb` | Fig. 11; abstract and Sec. 3.5.2 (numbers quoted in the text) | < 1 min |

`Fig11.ipynb` only post-processes the results of other notebooks, which must be run first: `Fig9.ipynb`, `Fig10.ipynb`, `Fig11-kappa0.ipynb` and `Fig11-refined.ipynb` (the mesh-convergence section of `Fig11-refined.ipynb` also uses the results of `Fig10.ipynb`).
The other notebooks are independent.
In each notebook, `do_*` switches in the "Parameters" section separate the finite-element computations from the post-processing.

`Fig10.ipynb`, `Fig11-kappa0.ipynb` and `Fig11-refined.ipynb` are computationally intensive: Binder may run out of memory or time out, but you can run them locally, using Docker or a conda installation (see below).

The finite-element computations use [dolfin_mech](https://github.com/mgenet/dolfin_mech) from the fork [Haotian-XIAO/dolfin_mech_HX](https://github.com/Haotian-XIAO/dolfin_mech_HX), at commit [`aace2bf`](https://github.com/Haotian-XIAO/dolfin_mech_HX/commit/aace2bfc1b5f2841589f58749b486a4694c10c01): it contains the fix of the simple-shear loading (the macroscopic symmetry penalty is skipped when all off-diagonal components of the macroscopic displacement gradient are prescribed) used for the results of the paper.

## Local run via Docker

Because these demos are computationally intensive, Binder may run out of memory or time out, but you can run them locally using Docker:
- Ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your machine.
- Pull the Image:
```bash
docker pull ghcr.io/haotian-xiao/microporoflow-paper-demos:latest
```
- Start the Container:
```bash
docker run -p 8888:8888 ghcr.io/haotian-xiao/microporoflow-paper-demos:latest
```
- Access the Notebooks: Look at your terminal output for a URL that starts with `http://127.0.0.1:8888/?token=...`, copy and paste that entire link into your web browser to access the demos.

Note: the image of the current environment (`.repo2docker/`) has not been built yet; until it is, use the system install below.

## Local run via system install

You can also run the demos locally by setting up your system:
- Ensure you have [Miniconda](https://docs.anaconda.com/free/miniconda) installed on your system, as well as [git](https://git-scm.com/install).
- Clone the Repository:
```bash
git clone https://github.com/Haotian-XIAO/MicroPoroFlow-Paper-Demos.git
cd MicroPoroFlow-Paper-Demos
```
- Create the conda environment, then install the packages that come from pip (the same commands as in `.repo2docker/postBuild`; do not run `postBuild` itself, it is written for the Docker image):
```bash
conda env create -n microporoflow -f .repo2docker/environment.yml
conda activate microporoflow
python -m pip install --no-deps "scipy==1.15.3" "gmsh==4.13.1" "myPythonLibrary==2025.4.17" "myVTKPythonLibrary==2025.4.18" "vtkpython-cbl==2025.4.17"
python -m pip install --no-deps "git+https://github.com/Haotian-XIAO/dolfin_mech_HX.git@aace2bfc1b5f2841589f58749b486a4694c10c01"
```
On Linux, the gmsh package also needs the system libraries listed in `.repo2docker/apt.txt` (e.g., `sudo apt install $(grep -v '^#' .repo2docker/apt.txt)` on Debian/Ubuntu).
This only needs to be done once. Now, every time you want to run the demos, you need to:
- Activate the environment:
```bash
conda activate microporoflow
```
- Launch Jupyter:
```bash
jupyter notebook
```
Your default web browser will automatically open.
From there, navigate into the demos/ folder and open the notebooks to get started!
