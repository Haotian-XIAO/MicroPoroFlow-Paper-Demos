# MicroPoroFlow Paper Demos

This repository contains the Jupyter notebooks used to generate Figures 3–10 of the MicroPoroFlow paper.

The numerical examples investigate the homogenized permeability of porous representative volume elements under liquid-pressure-gradient loading, macroscopic deformation, shear, and gas-pressure loading.

## Notebooks

| Figures | Notebook | Description |
|---|---|---|
| 3 | `demos/Figure3-Linear.ipynb` | Linear permeability homogenization |
| 4 | `demos/Figure4-Fluid-loading.ipynb` | Fluid-loading response |
| 5–7 | `demos/Figure5-7-Stress+Shear-effects.ipynb` | Uniaxial stretch and shear |
| 8 | `demos/Figure8-Gas-pressure-after-stretch.ipynb` | Gas pressure after stretch |
| 9 | `demos/Figure9-Gas-loading-modes.ipynb` | Gas pressure under different deformation modes |
| 10 | `demos/Figure10-Summary.ipynb` | Comparison of the main loading cases |

## Repository structure

- `demos/`: Jupyter notebooks
- `src/`: post-processing and plotting functions
- `results/`: locally generated simulation results, not tracked by Git

## Build the Jupyter Book

Install the documentation dependency:

```bash
python -m pip install -r requirements.txt
````

Build the book:

```bash
jupyter-book clean .
jupyter-book build .
```

The generated website is written to:

```text
_build/html/
```

The book uses the outputs already stored in the notebooks and does not rerun the full finite-element simulations during the documentation build.

## Authors

Haotian XIAO
Martin GENET

École Polytechnique, Palaiseau, France
