# MicroPoroFlow Paper Demos

This repository contains the Jupyter notebooks used to generate Figures 3–10 of the MicroPoroFlow paper.

Rendered notebooks are available at:

https://haotian-xiao.github.io/MicroPoroFlow-Paper-Demos/

## Notebooks

| Figures | Notebook                                         |
| ------- | ------------------------------------------------ |
| 3       | `demos/Figure3-Linear.ipynb`                     |
| 4       | `demos/Figure4-Fluid-loading.ipynb`              |
| 5–7     | `demos/Figure5-7-Stress+Shear-effects.ipynb`     |
| 8       | `demos/Figure8-Gas-pressure-after-stretch.ipynb` |
| 9       | `demos/Figure9-Gas-loading-modes.ipynb`          |
| 10      | `demos/Figure10-Summary.ipynb`                   |

## Run with Docker

Docker is the recommended option for reproducing the numerical results. Depending on the selected figure and the available hardware, a complete notebook run may take approximately 5–30 minutes on a standard laptop.

```bash
git clone https://github.com/Haotian-XIAO/MicroPoroFlow-Paper-Demos.git
cd MicroPoroFlow-Paper-Demos

docker build -t microporoflow:latest .

docker run --rm -it \
  -p 8888:8888 \
  -v "$(pwd)":/home/mambauser/project \
  microporoflow:latest \
  jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser
```

## Run with Conda

```bash
conda env create -f environment.yml
conda activate microporoflow
jupyter lab
```

## Authors

Haotian XIAO
Martin GENET

École Polytechnique, Palaiseau, France
