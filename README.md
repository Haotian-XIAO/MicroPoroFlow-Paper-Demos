# MicroPoroFlow Paper Demos

Welcome to the numerical demos accompanying the MicroPoroFlow paper by Haotian XIAO and Martin GENET.

Rendered notebooks can be viewed online at:

https://haotian-xiao.github.io/MicroPoroFlow-Paper-Demos/

This repository contains the Jupyter notebooks used to generate Figures 3–10 of the paper. The numerical examples investigate the homogenized permeability of porous representative volume elements under liquid-pressure-gradient loading, macroscopic deformation, shear, and gas-pressure loading.

## Notebooks

| Figures | Notebook                                         | Description                                            |
| ------- | ------------------------------------------------ | ------------------------------------------------------ |
| 3       | `demos/Figure3-Linear.ipynb`                     | Linear permeability homogenization                     |
| 4       | `demos/Figure4-Fluid-loading.ipynb`              | Fluid-loading response                                 |
| 5–7     | `demos/Figure5-7-Stress+Shear-effects.ipynb`     | Uniaxial loading and simple shear                      |
| 8       | `demos/Figure8-Gas-pressure-after-stretch.ipynb` | Gas-pressure loading after uniaxial stretch            |
| 9       | `demos/Figure9-Gas-loading-modes.ipynb`          | Gas-pressure effects under different deformation modes |
| 10      | `demos/Figure10-Summary.ipynb`                   | Comparison of the main loading cases                   |

## Repository structure

* `demos/`: Jupyter notebooks used to generate the paper figures
* `src/`: post-processing and plotting functions
* `results/`: locally generated numerical results, not tracked by Git
* `environment.yml`: reproducible Conda environment
* `Dockerfile`: reproducible Docker environment
* `_config.yml` and `_toc.yml`: Jupyter Book configuration

## Local run via Docker

Because some of the demonstrations are computationally intensive, running them locally with Docker is the recommended option.

### 1. Install Docker

Install Docker Desktop and make sure that the Docker engine is running.

### 2. Clone the repository

```bash
git clone https://github.com/Haotian-XIAO/MicroPoroFlow-Paper-Demos.git
cd MicroPoroFlow-Paper-Demos
```

### 3. Build the Docker image

```bash
docker build \
  --tag microporoflow:latest \
  .
```

The first build may take several minutes because FEniCS, PETSc, VTK, Gmsh, and the other numerical dependencies must be installed.

### 4. Start JupyterLab

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$(pwd)":/home/mambauser/project \
  microporoflow:latest \
  jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser
```

The terminal will display a URL beginning with:

```text
http://127.0.0.1:8888/lab?token=...
```

Copy the complete URL into a web browser. The notebooks are located in the `demos/` directory.

The repository is mounted inside the container, so generated results and modified notebooks remain available on the host machine after the container is stopped.

### 5. Stop JupyterLab

Press:

```text
Ctrl + C
```

in the terminal running the container.

## Local run via Conda

The demos can also be run through a local Conda installation. This setup depends more strongly on the operating system and the availability of platform-specific FEniCS and Gmsh packages.


### 1. Install prerequisites

Install Miniconda or Anaconda, together with Git.

### 2. Clone the repository

```bash
git clone https://github.com/Haotian-XIAO/MicroPoroFlow-Paper-Demos.git
cd MicroPoroFlow-Paper-Demos
```

### 3. Create the environment

```bash
conda env create -f environment.yml
```

This step only needs to be performed once.

### 4. Activate the environment

```bash
conda activate microporoflow
```

### 5. Launch JupyterLab

```bash
jupyter lab
```

Navigate to the `demos/` directory and open the desired notebook.

## Build the Jupyter Book locally

Install the documentation dependency:

```bash
python -m pip install -r requirements.txt
```

Build the book:

```bash
jupyter-book clean .
jupyter-book build .
```

The generated website is written to:

```text
_build/html/
```

Open it on macOS with:

```bash
open _build/html/index.html
```

## Authors

Haotian XIAO
Martin GENET

École Polytechnique, Palaiseau, France
