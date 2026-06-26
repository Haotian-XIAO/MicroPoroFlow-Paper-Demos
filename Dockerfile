FROM mambaorg/micromamba:2.8.1

# Copy the Conda environment specification.
COPY --chown=$MAMBA_USER:$MAMBA_USER \
    environment.yml /tmp/environment.yml

# Remove the dolfin_mech Git dependency temporarily.
# Its dependencies are installed first through Conda and pip.
RUN grep -v \
        'git+https://github.com/Haotian-XIAO/dolfin_mech_HX.git@' \
        /tmp/environment.yml \
        > /tmp/environment-docker.yml \
    && micromamba install \
        --yes \
        --name base \
        --file /tmp/environment-docker.yml \
    && micromamba install \
        --yes \
        --name base \
        python-gmsh \
    && micromamba clean --all --yes

# Activate the micromamba base environment for subsequent RUN commands.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Verify that the Gmsh Python interface is available.
RUN python -c \
    "import gmsh; print('gmsh:', gmsh.__file__)"

# Install the exact dolfin_mech revision without re-installing its dependencies.
RUN python -m pip install --no-deps \
    "git+https://github.com/Haotian-XIAO/dolfin_mech_HX.git@c889cd67e3dd88ee723978e780f6a79e0b366d99"

# Copy the paper demonstration repository.
WORKDIR /home/mambauser/project

COPY --chown=$MAMBA_USER:$MAMBA_USER \
    . /home/mambauser/project

# Final installation test.
RUN python -c \
    "import dolfin, dolfin_mech, gmsh, myPythonLibrary, numpy, scipy, matplotlib, matplotlib_inline, meshio; \
    print('DOLFIN:', dolfin.__version__); \
    print('dolfin_mech:', dolfin_mech.__file__); \
    print('gmsh:', gmsh.__file__); \
    print('NumPy:', numpy.__version__); \
    print('SciPy:', scipy.__version__); \
    print('matplotlib-inline:', matplotlib_inline.__version__)"

CMD ["bash"]
