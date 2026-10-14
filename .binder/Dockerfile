# Binder image: the environment of .repo2docker/ is built once and pushed to ghcr.io by
# .github/workflows/build-and-push-docker-image.yml; Binder only adds the repository content on top of it.
# TO UPDATE AFTER THE BUILD: replace the tag below by the tag of the image built from the current .repo2docker/.
# The tag below is the previous image (old environment, dolfin_mech c889cd6), not the current environment.
FROM ghcr.io/haotian-xiao/microporoflow-paper-demos:15df1db56b2a

ARG NB_USER=jovyan
ARG NB_UID=1000

USER ${NB_USER}

ENV HOME=/home/${NB_USER}

COPY --chown=${NB_UID}:${NB_UID} . ${HOME}

WORKDIR ${HOME}
