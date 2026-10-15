# Binder image: the environment of .repo2docker/ is built once and pushed to ghcr.io by
# .github/workflows/build-and-push-docker-image.yml; Binder only adds the repository content on top of it.
FROM ghcr.io/haotian-xiao/microporoflow-paper-demos:c2265396c596

ARG NB_USER=jovyan
ARG NB_UID=1000

USER ${NB_USER}

ENV HOME=/home/${NB_USER}

COPY --chown=${NB_UID}:${NB_UID} . ${HOME}

WORKDIR ${HOME}
