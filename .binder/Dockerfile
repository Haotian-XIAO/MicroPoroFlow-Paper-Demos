FROM ghcr.io/haotian-xiao/microporoflow-paper-demos:15df1db56b2a

ARG NB_USER=jovyan
ARG NB_UID=1000

USER ${NB_USER}

ENV HOME=/home/${NB_USER}

COPY --chown=${NB_UID}:${NB_UID} . ${HOME}

WORKDIR ${HOME}
