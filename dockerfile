# Build separately; RunPod Pods do not require a nested Docker daemon.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:e711c99333fdfe8ae1e677b4972be6c5021f0128a1d31f775c7e58d88921b6a9
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/project
COPY . ./
# One filesystem-changing installation layer also supports vfs builders with
# limited disk; TLS trust is optional and supplied as a transient BuildKit secret.
RUN --mount=type=secret,id=build_ca,required=false \
    rm -f /etc/apt/sources.list.d/cuda* && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip git aspell aspell-en libaspell15 \
      libaspell-dev gcc python3.12-dev ca-certificates tini docker.io openssh-client && \
    rm -rf /var/lib/apt/lists/* && python3.12 -m venv /opt/illegibility && \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    /opt/illegibility/bin/python -m pip install --no-cache-dir --require-hashes -r requirements.lock && \
    /opt/illegibility/bin/python -m pip install --no-deps --no-build-isolation -e .
ENV PATH=/opt/illegibility/bin:$PATH
ARG SOURCE_CODE_COMMIT
ENV SOURCE_CODE_COMMIT=$SOURCE_CODE_COMMIT
ENV DATA_ROOT=/workspace/data ARTIFACT_ROOT=/workspace/artifacts MODEL_CACHE_ROOT=/workspace/model_cache
ENTRYPOINT ["/usr/bin/tini", "--", "illegibility"]
CMD ["hardware"]
