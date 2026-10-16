# exl3xpu: EXL3 inference on Intel Arc (Battlemage) via vLLM XPU.
FROM intel/llm-scaler-vllm:0.26.0-b2@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ARG TRELLIS_SERVE_COMMIT=c7c883729d72db9220d1085015798ab490609d7c
RUN git init -q /opt/trellis-serve && cd /opt/trellis-serve \
 && git fetch -q --depth 1 https://github.com/sybil-solutions/trellis-serve "${TRELLIS_SERVE_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${TRELLIS_SERVE_COMMIT}" \
 && echo "${TRELLIS_SERVE_COMMIT}" > COMMIT && rm -rf .git \
 && cp -a core /opt/trellis-core \
 && mkdir /opt/exl3xpu \
 && cp -a xpu/pyproject.toml xpu/exl3xpu xpu/csrc xpu/scripts xpu/models xpu/bench xpu/tests /opt/exl3xpu/
LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.trellis-serve.commit="${TRELLIS_SERVE_COMMIT}"
RUN pip install --no-deps --no-build-isolation -e /opt/trellis-core
WORKDIR /opt/exl3xpu
RUN bash scripts/build_ext.sh && pip install --no-deps --no-build-isolation -e . \
 && python3 -c "from importlib.metadata import entry_points as e; assert any(x.name=='exl3xpu' for x in e(group='vllm.general_plugins'))"
ENV MODELS_DIR=/models
ENTRYPOINT ["python3", "scripts/serve.py"]
CMD ["models/qwen3.8-27b-exl3-4.00bpw", "--gpu", "0"]
