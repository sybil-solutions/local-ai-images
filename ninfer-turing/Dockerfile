# syntax=docker/dockerfile:1.7
# NInfer (iamwavecut/ninfer-all) for the local-ai registry: the upstream Dockerfile at the pinned commit, built
# for sm_75 (Turing) only. Same toolchain, build script, staging and runtime as upstream; the source is fetched
# by commit instead of COPY, and both base images are pinned by digest. Upstream CI published this commit's
# all-architecture build as ghcr.io/iamwavecut/ninfer-all:sha-d1a0e0704144; this image carries its sm75 build
# at the same path, /opt/ninfer/sm75/ninfer-serve.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance,
# an SBOM, and a GitHub build attestation on the digest.
FROM nvidia/cuda:13.4.2-devel-ubuntu26.04@sha256:14fd494fd8496817a70700e4d2c52bd42de56c5d39c041bf8342814614d5fe56 AS build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates cmake file git libavcodec-dev libavformat-dev libavutil-dev libcurl4-openssl-dev \
        libswscale-dev ninja-build pkg-config python3 python3-jinja2 rsync zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
ARG NINFER_COMMIT=d1a0e070414414af3f1e4b2c60cc2ff74a4ddb9b
WORKDIR /src
RUN git init -q . && git remote add origin https://github.com/iamwavecut/ninfer-all.git \
    && git fetch -q --depth 1 origin "$NINFER_COMMIT" && git -c advice.detachedHead=false checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "$NINFER_COMMIT"
ARG ARCHS=75
ARG JOBS=""
RUN export NINFER_TESTS=OFF NINFER_MULTICALL=ON \
    && if [ -n "$JOBS" ]; then export NINFER_JOBS="$JOBS"; fi \
    && for arch in $ARCHS; do \
         NINFER_CUDA_ARCH="$arch" NINFER_BUILD_DIR="/build/sm$arch" scripts/build-native.sh configure \
         && NINFER_BUILD_DIR="/build/sm$arch" scripts/build-native.sh target ninfer-multicall \
         && docker/stage-dist.sh "/build/sm$arch" "$arch" /dist || exit 1; \
       done

FROM ubuntu:26.04@sha256:c098726dd9b9733f3e885f8de0ef375206a311b0a317eadc40d347b1115fb944
ARG DEBIAN_FRONTEND=noninteractive
COPY --from=build /dist/packages/ /tmp/ninfer-packages/
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl \
    && curl -fsSLo /tmp/cuda-keyring.deb \
       https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && sort -u /tmp/ninfer-packages/*.txt | xargs apt-get install --yes --no-install-recommends aria2 \
    && find /usr/local -name 'libcublas.so.*' -printf '%h\n' | sort -u > /etc/ld.so.conf.d/ninfer-cuda.conf \
    && ldconfig \
    && apt-get purge --yes cuda-keyring \
    && rm -rf /var/lib/apt/lists/* /tmp/ninfer-packages /tmp/cuda-keyring.deb
COPY --from=build --chmod=0755 /dist/bin/ /opt/ninfer/
COPY --from=build /src/scripts/run.sh /src/scripts/download-model.sh /opt/ninfer/
COPY --from=build /src/docker/entrypoint.sh /usr/local/bin/ninfer-entrypoint
RUN for build in /opt/ninfer/sm*; do \
      for name in ninfer ninfer-serve ninfer-calibrate ninfer-perplexity ninfer-ngram-profile; do \
        ln -s ninfer-multicall "$build/$name"; \
      done; \
    done
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0" \
    NINFER_MODEL_DIR=/models \
    NINFER_DEVICE_PROFILES=/cache/device-profiles.json \
    NINFER_HOST=0.0.0.0 \
    NINFER_PORT=8080
VOLUME ["/models", "/grafts", "/cache"]
WORKDIR /models
EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["ninfer-entrypoint"]
CMD ["run", "qwen38-27b"]
