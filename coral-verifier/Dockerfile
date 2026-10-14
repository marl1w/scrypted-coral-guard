# Coral Verifier: Debian bookworm (Python 3.11) so the tflite-runtime 2.17.1 cp311 wheel and libedgetpu 16.0tf2.17.1 (a known working pair)
# match. Hardcoded base: no build.yaml needed.
# Base pinned by digest (debian:bookworm-slim, amd64 manifest 2026-10): rebuilds are reproducible; bump deliberately.
FROM debian:bookworm-slim@sha256:a4672c0cb26fbdde88e38fa2dfb6c681942306680e41e4378b28770b6e79ee91 AS base
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 OPENCV_IO_MAX_IMAGE_PIXELS=16777216

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl python3 python3-venv libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Edge TPU runtime, standard (not max) frequency: cooler, safer on a USB stick. feranick/libedgetpu release 16.0TF2.17.1-1 (bookworm amd64).
# The -std deb's sha256 was computed by us on download (the release publishes none); the sibling -max deb's hash matches the published upstream value.
ARG EDGETPU_DEB=libedgetpu1-std_16.0tf2.17.1-1.bookworm_amd64.deb
ARG EDGETPU_SHA=a3e1f9ae3e4725032a863ea38f52aa396ee18fdb65135fed56e8efae5c2e467d
RUN curl -fsSL -o /tmp/edgetpu.deb "https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-1/${EDGETPU_DEB}" \
    && echo "${EDGETPU_SHA}  /tmp/edgetpu.deb" | sha256sum -c - \
    && apt-get update && apt-get install -y --no-install-recommends /tmp/edgetpu.deb && rm -rf /var/lib/apt/lists/* /tmp/edgetpu.deb

# Python: tflite_runtime wheel from frigate-nvr/TFlite-builds v2.17.1 (cp311, x86_64) + pinned PyPI packages (no pillow: JPEG decode is cv2)
ARG TFLITE_WHL=tflite_runtime-2.17.1-cp311-cp311-linux_x86_64.whl
ARG TFLITE_SHA=a6b4e57f13c536a737aeb0ebd27618177ddbbabb1e4242c9c3403cd566fe71b6
RUN python3 -m venv /opt/venv \
    && curl -fsSL -o /tmp/${TFLITE_WHL} "https://github.com/frigate-nvr/TFlite-builds/releases/download/v2.17.1/${TFLITE_WHL}" \
    && echo "${TFLITE_SHA}  /tmp/${TFLITE_WHL}" | sha256sum -c - \
    && /opt/venv/bin/pip install /tmp/${TFLITE_WHL} numpy==1.26.4 opencv-python-headless==4.11.0.86 \
    && rm -f /tmp/${TFLITE_WHL}
ENV PATH=/opt/venv/bin:$PATH

# Models are NOT in the repo: fetched from dbro/frigate-detector-edgetpu-yolo9 releases (sha256 computed on download and pinned).
ARG M320_URL=https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v1.0/yolov9-s-relu6-best_320_int8_edgetpu.tflite
ARG M320_SHA=49a38e77439b475d0d259a6d9fe7ea32fcaa58de3748499dd120aceebad9c591
ARG M512_URL=https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v1.5/yolov9-s-relu6-tpumax_512_int8_edgetpu.tflite
ARG M512_SHA=f634f60259101556825fd69662f96282e74364b507f83e6382cc34d5d0e6170e
RUN mkdir -p /models \
    && curl -fsSL -o /models/yolov9-s-relu6-best_320_int8_edgetpu.tflite "${M320_URL}" && echo "${M320_SHA}  /models/yolov9-s-relu6-best_320_int8_edgetpu.tflite" | sha256sum -c - \
    && curl -fsSL -o /models/yolov9-s-relu6-tpumax_512_int8_edgetpu.tflite "${M512_URL}" && echo "${M512_SHA}  /models/yolov9-s-relu6-tpumax_512_int8_edgetpu.tflite" | sha256sum -c - \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Test stage (not built by Supervisor, which builds the last stage): docker build --target test ... ; it alone carries fake_tpu.py and the tests.
FROM base AS test
RUN pip install pytest==8.3.5
COPY verifier /app
COPY config.yaml /app/config.yaml
COPY tests /tests
ENV VERIFIER_FAKE_TPU=1
WORKDIR /tests
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "/tests"]

# Runtime stage: service code WITHOUT fake_tpu.py (no test hooks in the image that talks to the network).
FROM base AS runtime
ARG BUILD_VERSION
ENV BUILD_VERSION=${BUILD_VERSION}
COPY verifier /app
COPY config.yaml /app/config.yaml
RUN rm -f /app/fake_tpu.py
COPY run.sh /run.sh
RUN chmod +x /run.sh
CMD ["/run.sh"]
