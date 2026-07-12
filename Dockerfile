# rPPG Heart Rate Monitor -- runtime image for the live dashboard.
#
# This packages the CHROM/POS/green/auto pipeline that the app actually serves.
# PhysNet training/eval (torch, onnx) is not exercised at runtime by the web app,
# but requirements.txt is used as-is (a deliberate Phase 0 decision -- see that
# file's own header comment) rather than maintaining a second, slimmed dependency
# list. The resulting image is large (heavy ML deps); see README for the honest
# size expectation.
FROM python:3.11-slim

# libgl1/libglib2.0-0: the same OpenCV/MediaPipe system libraries already
# validated in CI (.github/workflows/ci.yml) for headless Linux -- reusing a
# known-working combination rather than guessing at a new one.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first so the (large, slow) pip install layer is only
# rebuilt when dependencies actually change, not on every source edit.
COPY requirements.txt pyproject.toml README.md ./
# CPU-only torch wheels: the live app never touches PhysNet/CUDA (that's Phase 4,
# still not wired into the runtime), so the default PyPI wheel's bundled CUDA 12.1
# libraries (several GB of nvidia-cublas/cudnn/cufft/etc.) are pure waste here.
# Same fix CI already applies for the same reason (.github/workflows/ci.yml).
# --timeout/--retries: default pip settings can spuriously time out on a slow
# connection for these multi-hundred-MB downloads.
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
    torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

# Only what the live app needs to run -- datasets/, results/, scripts/, tests/
# are intentionally left out of the image (see .dockerignore).
COPY src/ src/
COPY models/ models/

# Container-appropriate override of the app's own secure-by-default 127.0.0.1
# (see src/logging_config.py-style env var convention and Phase 3's hardening):
# inside a container, Docker's `-p host:container` port publishing is the real
# exposure boundary, so binding to all interfaces *inside* the container is the
# standard, expected pattern -- this does not change the bare-metal default.
ENV RPPG_HOST=0.0.0.0
ENV RPPG_PORT=8000

EXPOSE 8000

# Goes through main.py's real __main__ entrypoint, so RPPG_PORT/RPPG_CORS_ORIGINS
# overrides are honored -- not a bare `uvicorn ...` invocation that would ignore them.
CMD ["python", "-m", "src.app.main"]
