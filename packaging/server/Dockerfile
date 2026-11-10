# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR GPL-3.0-or-later
# (GPL too: the image it builds runs the GPL program piper-runner, and GPL-3.0 counts its build scripts in its source)
# `peresmeshnik serve` on a rented or own NVIDIA box, for yourself (not as a paid service: the licenses of the voice
# models are non-commercial). Only this file is published: no image (it would ship GPL, LGPL and the CPML weights) and
# no weights; you build it, and the models download at the first start, after their licenses are shown.
#
#   docker build -f packaging/server/Dockerfile -t peresmeshnik-server .
#   docker run --gpus all -v peresmeshnik:/data --tmpfs /data/data/Peresmeshnik/server/voices:uid=1000,mode=0700 \
#     --stop-timeout 60 -p 127.0.0.1:47000:47000 -it peresmeshnik-server serve --fetch en-ru
#
# The voices of a session live in memory only (--tmpfs): a kill, an out-of-memory or a reboot leaves none in the volume;
# the server removes them itself on stop, which may wait for a side still loading its models (--stop-timeout 60).
# The port is published on 127.0.0.1 only: reach it from the laptop through `ssh -tt -L 47000:127.0.0.1:47000 <box>
# sleep infinity` (with a terminal: `ssh -N` keeps Nagle on and adds up to a round trip to small messages, BENCH
# "Remote") and pair with `https://127.0.0.1:47000/#<code>` (the line `serve` prints names the container's own
# address); with Tailscale on the box publish on its tailnet address instead (`-p 100.x.y.z:47000:47000`) and pair with
# that. Everything that lasts is in the volume /data: the certificate and the paired devices (a new
# certificate would unpair every laptop), the models, the settings.
#
# Them -> me's MT runs on the GPU (CTranslate2 with the CUDA libraries of the base image), its ASR on the CPU (the
# app's onnxruntime is the CPU build); my cloned voice runs on the GPU in its own venv (the Linux voice pack: PyTorch
# with CUDA, every package pinned with its hash).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc
COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    XDG_CONFIG_HOME=/data/config XDG_DATA_HOME=/data/data XDG_CACHE_HOME=/data/cache \
    PERESMESHNIK_PIPER_CMD="/app/.venv/bin/python /app/piper-runner/piper_runner.py" \
    PERESMESHNIK_VOICE_CMD="/opt/clone/bin/python /app/voice-runner/voice_runner.py"
WORKDIR /app
COPY . /app
# the app (no Qt or sound device is touched by `serve`); the Piper runner's own packages beside it, hash-pinned, as the
# app builds do (packaging/piper-requirements.txt: no download at the first voice); the clone's own venv
RUN uv python install 3.12 \
 && uv sync --locked --no-dev --python 3.12 \
 && uv pip install -p /app/.venv/bin/python --no-deps --require-hashes -r packaging/piper-requirements.txt \
 && uv venv -p 3.12 /opt/clone \
 && uv pip install -p /opt/clone/bin/python --require-hashes -r voice-runner/requirements-linux-cuda.txt \
      --build-constraints voice-runner/requirements-linux-cuda.txt \
 && uv cache clean

# not root: the GPU works for any user of the container (NVIDIA Container Toolkit); a named volume takes /data's owner, a
# folder of the host mounted at /data must belong to uid 1000
RUN useradd --uid 1000 --create-home peresmeshnik && mkdir -p /data && chown peresmeshnik /data
USER peresmeshnik
VOLUME /data
EXPOSE 47000
# any `peresmeshnik` command: `serve --fetch en-ru` once (it shows the models' licenses: -it), then `serve`, which
# listens on the container's private address (the port published above) and prints the pairing code. The program
# itself is PID 1 (no `uv run` between): `docker stop`'s SIGTERM reaches it and the session's voices are removed
ENTRYPOINT ["/app/.venv/bin/peresmeshnik"]
CMD ["serve"]
