#!/usr/bin/env bash
# Run a command on this host with an unlimited memlock limit, which XRT needs
# to map NPU buffers (the default 8 MB limit makes XRT fail with
# "mmap ... failed (err=-11)").
#
# Uses Docker only to lift the rlimit: the container chroots into the host's
# own filesystem and drops back to the calling user, so the command runs with
# the host's XRT, venv, and files.
#
# Not needed once the memlock limit is raised permanently (see README.md).
#
# Usage: ./npu-run.sh <command> [args...]
set -euo pipefail

if [ "$(ulimit -l)" = "unlimited" ]; then
  exec "$@"
fi

exec docker run --rm -i \
  --ulimit memlock=-1:-1 \
  --device /dev/accel/accel0 \
  -v /:/host \
  alpine chroot /host \
  setpriv --reuid="$(id -u)" --regid="$(id -g)" --init-groups \
  env -i HOME="$HOME" PATH="$PATH" TERM="${TERM:-dumb}" \
      PYTHONPATH="${PYTHONPATH:-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      VIRTUAL_ENV="${VIRTUAL_ENV:-}" \
  bash -c 'cd "$1" && shift && exec "$@"' _ "$PWD" "$@"
