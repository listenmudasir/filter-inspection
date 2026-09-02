#!/usr/bin/env bash
# Launch the NIRcam-first GUI with the hybrid EfficientAD backend.
#
#   ./run_gui.sh
#
# Set ENV_PY to use a different interpreter:
#   ENV_PY=/path/to/python ./run_gui.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PY="${ENV_PY:-/home/m100/miniconda3/envs/nircam/bin/python}"

if [[ ! -x "$ENV_PY" ]]; then
    echo "interpreter not found: $ENV_PY" >&2
    echo "create it with:  conda create -n nircam python=3.10 -y" >&2
    exit 1
fi

# Ignore ~/.local/lib/python3.10/site-packages.
#
# This machine has packages installed there with `pip install --user`, and
# user-site is on sys.path for EVERY python3.10 interpreter -- including a
# fresh conda env. A stray `anomalib` and a CUDA build of torch that the
# driver is too old for both live there, and they shadow the pinned versions
# we install into the env. Without this line the env's pins are decorative.
export PYTHONNOUSERSITE=1

# The Hikvision MVS runtime libraries.
export MVCAM_COMMON_RUNENV="${MVCAM_COMMON_RUNENV:-/opt/MVS/lib}"
_arch="$(uname -m)"
case "$_arch" in
    x86_64)  _mvs_lib=/opt/MVS/lib/64 ;;
    aarch64) _mvs_lib=/opt/MVS/lib/aarch64 ;;
    *)       _mvs_lib=/opt/MVS/lib/32 ;;
esac
# ---------------------------------------------------------------------------
# Qt environment hygiene. Do NOT simply inherit the caller's shell.
#
# Two contaminations were observed on this machine, both fatal and both from
# the interactive shell rather than from this script:
#
#   1. /opt/MVS/bin on LD_LIBRARY_PATH. The Hikvision SDK ships its own Qt5
#      (libQt5Core, libQt5Gui, libQt5XcbQpa ...). Those win over PyQt5's, and
#      PyQt5's libqxcb.so then fails with:
#         /opt/MVS/bin/libQt5XcbQpa.so.5: undefined symbol:
#         _ZN23QPlatformVulkanInstance22presentAboutToBeQueuedEP7QWindow
#      The camera libraries live in /opt/MVS/lib/{64,32} and do NOT need
#      /opt/MVS/bin, so it is filtered out entirely.
#
#   2. QT_QPA_PLATFORM_PLUGIN_PATH left pointing at a DIFFERENT conda env
#      (".../envs/waste/...") from an earlier session. Qt then loads that
#      env's plugins against this env's Qt and aborts.
#
# Both produce the same symptom -- "could not load the Qt platform plugin
# xcb" -- and neither is caused by anything in this repo, which is exactly why
# the launcher has to be explicit rather than trusting the environment.
# ---------------------------------------------------------------------------
_clean_ld=""
IFS=':' read -ra _parts <<< "${LD_LIBRARY_PATH:-}"
for _p in "${_parts[@]}"; do
    [[ -z "$_p" ]] && continue
    [[ "$_p" == /opt/MVS/bin* ]] && continue          # Qt5 conflict, see above
    _clean_ld="${_clean_ld:+$_clean_ld:}$_p"
done
export LD_LIBRARY_PATH="${_mvs_lib}:${_clean_ld}"

# Point Qt at THIS interpreter's PyQt5, whatever the caller had set.
unset QT_PLUGIN_PATH QT_DEBUG_PLUGINS
_qt_plugins="$("$ENV_PY" -c 'import PyQt5,os;print(os.path.join(os.path.dirname(PyQt5.__file__),"Qt5","plugins"))' 2>/dev/null || true)"
if [[ -d "$_qt_plugins/platforms" ]]; then
    export QT_QPA_PLATFORM_PLUGIN_PATH="$_qt_plugins/platforms"
else
    unset QT_QPA_PLATFORM_PLUGIN_PATH
fi

cd "$HERE"
exec "$ENV_PY" BasicDemo.py "$@"
