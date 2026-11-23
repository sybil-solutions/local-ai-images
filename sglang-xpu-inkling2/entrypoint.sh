#!/bin/bash
# Omarchy Local AI passes Intel GPUs as render nodes only. oneCCL (initialised even for one rank, and the TP2
# all-reduce) resolves devices through /dev/dri/by-path, which `docker --device` does not create: link each passed
# node under the name udev gives it, then run the recipe argv (python3 -m sglang.launch_server ...).
set -e
mkdir -p /dev/dri/by-path
for n in /dev/dri/renderD* /dev/dri/card[0-9]*; do
    [ -e "$n" ] || continue
    b=${n##*/}; case $b in render*) k=render;; *) k=card;; esac
    pci=$(basename "$(readlink -f "/sys/class/drm/$b/device")")
    ln -sf "../$b" "/dev/dri/by-path/pci-$pci-$k"
done
exec "$@"
