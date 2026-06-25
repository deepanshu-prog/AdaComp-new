#!/bin/bash
# AdaComp: Run calibration on the cluster to find optimal thresholds.
# This must NOT use LD_PRELOAD — it measures raw MPI directly.

PROJ_DIR="/home/itadmin/Desktop/hybrid_project"
LIB_PATH="$PROJ_DIR/SZ3/install/lib:$PROJ_DIR/zstd/lib"
NODES="node0,node1,node2"

echo "============================================"
echo "  AdaComp Calibration"
echo "  Measuring network + compression factors"
echo "============================================"

# Sync calibration binary
for node in node1 node2; do
    scp $PROJ_DIR/calibrate_adacomp itadmin@$node:$PROJ_DIR/
done

mpirun --prefix /usr -host $NODES -np 3 \
    -x LD_LIBRARY_PATH=$LIB_PATH \
    $PROJ_DIR/calibrate_adacomp

echo ""
if [ -f adacomp.conf ]; then
    echo "Configuration saved. Contents:"
    echo "--------------------------------------------"
    cat adacomp.conf
    echo "--------------------------------------------"
    echo "Copy adacomp.conf to your working directory"
    echo "before running experiments."
else
    echo "ERROR: Calibration did not produce adacomp.conf"
fi
