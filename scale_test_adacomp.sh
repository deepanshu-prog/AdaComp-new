#!/bin/bash
# AdaComp: Scalability sweep across message sizes.
# Tests all four modes and records CSV for plotting.

PROJ_DIR="/home/itadmin/Desktop/hybrid_project"
LIB_PATH="$PROJ_DIR/SZ3/install/lib:$PROJ_DIR/zstd/lib"
INTERCEPTOR="$PROJ_DIR/libadacomp.so"
EXECUTABLE="$PROJ_DIR/test_adacomp"
NODES="node0,node1,node2"

SIZES=(64 256 1024 4096 16384 65536 262144 1048576 4194304)
RUNS=3

CSV="adacomp_scale_results.csv"
echo "Size,Method,Time_us,Ratio,Accuracy" > $CSV

# Sync binaries
echo "Syncing binaries to cluster..."
for node in node1 node2; do
    scp $INTERCEPTOR $EXECUTABLE itadmin@$node:$PROJ_DIR/
done

run_test() {
    SIZE=$1; MODE=$2; LABEL=$3

    BEST_TIME=999999999
    BEST_RATIO="1.00"
    BEST_ACC="100.000000"

    for ((r=1; r<=RUNS; r++)); do
        OUT=$(mpirun --prefix /usr -host $NODES -np 3 \
            -x ADACOMP_MODE=$MODE \
            -x ADACOMP_VERBOSE=1 \
            -x LD_LIBRARY_PATH=$LIB_PATH \
            -x LD_PRELOAD=$INTERCEPTOR \
            $EXECUTABLE $SIZE allreduce 2>/dev/null)

        LINE=$(echo "$OUT" | grep "^ADACOMP,")
        if [ -n "$LINE" ]; then
            TIME=$(echo "$LINE" | cut -d',' -f5)
            RATIO=$(echo "$LINE" | cut -d',' -f6)
            ACC=$(echo "$LINE" | cut -d',' -f7)
            IS_BETTER=$(echo "$TIME < $BEST_TIME" | bc -l 2>/dev/null || echo 0)
            if [ "$IS_BETTER" = "1" ]; then
                BEST_TIME=$TIME
                BEST_RATIO=$RATIO
                BEST_ACC=$ACC
            fi
        fi
    done

    if [ "$BEST_TIME" != "999999999" ]; then
        echo "$SIZE,$LABEL,$BEST_TIME,$BEST_RATIO,$BEST_ACC" >> $CSV
        printf "  %-12s Size=%-8d Time=%-10s Ratio=%-6s Acc=%s%%\n" \
               "[$LABEL]" "$SIZE" "${BEST_TIME}us" "$BEST_RATIO" "$BEST_ACC"
    fi
}

echo ""
echo "============================================"
echo "  AdaComp Scalability Test"
echo "============================================"

for S in "${SIZES[@]}"; do
    echo ""
    echo "--- $S floats ($(( S * 4 / 1024 )) KB) ---"
    run_test $S 0 "Raw"
    run_test $S 1 "Lossless"
    run_test $S 2 "Lossy"
    run_test $S 3 "Adaptive"
    run_test $S 4 "Pipelined"
done

echo ""
echo "============================================"
echo "Results saved to $CSV"
echo "Generate plot: python3 plot_adacomp.py"
