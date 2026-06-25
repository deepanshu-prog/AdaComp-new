#!/bin/bash
# AdaComp: Run all experiments — baseline, lossless, lossy, and adaptive.

PROJ_DIR="/home/itadmin/Desktop/hybrid_project"
LIB_PATH="$PROJ_DIR/SZ3/install/lib:$PROJ_DIR/zstd/lib"
INTERCEPTOR="$PROJ_DIR/libadacomp.so"
EXECUTABLE="$PROJ_DIR/test_adacomp"
NODES="node0,node1,node2"
COUNT=${1:-1048576}

# Sync binaries
for node in node1 node2; do
    scp $INTERCEPTOR $EXECUTABLE itadmin@$node:$PROJ_DIR/
done

RESULTS="adacomp_results.txt"

echo "============================================"  | tee $RESULTS
echo "  AdaComp: Adaptive Multi-Tier Compression"   | tee -a $RESULTS
echo "  Middleware for MPI Collectives"              | tee -a $RESULTS
echo "============================================"  | tee -a $RESULTS
echo "Date: $(date)"                                 | tee -a $RESULTS
echo "Message size: $COUNT floats"                   | tee -a $RESULTS
echo ""                                              | tee -a $RESULTS

run_mode() {
    MODE=$1; LABEL=$2; COLLECTIVE=$3
    echo "--- [$LABEL] $COLLECTIVE ---"              | tee -a $RESULTS
    mpirun --prefix /usr -host $NODES -np 3 \
        -x ADACOMP_MODE=$MODE \
        -x ADACOMP_VERBOSE=1 \
        -x LD_LIBRARY_PATH=$LIB_PATH \
        -x LD_PRELOAD=$INTERCEPTOR \
        $EXECUTABLE $COUNT $COLLECTIVE              | tee -a $RESULTS
    echo ""                                          | tee -a $RESULTS
}

echo "=== MPI_Allreduce ==="                         | tee -a $RESULTS
run_mode 0 "Baseline (Raw MPI)"          allreduce
run_mode 1 "Lossless Only (zstd)"        allreduce
run_mode 2 "Lossy Only (SZ3)"            allreduce
run_mode 3 "Adaptive Multi-Tier"         allreduce
run_mode 4 "Adaptive+Pipelined"          allreduce

echo "=== MPI_Bcast ==="                             | tee -a $RESULTS
run_mode 0 "Baseline (Raw MPI)"      bcast
run_mode 3 "Adaptive Multi-Tier"     bcast

echo "=== MPI_Reduce ==="                            | tee -a $RESULTS
run_mode 0 "Baseline (Raw MPI)"      reduce
run_mode 3 "Adaptive Multi-Tier"     reduce

echo "============================================"  | tee -a $RESULTS
echo "Results saved to $RESULTS"
