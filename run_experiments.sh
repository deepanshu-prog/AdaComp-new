#!/bin/bash

# Define paths
PROJ_DIR="/home/itadmin/Desktop/hybrid_project"
LIB_PATH="$PROJ_DIR/SZ3/install/lib"
INTERCEPTOR="$PROJ_DIR/libhybrid.so"
EXECUTABLE="$PROJ_DIR/test_mpi"
NODES="node0,node1,node2"

# Clear previous results
echo "M.Tech Project: Hybrid MPI Compression Results" > results.txt
echo "Date: $(date)" >> results.txt
echo "------------------------------------------------" >> results.txt

# Function to run experiment
run_test() {
    MODE=$1
    LABEL=$2
    echo "Running $LABEL..."
    echo "METHOD: $LABEL" >> results.txt
    mpirun --prefix /usr -host $NODES -np 3 \
        -x COMPRESSION_MODE=$MODE \
        -x LD_LIBRARY_PATH=$LIB_PATH \
        -x LD_PRELOAD=$INTERCEPTOR \
        $EXECUTABLE >> results.txt
    echo "------------------------------------------------" >> results.txt
}

# Run the 3 Methods
run_test 2 "BASELINE (RAW MPI)"
run_test 0 "LOSSLESS COMPRESSION"
run_test 1 "LOSSY COMPRESSION"

echo "All tests complete. Check results.txt for the comparison table."