#!/bin/bash

# Configuration
PROJ_DIR="/home/itadmin/Desktop/hybrid_project"
LIB_PATH="$PROJ_DIR/SZ3/install/lib"
INTERCEPTOR="$PROJ_DIR/libhybrid.so"
EXECUTABLE="$PROJ_DIR/test_mpi"
NODES="node0,node1,node2"

# Data sizes to test (Number of floats)
# We test from very small (to show switching) to very large (to show lossy efficiency)
SIZES=(1024 10240 102400 512000 1048576 2097152)

# Syncing binaries to all nodes before starting
echo "Synchronizing binaries to cluster..."
for node in node1 node2; do
    scp $INTERCEPTOR itadmin@$node:$PROJ_DIR/
    scp $EXECUTABLE itadmin@$node:$PROJ_DIR/
done

# Prepare CSV header
echo "Size,Method,Time_us,Ratio,Accuracy" > scale_results.csv

run_iteration() {
    SIZE=$1
    MODE=$2 # 1: With Switching, 2: Always Lossy
    LABEL=$3

    echo "------------------------------------------------"
    echo "Testing $LABEL | Size: $SIZE floats"
    
    # Run the MPI job
    # We pass COMPRESSION_MODE to libhybrid.so
    OUT=$(mpirun --prefix /usr -host $NODES -np 3 \
        -x COMPRESSION_MODE=$MODE \
        -x LD_LIBRARY_PATH=$LIB_PATH \
        -x LD_PRELOAD=$INTERCEPTOR \
        $EXECUTABLE $SIZE)

    # Extracting Data
    # Note: Our libhybrid.cpp now prints BASELINE_USED or LOSSY_USED
    # We grab the line regardless of which one was triggered
    STAT_LINE=$(echo "$OUT" | grep -E "BASELINE_USED|LOSSY_USED")
    
    if [ -z "$STAT_LINE" ]; then
        echo "Error: No stats found in output for $LABEL at size $SIZE"
    else
        # Extract columns from the CSV-style output in libhybrid.cpp
        TIME=$(echo "$STAT_LINE" | cut -d',' -f3)
        RATIO=$(echo "$STAT_LINE" | cut -d',' -f4)
        ACC=$(echo "$STAT_LINE" | cut -d',' -f5)
        
        # Save to our main results file
        echo "$SIZE,$LABEL,$TIME,$RATIO,$ACC" >> scale_results.csv
        echo "Result: $TIME us | Ratio: $RATIO | Acc: $ACC%"
    fi
}

# Execution Loop
for S in "${SIZES[@]}"; do
    # Method 1: Hybrid Switching (The Innovation)
    run_iteration $S 1 "Hybrid_Switching"
    
    # Method 2: Always Lossy (The Base Paper approach)
    run_iteration $S 2 "Always_Lossy"
done

echo "------------------------------------------------"
echo "Scalability test complete. Results saved to scale_results.csv"