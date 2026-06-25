#include <mpi.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>
#include <cmath>

extern "C" {
    #include "sz3c.h"
}

static int (*real_allreduce)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm) = NULL;

// Threshold for Switching (Mode 1)
const int SWITCH_THRESHOLD = 20480; // 20k floats (~80KB)

extern "C" int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    
    if (!real_allreduce) {
        real_allreduce = (int (*)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm))dlsym(RTLD_NEXT, "MPI_Allreduce");
    }

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Get Mode: 1 = With Switching, 2 = Always Lossy
    char* mode_env = getenv("COMPRESSION_MODE");
    int mode = (mode_env != NULL) ? atoi(mode_env) : 1; 

    bool perform_compression = true;

    // --- LOGIC FOR METHOD 1: WITH SWITCHING ---
    if (mode == 1) {
        if (count < SWITCH_THRESHOLD || datatype != MPI_FLOAT) {
            perform_compression = false;
        }
    }
    // --- LOGIC FOR METHOD 2: ALWAYS LOSSY ---
    else if (mode == 2) {
        perform_compression = (datatype == MPI_FLOAT);
    }

    // --- EXECUTION PATH A: BASELINE (IF SWITCHED OFF) ---
    if (!perform_compression) {
        double t_start = MPI_Wtime();
        int res = real_allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        double t_end = MPI_Wtime();
        if (rank == 0 && datatype == MPI_FLOAT) {
            std::cout << "BASELINE_USED," << count << "," << (t_end - t_start)*1e6 << ",1.0,100.0" << std::endl;
        }
        return res;
    }

    // --- EXECUTION PATH B: LOSSY COMPRESSION (SZ3) ---
    double t_start = MPI_Wtime();
    const double absBound = 1e-4;
    size_t outSize = 0;

    // 1. Compress
    unsigned char* compressed = SZ_compress_args(0, (void*)sendbuf, &outSize, 1, absBound, 0, 0, 0, 0, 0, 0, (size_t)count);

    // 2. Metadata Exchange
    std::vector<long> all_sizes(size);
    long my_size = (long)outSize;
    MPI_Allgather(&my_size, 1, MPI_LONG, all_sizes.data(), 1, MPI_LONG, comm);

    // 3. Network Transfer
    std::vector<int> counts(size), displs(size);
    int total_bytes = 0;
    for(int i=0; i<size; i++) {
        counts[i] = (int)all_sizes[i];
        displs[i] = total_bytes;
        total_bytes += counts[i];
    }
    std::vector<unsigned char> global_buf(total_bytes);
    MPI_Allgatherv(compressed, (int)outSize, MPI_BYTE, global_buf.data(), counts.data(), displs.data(), MPI_BYTE, comm);

    // 4. Decompress and Reduce
    float* result_buf = (float*)recvbuf;
    memset(result_buf, 0, count * sizeof(float));
    double max_err = 0;

    for(int i=0; i<size; i++) {
        float* dec = (float*)SZ_decompress(0, &global_buf[displs[i]], counts[i], 0, 0, 0, 0, (size_t)count);
        
        // Error validation (local rank only)
        if (i == rank) {
            float* orig = (float*)sendbuf;
            for(int j=0; j<count; j++) {
                double diff = std::abs((double)orig[j] - (double)dec[j]);
                if (diff > max_err) max_err = diff;
            }
        }
        for(int j=0; j<count; j++) result_buf[j] += dec[j];
        free(dec);
    }

    double t_end = MPI_Wtime();
    if (rank == 0) {
        double ratio = (double)(count * 4) / outSize;
        double acc = 100.0 * (1.0 - max_err);
        std::cout << "LOSSY_USED," << count << "," << (t_end - t_start)*1e6 << "," 
                  << std::fixed << std::setprecision(2) << ratio << "," 
                  << std::fixed << std::setprecision(6) << acc << std::endl;
    }

    free(compressed);
    return MPI_SUCCESS;
}