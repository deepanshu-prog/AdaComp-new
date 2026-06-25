// AdaComp Test Harness
// Exercises MPI_Allreduce, MPI_Bcast, and MPI_Reduce with verification.
// Usage: mpirun ... test_adacomp [count] [allreduce|bcast|reduce]

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int count = (argc > 1) ? atoi(argv[1]) : 1048576;
    const char* test = (argc > 2) ? argv[2] : "allreduce";

    float* send = new float[count];
    float* recv = new float[count];

    // Deterministic data: rank-specific values with noise
    srand(42 + rank);
    for (int i = 0; i < count; i++)
        send[i] = (float)(rank + 1) + (float)rand() / (float)RAND_MAX;

    if (rank == 0)
        std::cout << "AdaComp Test | Collective: " << test
                  << " | Size: " << count << " floats"
                  << " | Procs: " << nprocs << std::endl;

    // ---- MPI_Allreduce ----
    if (strcmp(test, "allreduce") == 0) {
        MPI_Allreduce(send, recv, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            // Sum of (r+1 + noise) for r=0..nprocs-1 should be in
            // [sum_of_ranks, sum_of_ranks + nprocs]
            float sum_ranks = 0;
            for (int r = 0; r < nprocs; r++) sum_ranks += (float)(r + 1);
            bool ok = (recv[0] >= sum_ranks && recv[0] <= sum_ranks + nprocs);
            std::cout << "Result[0]=" << recv[0]
                      << "  Expected=[" << sum_ranks << ", " << sum_ranks + nprocs << "]"
                      << "  Valid=" << (ok ? "YES" : "NO") << std::endl;
        }
    }
    // ---- MPI_Bcast ----
    else if (strcmp(test, "bcast") == 0) {
        if (rank == 0)
            for (int i = 0; i < count; i++)
                send[i] = (float)i * 0.001f;

        memcpy(recv, send, (size_t)count * sizeof(float));
        MPI_Bcast(recv, count, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Every rank checks first 100 elements against root's values
        int local_ok = 1;
        for (int i = 0; i < std::min(count, 100); i++) {
            float expected = (float)i * 0.001f;
            if (std::abs(recv[i] - expected) > 0.01f) { local_ok = 0; break; }
        }

        int global_ok;
        MPI_Reduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "Bcast verification: " << (global_ok ? "PASS" : "FAIL") << std::endl;
    }
    // ---- MPI_Reduce ----
    else if (strcmp(test, "reduce") == 0) {
        MPI_Reduce(send, recv, count, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            float sum_ranks = 0;
            for (int r = 0; r < nprocs; r++) sum_ranks += (float)(r + 1);
            bool ok = (recv[0] >= sum_ranks && recv[0] <= sum_ranks + nprocs);
            std::cout << "Result[0]=" << recv[0]
                      << "  Expected=[" << sum_ranks << ", " << sum_ranks + nprocs << "]"
                      << "  Valid=" << (ok ? "YES" : "NO") << std::endl;
        }
    }
    else {
        if (rank == 0)
            std::cerr << "Unknown test: " << test
                      << " (use: allreduce, bcast, reduce)" << std::endl;
    }

    delete[] send;
    delete[] recv;
    MPI_Finalize();
    return 0;
}
