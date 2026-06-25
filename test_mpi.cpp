#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read count from command line; default to 1 million for the 'sweet spot' search
    int count = (argc > 1) ? atoi(argv[1]) : 1048576;

    float* send_data = new float[count];
    float* recv_data = new float[count];

    // Seed the random generator
    srand(time(NULL) + rank);

    for (int i = 0; i < count; i++) {
        // Increased Noise: Generates values with 6-7 decimal places of randomness
        // This is complex enough that Lossy will have to approximate it.
        float noise = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        send_data[i] = static_cast<float>(rank + 1) + noise; 
    }

    if (rank == 0) {
        std::cout << "Executing Hybrid MPI Test | Size: " << count << " floats" << std::endl;
    }

    // The call intercepted by libhybrid.so
    MPI_Allreduce(send_data, recv_data, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Reduction Complete. Check Terminal for Compression Stats." << std::endl;
    }

    delete[] send_data;
    delete[] recv_data;
    MPI_Finalize();

    return 0;
}