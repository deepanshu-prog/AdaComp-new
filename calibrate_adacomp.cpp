// AdaComp Calibration Tool
// Empirically measures raw MPI, lossless (zstd), and lossy (SZ3) performance
// across message sizes to find optimal switching thresholds.
// Run WITHOUT LD_PRELOAD: mpirun -np 3 ./calibrate_adacomp

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

extern "C" {
    #include "sz3c.h"
}
#include "zstd.h"

static const int WARMUP    = 5;
static const int MEASURED  = 20;
static const double ABS_ERR = 1e-4;
static const int ZSTD_LVL  = 1;
static const int PIPE_K    = 4;

static const int SIZES[] = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152
};
static const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);

// ============================================================
// Benchmark: Raw MPI_Allreduce
// ============================================================
static double bench_raw(float* send, float* recv, int count, MPI_Comm comm) {
    for (int i = 0; i < WARMUP; i++)
        MPI_Allreduce(send, recv, count, MPI_FLOAT, MPI_SUM, comm);
    MPI_Barrier(comm);

    double total = 0;
    for (int i = 0; i < MEASURED; i++) {
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();
        MPI_Allreduce(send, recv, count, MPI_FLOAT, MPI_SUM, comm);
        double t1 = MPI_Wtime();
        total += (t1 - t0);
    }
    return (total / MEASURED) * 1e6;
}

// ============================================================
// Benchmark: Lossless (zstd) compressed allreduce
// ============================================================
static double bench_lossless(float* send, float* recv, int count,
                             int nprocs, MPI_Comm comm, double* ratio_out) {
    size_t src_size = (size_t)count * sizeof(float);
    size_t bound = ZSTD_compressBound(src_size);

    for (int i = 0; i < WARMUP; i++) {
        unsigned char* comp = (unsigned char*)malloc(bound);
        size_t cs = ZSTD_compress(comp, bound, send, src_size, ZSTD_LVL);
        int my_s = (int)cs;
        std::vector<int> sizes(nprocs), disp(nprocs);
        MPI_Allgather(&my_s, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm);
        int tot = 0;
        for (int j = 0; j < nprocs; j++) { disp[j] = tot; tot += sizes[j]; }
        std::vector<unsigned char> gb(tot);
        MPI_Allgatherv(comp, my_s, MPI_BYTE, gb.data(), sizes.data(), disp.data(), MPI_BYTE, comm);
        memset(recv, 0, src_size);
        for (int j = 0; j < nprocs; j++) {
            float* dec = (float*)malloc(src_size);
            ZSTD_decompress(dec, src_size, &gb[disp[j]], sizes[j]);
            for (int k = 0; k < count; k++) recv[k] += dec[k];
            free(dec);
        }
        free(comp);
    }

    MPI_Barrier(comm);
    double total = 0;
    size_t last_cs = 0;

    for (int i = 0; i < MEASURED; i++) {
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        unsigned char* comp = (unsigned char*)malloc(bound);
        size_t cs = ZSTD_compress(comp, bound, send, src_size, ZSTD_LVL);
        last_cs = cs;
        int my_s = (int)cs;
        std::vector<int> sizes(nprocs), disp(nprocs);
        MPI_Allgather(&my_s, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm);
        int tot = 0;
        for (int j = 0; j < nprocs; j++) { disp[j] = tot; tot += sizes[j]; }
        std::vector<unsigned char> gb(tot);
        MPI_Allgatherv(comp, my_s, MPI_BYTE, gb.data(), sizes.data(), disp.data(), MPI_BYTE, comm);
        memset(recv, 0, src_size);
        for (int j = 0; j < nprocs; j++) {
            float* dec = (float*)malloc(src_size);
            ZSTD_decompress(dec, src_size, &gb[disp[j]], sizes[j]);
            for (int k = 0; k < count; k++) recv[k] += dec[k];
            free(dec);
        }

        double t1 = MPI_Wtime();
        total += (t1 - t0);
        free(comp);
    }

    *ratio_out = (double)src_size / (double)last_cs;
    return (total / MEASURED) * 1e6;
}

// ============================================================
// Benchmark: Lossy (SZ3) compressed allreduce
// ============================================================
static double bench_lossy(float* send, float* recv, int count,
                          int nprocs, MPI_Comm comm, double* ratio_out) {
    for (int i = 0; i < WARMUP; i++) {
        size_t cs = 0;
        unsigned char* comp = SZ_compress_args(0, (void*)send, &cs,
                              1, ABS_ERR, 0, 0, 0, 0, 0, 0, (size_t)count);
        int my_s = (int)cs;
        std::vector<int> sizes(nprocs), disp(nprocs);
        MPI_Allgather(&my_s, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm);
        int tot = 0;
        for (int j = 0; j < nprocs; j++) { disp[j] = tot; tot += sizes[j]; }
        std::vector<unsigned char> gb(tot);
        MPI_Allgatherv(comp, my_s, MPI_BYTE, gb.data(), sizes.data(), disp.data(), MPI_BYTE, comm);
        memset(recv, 0, (size_t)count * sizeof(float));
        for (int j = 0; j < nprocs; j++) {
            float* dec = (float*)SZ_decompress(0, &gb[disp[j]], sizes[j],
                          0, 0, 0, 0, (size_t)count);
            for (int k = 0; k < count; k++) recv[k] += dec[k];
            free(dec);
        }
        free(comp);
    }

    MPI_Barrier(comm);
    double total = 0;
    size_t last_cs = 0;

    for (int i = 0; i < MEASURED; i++) {
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        size_t cs = 0;
        unsigned char* comp = SZ_compress_args(0, (void*)send, &cs,
                              1, ABS_ERR, 0, 0, 0, 0, 0, 0, (size_t)count);
        last_cs = cs;
        int my_s = (int)cs;
        std::vector<int> sizes(nprocs), disp(nprocs);
        MPI_Allgather(&my_s, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm);
        int tot = 0;
        for (int j = 0; j < nprocs; j++) { disp[j] = tot; tot += sizes[j]; }
        std::vector<unsigned char> gb(tot);
        MPI_Allgatherv(comp, my_s, MPI_BYTE, gb.data(), sizes.data(), disp.data(), MPI_BYTE, comm);
        memset(recv, 0, (size_t)count * sizeof(float));
        for (int j = 0; j < nprocs; j++) {
            float* dec = (float*)SZ_decompress(0, &gb[disp[j]], sizes[j],
                          0, 0, 0, 0, (size_t)count);
            for (int k = 0; k < count; k++) recv[k] += dec[k];
            free(dec);
        }

        double t1 = MPI_Wtime();
        total += (t1 - t0);
        free(comp);
    }

    *ratio_out = (double)((size_t)count * sizeof(float)) / (double)last_cs;
    return (total / MEASURED) * 1e6;
}

// ============================================================
// Benchmark: Pipelined lossy compressed allreduce
// Splits data into PIPE_K chunks, overlaps compression with
// non-blocking MPI_Iallgatherv.
// ============================================================
static double bench_pipelined(float* send, float* recv, int count,
                              int nprocs, MPI_Comm comm) {
    if (count < PIPE_K * 16) return 1e18; // too small for pipelining

    int base = count / PIPE_K;
    int rem  = count % PIPE_K;

    auto run_once = [&]() {
        memset(recv, 0, (size_t)count * sizeof(float));
        struct Slot {
            unsigned char* sbuf; int ssize;
            std::vector<int> rsizes, rdisp;
            std::vector<unsigned char> rdata;
            MPI_Request req;
            int off, cnt;
        };
        Slot slots[2]; slots[0].sbuf = NULL; slots[1].sbuf = NULL;
        int cur = 0; bool prev_active = false;

        for (int k = 0; k < PIPE_K; k++) {
            int off = k * base + std::min(k, rem);
            int cnt = base + (k < rem ? 1 : 0);
            Slot& s = slots[cur];
            s.off = off; s.cnt = cnt;
            s.rsizes.resize(nprocs);

            size_t cs = 0;
            s.sbuf = SZ_compress_args(0, (void*)&send[off], &cs,
                      1, ABS_ERR, 0, 0, 0, 0, 0, 0, (size_t)cnt);
            s.ssize = (int)cs;
            MPI_Allgather(&s.ssize, 1, MPI_INT, s.rsizes.data(), 1, MPI_INT, comm);
            s.rdisp.resize(nprocs);
            int tot = 0;
            for (int i = 0; i < nprocs; i++) { s.rdisp[i] = tot; tot += s.rsizes[i]; }
            s.rdata.resize(tot);
            MPI_Iallgatherv(s.sbuf, s.ssize, MPI_BYTE,
                            s.rdata.data(), s.rsizes.data(), s.rdisp.data(),
                            MPI_BYTE, comm, &s.req);

            if (prev_active) {
                Slot& p = slots[1 - cur];
                MPI_Wait(&p.req, MPI_STATUS_IGNORE);
                for (int i = 0; i < nprocs; i++) {
                    float* dec = (float*)SZ_decompress(0, &p.rdata[p.rdisp[i]],
                                  p.rsizes[i], 0, 0, 0, 0, (size_t)p.cnt);
                    for (int j = 0; j < p.cnt; j++) recv[p.off + j] += dec[j];
                    free(dec);
                }
                free(p.sbuf); p.sbuf = NULL;
            }
            prev_active = true;
            cur = 1 - cur;
        }
        if (prev_active) {
            Slot& p = slots[1 - cur];
            MPI_Wait(&p.req, MPI_STATUS_IGNORE);
            for (int i = 0; i < nprocs; i++) {
                float* dec = (float*)SZ_decompress(0, &p.rdata[p.rdisp[i]],
                              p.rsizes[i], 0, 0, 0, 0, (size_t)p.cnt);
                for (int j = 0; j < p.cnt; j++) recv[p.off + j] += dec[j];
                free(dec);
            }
            free(p.sbuf);
        }
    };

    for (int i = 0; i < WARMUP; i++) run_once();
    MPI_Barrier(comm);

    double total = 0;
    for (int i = 0; i < MEASURED; i++) {
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();
        run_once();
        double t1 = MPI_Wtime();
        total += (t1 - t0);
    }
    return (total / MEASURED) * 1e6;
}

// ============================================================
// Main: sweep sizes, find thresholds, write config
// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) {
        std::cout << "============================================" << std::endl;
        std::cout << "  AdaComp Calibration Tool" << std::endl;
        std::cout << "  Processes: " << nprocs << std::endl;
        std::cout << "  Warmup iterations: " << WARMUP << std::endl;
        std::cout << "  Measured iterations: " << MEASURED << std::endl;
        std::cout << "  Error bound: " << ABS_ERR << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << std::endl;
    }

    std::ofstream csv;
    if (rank == 0)
        csv.open("calibration_results.csv");
    if (rank == 0)
        csv << "Size,Raw_us,Lossless_us,Lossy_us,Pipelined_us,Lossless_Ratio,Lossy_Ratio" << std::endl;

    // Storage for threshold detection
    std::vector<int>    sizes_vec;
    std::vector<double> raw_times, lossless_times, lossy_times, pipelined_times;

    for (int si = 0; si < NUM_SIZES; si++) {
        int count = SIZES[si];

        float* send = new float[count];
        float* recv = new float[count];
        srand(42 + rank);
        for (int i = 0; i < count; i++)
            send[i] = (float)(rank + 1) + (float)rand() / (float)RAND_MAX;

        if (rank == 0)
            std::cout << "Calibrating size " << count << " floats ("
                      << (count * 4 / 1024) << " KB)..." << std::flush;

        double t_raw = bench_raw(send, recv, count, MPI_COMM_WORLD);

        double r_lossless = 1.0, r_lossy = 1.0;
        double t_lossless = bench_lossless(send, recv, count, nprocs, MPI_COMM_WORLD, &r_lossless);
        double t_lossy    = bench_lossy(send, recv, count, nprocs, MPI_COMM_WORLD, &r_lossy);
        double t_pipe     = bench_pipelined(send, recv, count, nprocs, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << " Raw=" << std::fixed << std::setprecision(0) << t_raw
                      << "us  Lossless=" << t_lossless
                      << "us  Lossy=" << t_lossy
                      << "us  Pipelined=" << t_pipe << "us" << std::endl;

            csv << count << ","
                << std::fixed << std::setprecision(1) << t_raw << ","
                << t_lossless << "," << t_lossy << "," << t_pipe << ","
                << std::setprecision(2) << r_lossless << ","
                << r_lossy << std::endl;

            sizes_vec.push_back(count);
            raw_times.push_back(t_raw);
            lossless_times.push_back(t_lossless);
            lossy_times.push_back(t_lossy);
            pipelined_times.push_back(t_pipe);
        }

        delete[] send;
        delete[] recv;
    }

    // ---- Threshold detection (rank 0 only) ----
    if (rank == 0) {
        csv.close();

        // T1: first size where lossless beats raw
        int threshold_lossless = sizes_vec.back();
        for (size_t i = 0; i < sizes_vec.size(); i++) {
            if (lossless_times[i] < raw_times[i]) {
                threshold_lossless = sizes_vec[i];
                break;
            }
        }

        // T2: first size where lossy beats lossless
        int threshold_lossy = sizes_vec.back();
        for (size_t i = 0; i < sizes_vec.size(); i++) {
            if (lossy_times[i] < lossless_times[i]) {
                threshold_lossy = sizes_vec[i];
                break;
            }
        }

        // Ensure T2 >= T1
        if (threshold_lossy < threshold_lossless)
            threshold_lossy = threshold_lossless;

        // T3: first size where pipelined beats non-pipelined lossy
        int threshold_pipeline = sizes_vec.back();
        for (size_t i = 0; i < sizes_vec.size(); i++) {
            if (pipelined_times[i] < lossy_times[i] && sizes_vec[i] >= threshold_lossy) {
                threshold_pipeline = sizes_vec[i];
                break;
            }
        }

        std::cout << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "  Calibration Results" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "  T1 (Raw -> Lossless):       " << threshold_lossless << " floats ("
                  << (threshold_lossless * 4 / 1024) << " KB)" << std::endl;
        std::cout << "  T2 (Lossless -> Lossy):      " << threshold_lossy << " floats ("
                  << (threshold_lossy * 4 / 1024) << " KB)" << std::endl;
        std::cout << "  T3 (Lossy -> Pipelined):     " << threshold_pipeline << " floats ("
                  << (threshold_pipeline * 4 / 1024) << " KB)" << std::endl;
        std::cout << std::endl;
        std::cout << "  Switching strategy (mode 4):" << std::endl;
        std::cout << "    count < " << threshold_lossless << "  -->  Raw MPI" << std::endl;
        std::cout << "    " << threshold_lossless << " <= count < " << threshold_lossy
                  << "  -->  Lossless (zstd)" << std::endl;
        std::cout << "    " << threshold_lossy << " <= count < " << threshold_pipeline
                  << "  -->  Lossy (SZ3)" << std::endl;
        std::cout << "    count >= " << threshold_pipeline
                  << "  -->  Pipelined Lossy (SZ3 + overlap)" << std::endl;
        std::cout << "============================================" << std::endl;

        // Write config file
        std::ofstream conf("adacomp.conf");
        conf << "# AdaComp configuration (auto-generated by calibrate_adacomp)" << std::endl;
        conf << "# Cluster: " << nprocs << " processes" << std::endl;
        conf << "ADACOMP_MODE=4" << std::endl;
        conf << "ADACOMP_THRESHOLD_LOSSLESS=" << threshold_lossless << std::endl;
        conf << "ADACOMP_THRESHOLD_LOSSY=" << threshold_lossy << std::endl;
        conf << "ADACOMP_THRESHOLD_PIPELINE=" << threshold_pipeline << std::endl;
        conf << "ADACOMP_PIPELINE_CHUNKS=" << PIPE_K << std::endl;
        conf << "ADACOMP_ERROR_BOUND=" << std::scientific << ABS_ERR << std::endl;
        conf << "ADACOMP_ZSTD_LEVEL=" << ZSTD_LVL << std::endl;
        conf << "ADACOMP_VERBOSE=0" << std::endl;
        conf.close();

        std::cout << std::endl;
        std::cout << "Config written to: adacomp.conf" << std::endl;
        std::cout << "Detailed CSV:      calibration_results.csv" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
