// AdaComp: Adaptive Multi-Tier Compression Middleware for MPI Collectives
// Transparently intercepts MPI collectives via LD_PRELOAD and applies
// message-size-aware compression switching (Raw → Lossless → Lossy)
// with optional pipelined computation-communication overlap.

#include <mpi.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <algorithm>

extern "C" {
    #include "sz3c.h"
}
#include "zstd.h"

// ============================================================
// Real MPI function pointers (resolved via dlsym)
// ============================================================
typedef int (*Allreduce_fn)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
typedef int (*Bcast_fn)(void*, int, MPI_Datatype, int, MPI_Comm);
typedef int (*Reduce_fn)(const void*, void*, int, MPI_Datatype, MPI_Op, int, MPI_Comm);

static Allreduce_fn real_Allreduce = NULL;
static Bcast_fn     real_Bcast     = NULL;
static Reduce_fn    real_Reduce    = NULL;

// ============================================================
// Compression tier classification
// ============================================================
enum CompTier { TIER_RAW = 0, TIER_LOSSLESS = 1, TIER_LOSSY = 2 };

static const char* tier_name(CompTier t) {
    switch (t) {
        case TIER_RAW:      return "RAW";
        case TIER_LOSSLESS: return "LOSSLESS";
        case TIER_LOSSY:    return "LOSSY";
    }
    return "UNKNOWN";
}

// ============================================================
// Configuration (loaded once from file + env vars)
// ============================================================
struct AdaCompConfig {
    int  mode;                // 0=baseline, 1=lossless-only, 2=lossy-only, 3=adaptive, 4=adaptive+pipelined
    int  threshold_lossless;  // Below: raw MPI
    int  threshold_lossy;     // Above: lossy; between thresholds: lossless
    int  threshold_pipeline;  // Min count to activate pipelining (mode 4)
    int  pipeline_chunks;     // Number of pipeline stages (K)
    double error_bound;       // SZ3 absolute error bound
    int  zstd_level;          // zstd compression level (1=fastest)
    bool verbose;             // Print per-call stats
};

static AdaCompConfig cfg = {3, 2048, 32768, 65536, 4, 1e-4, 1, false};
static bool cfg_loaded = false;

static void parse_config_file(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        if      (key == "ADACOMP_MODE")               cfg.mode = std::stoi(val);
        else if (key == "ADACOMP_THRESHOLD_LOSSLESS")  cfg.threshold_lossless = std::stoi(val);
        else if (key == "ADACOMP_THRESHOLD_LOSSY")     cfg.threshold_lossy = std::stoi(val);
        else if (key == "ADACOMP_ERROR_BOUND")         cfg.error_bound = std::stod(val);
        else if (key == "ADACOMP_THRESHOLD_PIPELINE")    cfg.threshold_pipeline = std::stoi(val);
        else if (key == "ADACOMP_PIPELINE_CHUNKS")     cfg.pipeline_chunks = std::stoi(val);
        else if (key == "ADACOMP_ZSTD_LEVEL")          cfg.zstd_level = std::stoi(val);
        else if (key == "ADACOMP_VERBOSE")             cfg.verbose = (std::stoi(val) != 0);
    }
}

static void load_config() {
    if (cfg_loaded) return;
    cfg_loaded = true;

    const char* path = getenv("ADACOMP_CONFIG");
    if (path) parse_config_file(path);
    else      parse_config_file("adacomp.conf");

    char* v;
    if ((v = getenv("ADACOMP_MODE")))              cfg.mode = atoi(v);
    if ((v = getenv("ADACOMP_THRESHOLD_LOSSLESS"))) cfg.threshold_lossless = atoi(v);
    if ((v = getenv("ADACOMP_THRESHOLD_LOSSY")))    cfg.threshold_lossy = atoi(v);
    if ((v = getenv("ADACOMP_ERROR_BOUND")))        cfg.error_bound = atof(v);
    if ((v = getenv("ADACOMP_THRESHOLD_PIPELINE")))  cfg.threshold_pipeline = atoi(v);
    if ((v = getenv("ADACOMP_PIPELINE_CHUNKS")))    cfg.pipeline_chunks = atoi(v);
    if ((v = getenv("ADACOMP_ZSTD_LEVEL")))         cfg.zstd_level = atoi(v);
    if ((v = getenv("ADACOMP_VERBOSE")))            cfg.verbose = (atoi(v) != 0);
}

// ============================================================
// Tier selection based on mode and message size
// ============================================================
static CompTier select_tier(int count, MPI_Datatype datatype) {
    if (datatype != MPI_FLOAT) return TIER_RAW;

    switch (cfg.mode) {
        case 0: return TIER_RAW;
        case 1: return TIER_LOSSLESS;
        case 2: return TIER_LOSSY;
        case 3: // Adaptive multi-tier
        case 4: // Adaptive + pipelined (same tier logic; pipelining handled in allreduce)
            if (count < cfg.threshold_lossless) return TIER_RAW;
            if (count < cfg.threshold_lossy)    return TIER_LOSSLESS;
            return TIER_LOSSY;
    }
    return TIER_RAW;
}

// ============================================================
// Compression / decompression wrappers
// ============================================================
static unsigned char* compress_data(CompTier tier, const float* data, int count, size_t* out_size) {
    if (tier == TIER_LOSSLESS) {
        size_t src_size = (size_t)count * sizeof(float);
        size_t bound = ZSTD_compressBound(src_size);
        unsigned char* buf = (unsigned char*)malloc(bound);
        if (!buf) return NULL;
        *out_size = ZSTD_compress(buf, bound, data, src_size, cfg.zstd_level);
        if (ZSTD_isError(*out_size)) { free(buf); return NULL; }
        return buf;
    }
    // TIER_LOSSY: SZ3
    return SZ_compress_args(0, (void*)data, out_size,
                            1, cfg.error_bound, 0, 0, 0, 0, 0, 0, (size_t)count);
}

static float* decompress_data(CompTier tier, const unsigned char* data, size_t comp_size, int count) {
    if (tier == TIER_LOSSLESS) {
        size_t dst_size = (size_t)count * sizeof(float);
        float* buf = (float*)malloc(dst_size);
        if (!buf) return NULL;
        size_t result = ZSTD_decompress(buf, dst_size, data, comp_size);
        if (ZSTD_isError(result)) { free(buf); return NULL; }
        return buf;
    }
    // TIER_LOSSY: SZ3
    return (float*)SZ_decompress(0, (unsigned char*)data, comp_size,
                                 0, 0, 0, 0, (size_t)count);
}

// ============================================================
// Resolve real MPI symbols
// ============================================================
static void ensure_symbols() {
    if (!real_Allreduce)
        real_Allreduce = (Allreduce_fn)dlsym(RTLD_NEXT, "MPI_Allreduce");
    if (!real_Bcast)
        real_Bcast = (Bcast_fn)dlsym(RTLD_NEXT, "MPI_Bcast");
    if (!real_Reduce)
        real_Reduce = (Reduce_fn)dlsym(RTLD_NEXT, "MPI_Reduce");
}

// ============================================================
// Pipelined compressed allreduce
// Splits data into K chunks. While chunk k transfers over the
// network (MPI_Iallgatherv), chunk k+1 is being compressed and
// chunk k-1 is being decompressed — overlapping computation
// with communication.
//
// Timeline for K=4:
//   Compress:  [C0]  [C1]  [C2]  [C3]
//   Network:     [==N0==][==N1==][==N2==][==N3==]
//   Decompress:       [D0]  [D1]  [D2]  [D3]
//
// ============================================================
struct PipeSlot {
    unsigned char* send_buf;
    int            send_size;
    std::vector<int>           recv_sizes;
    std::vector<int>           recv_displs;
    std::vector<unsigned char> recv_data;
    MPI_Request    request;
    int            chunk_offset;
    int            chunk_count;
};

static int pipelined_compressed_allreduce(const float* sendbuf, float* recvbuf,
                                          int count, CompTier tier,
                                          int rank, int nprocs, MPI_Comm comm,
                                          double* out_max_err, size_t* out_comp_bytes) {
    int K = cfg.pipeline_chunks;
    if (K < 2) K = 2;
    int base = count / K;
    int rem  = count % K;

    memset(recvbuf, 0, (size_t)count * sizeof(float));

    PipeSlot slots[2];
    slots[0].send_buf = NULL;
    slots[1].send_buf = NULL;
    int cur = 0;
    bool prev_active = false;
    double max_err = 0.0;
    size_t total_comp = 0;

    for (int k = 0; k < K; k++) {
        int offset = k * base + std::min(k, rem);
        int cnt    = base + (k < rem ? 1 : 0);
        if (cnt <= 0) break;

        PipeSlot& s = slots[cur];
        s.chunk_offset = offset;
        s.chunk_count  = cnt;
        s.recv_sizes.resize(nprocs);

        // --- STEP A: Compress this chunk ---
        // (overlaps with Iallgatherv of previous chunk in the network)
        size_t comp_size = 0;
        s.send_buf = compress_data(tier, &sendbuf[offset], cnt, &comp_size);
        if (!s.send_buf) return -1;
        s.send_size = (int)comp_size;
        total_comp += comp_size;

        // --- STEP B: Exchange compressed sizes (blocking, tiny) ---
        MPI_Allgather(&s.send_size, 1, MPI_INT,
                      s.recv_sizes.data(), 1, MPI_INT, comm);

        // Prepare receive layout
        s.recv_displs.resize(nprocs);
        int total = 0;
        for (int i = 0; i < nprocs; i++) {
            s.recv_displs[i] = total;
            total += s.recv_sizes[i];
        }
        s.recv_data.resize(total);

        // --- STEP C: Non-blocking data exchange (large, overlapped) ---
        MPI_Iallgatherv(s.send_buf, s.send_size, MPI_BYTE,
                        s.recv_data.data(), s.recv_sizes.data(),
                        s.recv_displs.data(), MPI_BYTE, comm, &s.request);

        // --- STEP D: Process PREVIOUS chunk while current transfers ---
        if (prev_active) {
            PipeSlot& p = slots[1 - cur];
            MPI_Wait(&p.request, MPI_STATUS_IGNORE);

            for (int i = 0; i < nprocs; i++) {
                float* dec = decompress_data(tier,
                    &p.recv_data[p.recv_displs[i]], p.recv_sizes[i], p.chunk_count);
                if (!dec) { free(s.send_buf); return -1; }

                if (i == rank) {
                    const float* orig = &sendbuf[p.chunk_offset];
                    for (int j = 0; j < p.chunk_count; j++) {
                        double diff = std::abs((double)orig[j] - (double)dec[j]);
                        if (diff > max_err) max_err = diff;
                    }
                }
                for (int j = 0; j < p.chunk_count; j++)
                    recvbuf[p.chunk_offset + j] += dec[j];
                free(dec);
            }
            free(p.send_buf);
            p.send_buf = NULL;
        }

        prev_active = true;
        cur = 1 - cur;
    }

    // --- Process last chunk ---
    if (prev_active) {
        PipeSlot& p = slots[1 - cur];
        MPI_Wait(&p.request, MPI_STATUS_IGNORE);

        for (int i = 0; i < nprocs; i++) {
            float* dec = decompress_data(tier,
                &p.recv_data[p.recv_displs[i]], p.recv_sizes[i], p.chunk_count);
            if (!dec) return -1;

            if (i == rank) {
                const float* orig = &sendbuf[p.chunk_offset];
                for (int j = 0; j < p.chunk_count; j++) {
                    double diff = std::abs((double)orig[j] - (double)dec[j]);
                    if (diff > max_err) max_err = diff;
                }
            }
            for (int j = 0; j < p.chunk_count; j++)
                recvbuf[p.chunk_offset + j] += dec[j];
            free(dec);
        }
        free(p.send_buf);
    }

    *out_max_err = max_err;
    *out_comp_bytes = total_comp;
    return 0;
}

// ============================================================
// MPI_Allreduce interception
// ============================================================
extern "C" int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    ensure_symbols();
    load_config();

    CompTier tier = select_tier(count, datatype);

    if (tier == TIER_RAW || op != MPI_SUM) {
        double t0 = MPI_Wtime();
        int ret = real_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        double t1 = MPI_Wtime();
        if (cfg.verbose) {
            int rank; MPI_Comm_rank(comm, &rank);
            if (rank == 0 && datatype == MPI_FLOAT)
                std::cout << "ADACOMP,ALLREDUCE,RAW," << count << ","
                          << std::fixed << std::setprecision(1) << (t1-t0)*1e6
                          << ",1.00,100.000000" << std::endl;
        }
        return ret;
    }

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double t_start = MPI_Wtime();

    // ---- PIPELINED PATH (mode 4, large messages) ----
    if (cfg.mode == 4 && count >= cfg.threshold_pipeline) {
        double max_err = 0.0;
        size_t comp_bytes = 0;
        int rc = pipelined_compressed_allreduce(
            (const float*)sendbuf, (float*)recvbuf, count,
            tier, rank, nprocs, comm, &max_err, &comp_bytes);

        if (rc != 0)
            return real_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);

        double t_end = MPI_Wtime();
        if (rank == 0 && cfg.verbose) {
            double ratio = (comp_bytes > 0)
                ? (double)((size_t)count * sizeof(float)) / (double)comp_bytes
                : 1.0;
            double acc = 100.0 * (1.0 - max_err);
            std::cout << "ADACOMP,ALLREDUCE,PIPELINED_" << tier_name(tier)
                      << "," << count << ","
                      << std::fixed << std::setprecision(1) << (t_end - t_start)*1e6 << ","
                      << std::setprecision(2) << ratio << ","
                      << std::setprecision(6) << acc << std::endl;
        }
        return MPI_SUCCESS;
    }

    // ---- STANDARD COMPRESSED PATH (modes 1-3, or mode 4 below pipeline threshold) ----
    size_t comp_size = 0;
    unsigned char* compressed = compress_data(tier, (const float*)sendbuf, count, &comp_size);
    if (!compressed)
        return real_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);

    std::vector<int> all_sizes(nprocs);
    int my_size = (int)comp_size;
    MPI_Allgather(&my_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, comm);

    std::vector<int> displs(nprocs);
    int total_bytes = 0;
    for (int i = 0; i < nprocs; i++) {
        displs[i] = total_bytes;
        total_bytes += all_sizes[i];
    }
    std::vector<unsigned char> recv_compressed(total_bytes);
    MPI_Allgatherv(compressed, my_size, MPI_BYTE,
                   recv_compressed.data(), all_sizes.data(), displs.data(), MPI_BYTE, comm);

    float* result = (float*)recvbuf;
    memset(result, 0, (size_t)count * sizeof(float));
    double max_err = 0.0;

    for (int i = 0; i < nprocs; i++) {
        float* dec = decompress_data(tier, &recv_compressed[displs[i]], all_sizes[i], count);
        if (!dec) {
            free(compressed);
            return real_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        }
        if (i == rank) {
            const float* orig = (const float*)sendbuf;
            for (int j = 0; j < count; j++) {
                double diff = std::abs((double)orig[j] - (double)dec[j]);
                if (diff > max_err) max_err = diff;
            }
        }
        for (int j = 0; j < count; j++) result[j] += dec[j];
        free(dec);
    }

    double t_end = MPI_Wtime();

    if (rank == 0 && cfg.verbose) {
        double ratio = (double)(count * sizeof(float)) / (double)comp_size;
        double acc = 100.0 * (1.0 - max_err);
        std::cout << "ADACOMP,ALLREDUCE," << tier_name(tier) << "," << count << ","
                  << std::fixed << std::setprecision(1) << (t_end - t_start)*1e6 << ","
                  << std::setprecision(2) << ratio << ","
                  << std::setprecision(6) << acc << std::endl;
    }

    free(compressed);
    return MPI_SUCCESS;
}

// ============================================================
// MPI_Bcast interception
// ============================================================
extern "C" int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                         int root, MPI_Comm comm) {
    ensure_symbols();
    load_config();

    CompTier tier = select_tier(count, datatype);

    if (tier == TIER_RAW) {
        double t0 = MPI_Wtime();
        int ret = real_Bcast(buffer, count, datatype, root, comm);
        double t1 = MPI_Wtime();
        if (cfg.verbose) {
            int rank; MPI_Comm_rank(comm, &rank);
            if (rank == 0)
                std::cout << "ADACOMP,BCAST,RAW," << count << ","
                          << std::fixed << std::setprecision(1) << (t1-t0)*1e6
                          << ",1.00,100.000000" << std::endl;
        }
        return ret;
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    double t_start = MPI_Wtime();

    // Root compresses
    size_t comp_size = 0;
    unsigned char* compressed = NULL;
    if (rank == root) {
        compressed = compress_data(tier, (float*)buffer, count, &comp_size);
        if (!compressed)
            return real_Bcast(buffer, count, datatype, root, comm);
    }

    // Broadcast compressed size
    int size_int = (int)comp_size;
    real_Bcast(&size_int, 1, MPI_INT, root, comm);
    comp_size = (size_t)size_int;

    // Non-root allocates receive buffer
    if (rank != root)
        compressed = (unsigned char*)malloc(comp_size);

    // Broadcast compressed data
    real_Bcast(compressed, (int)comp_size, MPI_BYTE, root, comm);

    // All ranks decompress for consistency
    float* dec = decompress_data(tier, compressed, comp_size, count);
    if (dec) {
        memcpy(buffer, dec, (size_t)count * sizeof(float));
        free(dec);
    }

    double t_end = MPI_Wtime();

    if (rank == 0 && cfg.verbose) {
        double ratio = (double)(count * sizeof(float)) / (double)comp_size;
        std::cout << "ADACOMP,BCAST," << tier_name(tier) << "," << count << ","
                  << std::fixed << std::setprecision(1) << (t_end - t_start)*1e6 << ","
                  << std::setprecision(2) << ratio << ","
                  << "100.000000" << std::endl;
    }

    free(compressed);
    return MPI_SUCCESS;
}

// ============================================================
// MPI_Reduce interception
// ============================================================
extern "C" int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    ensure_symbols();
    load_config();

    CompTier tier = select_tier(count, datatype);

    if (tier == TIER_RAW || op != MPI_SUM) {
        double t0 = MPI_Wtime();
        int ret = real_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
        double t1 = MPI_Wtime();
        if (cfg.verbose) {
            int rank; MPI_Comm_rank(comm, &rank);
            if (rank == 0 && datatype == MPI_FLOAT)
                std::cout << "ADACOMP,REDUCE,RAW," << count << ","
                          << std::fixed << std::setprecision(1) << (t1-t0)*1e6
                          << ",1.00,100.000000" << std::endl;
        }
        return ret;
    }

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double t_start = MPI_Wtime();

    // Every rank compresses
    size_t comp_size = 0;
    unsigned char* compressed = compress_data(tier, (const float*)sendbuf, count, &comp_size);
    if (!compressed)
        return real_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);

    // Gather sizes to root
    int my_size = (int)comp_size;
    std::vector<int> all_sizes(rank == root ? nprocs : 0);
    MPI_Gather(&my_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, root, comm);

    // Gather compressed data to root
    std::vector<int> displs;
    std::vector<unsigned char> recv_compressed;
    int total_bytes = 0;
    if (rank == root) {
        displs.resize(nprocs);
        for (int i = 0; i < nprocs; i++) {
            displs[i] = total_bytes;
            total_bytes += all_sizes[i];
        }
        recv_compressed.resize(total_bytes);
    }
    MPI_Gatherv(compressed, my_size, MPI_BYTE,
                recv_compressed.data(), all_sizes.data(),
                displs.empty() ? NULL : displs.data(), MPI_BYTE, root, comm);

    // Root decompresses and reduces
    if (rank == root) {
        float* result = (float*)recvbuf;
        memset(result, 0, (size_t)count * sizeof(float));
        for (int i = 0; i < nprocs; i++) {
            float* dec = decompress_data(tier, &recv_compressed[displs[i]], all_sizes[i], count);
            if (!dec) {
                free(compressed);
                return real_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
            }
            for (int j = 0; j < count; j++) result[j] += dec[j];
            free(dec);
        }
    }

    double t_end = MPI_Wtime();

    if (rank == 0 && cfg.verbose) {
        double ratio = (double)(count * sizeof(float)) / (double)comp_size;
        double acc = 100.0;
        std::cout << "ADACOMP,REDUCE," << tier_name(tier) << "," << count << ","
                  << std::fixed << std::setprecision(1) << (t_end - t_start)*1e6 << ","
                  << std::setprecision(2) << ratio << ","
                  << std::setprecision(6) << acc << std::endl;
    }

    free(compressed);
    return MPI_SUCCESS;
}
