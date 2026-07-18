#include "spmv_kernel_runner.cuh"

#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cusparse_v2.h>
#include <mpi.h>

#include <new>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

namespace cg = cooperative_groups;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void check_cuda(cudaError_t status,
                       const char *call,
                       const char *file,
                       int line,
                       int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d CUDA error at %s:%d in %s: %s\n",
                rank, file, line, call, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void check_cusparse(cusparseStatus_t status,
                           const char *call,
                           const char *file,
                           int line,
                           int rank) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "Rank %d cuSPARSE error at %s:%d in %s: status %d\n",
                rank, file, line, call, (int)status);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CHECK_CUDA(call, rank) \
    check_cuda((call), #call, __FILE__, __LINE__, (rank))
#define CHECK_CUSPARSE(call, rank) \
    check_cusparse((call), #call, __FILE__, __LINE__, (rank))

struct BaseRunner : SpmvKernelRunner {
    explicit BaseRunner(int rank_) : rank(rank_) {}
    int rank;

    double prep(const CSR_Matrix &, const CSR_Matrix &, float *, float *) override {
        return 0.0;
    }

    void teardown() override {}
};

// =============================================================================
// CSR scalar: one CUDA thread per row.
// =============================================================================

__global__ void csr_scalar_kernel(CSR_Matrix mat,
                                  const float *__restrict__ x,
                                  float *__restrict__ y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < mat.rows) {
        float sum = 0.0f;
        for (int j = mat.row_ptr[row]; j < mat.row_ptr[row + 1]; j++) {
            sum += mat.values[j] * x[mat.col_idx[j]];
        }
        y[row] = sum;
    }
}

struct CsrScalarRunner : BaseRunner {
    using BaseRunner::BaseRunner;

    const char *name() const override {
        return "scalar";
    }

    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) override {
        int threads = 256;
        int blocks = (d_csr.rows + threads - 1) / threads;
        if (blocks > 0) {
            csr_scalar_kernel<<<blocks, threads>>>(d_csr, d_x, d_y);
        }
    }
};

// =============================================================================
// CSR vector: one warp per row.
// =============================================================================

__global__ void csr_vector_kernel(const CSR_Matrix mat,
                                  const float *__restrict__ x,
                                  float *__restrict__ y) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int row = global_thread_id / 32;

    if (row < mat.rows) {
        int row_start = mat.row_ptr[row];
        int row_end = mat.row_ptr[row + 1];

        float sum = 0.0f;
        for (int j = row_start + lane; j < row_end; j += 32) {
            sum += mat.values[j] * x[mat.col_idx[j]];
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            y[row] = sum;
        }
    }
}

struct CsrVectorRunner : BaseRunner {
    using BaseRunner::BaseRunner;

    const char *name() const override {
        return "vector";
    }

    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) override {
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / 32;
        int blocks = (d_csr.rows + warps_per_block - 1) / warps_per_block;
        if (blocks > 0) {
            csr_vector_kernel<<<blocks, threads_per_block>>>(d_csr, d_x, d_y);
        }
    }
};

// =============================================================================
// CSR adaptive.
// =============================================================================

#define ADAPTIVE_NNZ_PER_BLOCK 1024
#define ADAPTIVE_THREADS_PER_BLOCK 256
#define ADAPTIVE_WARP_SIZE 32
#define ADAPTIVE_WARPS_PER_BLOCK \
    (ADAPTIVE_THREADS_PER_BLOCK / ADAPTIVE_WARP_SIZE)
#define ADAPTIVE_STREAM_PARALLEL_MAX 8
#define ADAPTIVE_VECTOR_MAX_ROWS 2

__device__ __forceinline__ float adaptive_warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ void adaptive_stream_to_lds(
    const float *__restrict__ values,
    const int *__restrict__ col_idx,
    const float *__restrict__ x,
    float *lds,
    int base,
    int nnz,
    int tid) {
    const float *vp = values + base;
    const int *cp = col_idx + base;
    const bool aligned =
        (reinterpret_cast<uintptr_t>(vp) & 0xF) == 0 &&
        (reinterpret_cast<uintptr_t>(cp) & 0xF) == 0;

    int i = 0;
    if (aligned) {
        const float4 *vp4 = reinterpret_cast<const float4 *>(vp);
        const int4 *cp4 = reinterpret_cast<const int4 *>(cp);
        const int nnz4 = nnz >> 2;
        for (int k = tid; k < nnz4; k += ADAPTIVE_THREADS_PER_BLOCK) {
            float4 v = vp4[k];
            int4 c = cp4[k];
            int o = k << 2;
            lds[o + 0] = v.x * __ldg(x + c.x);
            lds[o + 1] = v.y * __ldg(x + c.y);
            lds[o + 2] = v.z * __ldg(x + c.z);
            lds[o + 3] = v.w * __ldg(x + c.w);
        }
        i = nnz4 << 2;
    }
    for (int k = i + tid; k < nnz; k += ADAPTIVE_THREADS_PER_BLOCK) {
        lds[k] = vp[k] * __ldg(x + cp[k]);
    }
}

__global__ void csr_adaptive_kernel(CSR_Matrix mat,
                                    const int *__restrict__ block_row_start,
                                    const int *__restrict__ block_row_end,
                                    const float *__restrict__ x,
                                    float *__restrict__ y) {
    __shared__ float lds[ADAPTIVE_NNZ_PER_BLOCK];

    const int row_start = block_row_start[blockIdx.x];
    const int row_end = block_row_end[blockIdx.x];
    const int num_rows = row_end - row_start;
    const int tid = threadIdx.x;

    if (num_rows > ADAPTIVE_VECTOR_MAX_ROWS) {
        const int first_nz = mat.row_ptr[row_start];
        const int last_nz = mat.row_ptr[row_end];
        const int nnz = last_nz - first_nz;

        adaptive_stream_to_lds(mat.values, mat.col_idx, x, lds, first_nz, nnz,
                               tid);
        __syncthreads();

        if (num_rows <= ADAPTIVE_STREAM_PARALLEL_MAX) {
            const int warp_id = tid >> 5;
            const int lane = tid & 31;
            if (warp_id < num_rows) {
                const int row = row_start + warp_id;
                const int ls = mat.row_ptr[row] - first_nz;
                const int le = mat.row_ptr[row + 1] - first_nz;
                float sum = 0.0f;
                for (int j = ls + lane; j < le; j += ADAPTIVE_WARP_SIZE) {
                    sum += lds[j];
                }
                sum = adaptive_warp_reduce_sum(sum);
                if (lane == 0) {
                    y[row] = sum;
                }
            }
        } else {
            for (int local_row = tid; local_row < num_rows;
                 local_row += ADAPTIVE_THREADS_PER_BLOCK) {
                const int row = row_start + local_row;
                const int ls = mat.row_ptr[row] - first_nz;
                const int le = mat.row_ptr[row + 1] - first_nz;
                float sum = 0.0f;
#pragma unroll 4
                for (int j = ls; j < le; j++) {
                    sum += lds[j];
                }
                y[row] = sum;
            }
        }
    } else {
        const int warp_id = tid >> 5;
        const int lane = tid & 31;
        if (warp_id < num_rows) {
            const int row = row_start + warp_id;
            const int s = mat.row_ptr[row];
            const int e = mat.row_ptr[row + 1];
            float sum = 0.0f;
            for (int j = s + lane; j < e; j += ADAPTIVE_WARP_SIZE) {
                sum += mat.values[j] * __ldg(x + mat.col_idx[j]);
            }
            sum = adaptive_warp_reduce_sum(sum);
            if (lane == 0) {
                y[row] = sum;
            }
        }
    }
}

__global__ void csr_adaptive_longrow_kernel(
    CSR_Matrix mat,
    const int *__restrict__ long_row_idx,
    const int *__restrict__ long_chunk_off,
    const float *__restrict__ x,
    float *__restrict__ y) {
    __shared__ float lds[ADAPTIVE_NNZ_PER_BLOCK];
    __shared__ float warp_partials[ADAPTIVE_WARPS_PER_BLOCK];

    const int chunk = blockIdx.x;
    const int row = long_row_idx[chunk];
    const int off = long_chunk_off[chunk];
    const int tid = threadIdx.x;

    const int row_first = mat.row_ptr[row];
    const int row_last = mat.row_ptr[row + 1];
    const int chunk_first = row_first + off;
    int chunk_last = chunk_first + ADAPTIVE_NNZ_PER_BLOCK;
    if (chunk_last > row_last) {
        chunk_last = row_last;
    }
    const int nnz = chunk_last - chunk_first;

    adaptive_stream_to_lds(mat.values, mat.col_idx, x, lds, chunk_first, nnz,
                           tid);
    __syncthreads();

    float sum = 0.0f;
#pragma unroll 4
    for (int i = tid; i < nnz; i += ADAPTIVE_THREADS_PER_BLOCK) {
        sum += lds[i];
    }

    sum = adaptive_warp_reduce_sum(sum);
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
        warp_partials[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < ADAPTIVE_WARPS_PER_BLOCK) ? warp_partials[lane] : 0.0f;
        sum = adaptive_warp_reduce_sum(sum);
        if (lane == 0) {
            atomicAdd(&y[row], sum);
        }
    }
}

struct AdaptiveBlockPlan {
    std::vector<int> nstart;
    std::vector<int> nend;
    std::vector<int> lrow;
    std::vector<int> loff;
};

static void build_adaptive_block_plan(const CSR_Matrix *csr,
                                      AdaptiveBlockPlan &plan) {
    const int rows = csr->rows;
    const int *row_ptr = csr->row_ptr;

    int seg_start = 0;
    int seg_nnz = 0;
    int last_break = 0;

    auto flush_segment = [&](int seg_end) {
        if (seg_end > seg_start) {
            plan.nstart.push_back(seg_start);
            plan.nend.push_back(seg_end);
        }
        seg_start = seg_end;
        seg_nnz = 0;
        last_break = seg_end;
    };

    int i = 0;
    while (i < rows) {
        const int row_nnz = row_ptr[i + 1] - row_ptr[i];

        if (row_nnz > ADAPTIVE_NNZ_PER_BLOCK) {
            flush_segment(i);
            const int chunks =
                (row_nnz + ADAPTIVE_NNZ_PER_BLOCK - 1) /
                ADAPTIVE_NNZ_PER_BLOCK;
            for (int c = 0; c < chunks; c++) {
                plan.lrow.push_back(i);
                plan.loff.push_back(c * ADAPTIVE_NNZ_PER_BLOCK);
            }
            i++;
            seg_start = i;
            last_break = i;
            continue;
        }

        if (seg_nnz + row_nnz > ADAPTIVE_NNZ_PER_BLOCK) {
            if (i > last_break) {
                flush_segment(i);
                continue;
            }
            flush_segment(i + 1);
            i++;
            continue;
        }

        seg_nnz += row_nnz;
        i++;
    }
    flush_segment(rows);
}

struct CsrAdaptiveRunner : BaseRunner {
    using BaseRunner::BaseRunner;

    int *d_nstart = NULL;
    int *d_nend = NULL;
    int *d_lrow = NULL;
    int *d_loff = NULL;
    int num_normal = 0;
    int num_long = 0;
    int rows = 0;

    const char *name() const override {
        return "adaptive";
    }

    double prep(const CSR_Matrix &h_csr,
                const CSR_Matrix &,
                float *,
                float *) override {
        double start = now_seconds();

        AdaptiveBlockPlan plan;
        build_adaptive_block_plan(&h_csr, plan);
        num_normal = (int)plan.nstart.size();
        num_long = (int)plan.lrow.size();
        rows = h_csr.rows;

        if (num_normal > 0) {
            CHECK_CUDA(cudaMalloc((void **)&d_nstart,
                                  (size_t)num_normal * sizeof(int)),
                       rank);
            CHECK_CUDA(cudaMalloc((void **)&d_nend,
                                  (size_t)num_normal * sizeof(int)),
                       rank);
            CHECK_CUDA(cudaMemcpy(d_nstart, plan.nstart.data(),
                                  (size_t)num_normal * sizeof(int),
                                  cudaMemcpyHostToDevice),
                       rank);
            CHECK_CUDA(cudaMemcpy(d_nend, plan.nend.data(),
                                  (size_t)num_normal * sizeof(int),
                                  cudaMemcpyHostToDevice),
                       rank);
        }
        if (num_long > 0) {
            CHECK_CUDA(cudaMalloc((void **)&d_lrow,
                                  (size_t)num_long * sizeof(int)),
                       rank);
            CHECK_CUDA(cudaMalloc((void **)&d_loff,
                                  (size_t)num_long * sizeof(int)),
                       rank);
            CHECK_CUDA(cudaMemcpy(d_lrow, plan.lrow.data(),
                                  (size_t)num_long * sizeof(int),
                                  cudaMemcpyHostToDevice),
                       rank);
            CHECK_CUDA(cudaMemcpy(d_loff, plan.loff.data(),
                                  (size_t)num_long * sizeof(int),
                                  cudaMemcpyHostToDevice),
                       rank);
        }
        CHECK_CUDA(cudaDeviceSynchronize(), rank);
        return now_seconds() - start;
    }

    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) override {
        if (num_long > 0) {
            CHECK_CUDA(cudaMemsetAsync(d_y, 0, (size_t)rows * sizeof(float)),
                       rank);
        }
        if (num_normal > 0) {
            csr_adaptive_kernel<<<num_normal, ADAPTIVE_THREADS_PER_BLOCK>>>(
                d_csr, d_nstart, d_nend, d_x, d_y);
        }
        if (num_long > 0) {
            csr_adaptive_longrow_kernel<<<num_long, ADAPTIVE_THREADS_PER_BLOCK>>>(
                d_csr, d_lrow, d_loff, d_x, d_y);
        }
    }

    void teardown() override {
        cudaFree(d_nstart);
        cudaFree(d_nend);
        cudaFree(d_lrow);
        cudaFree(d_loff);
        d_nstart = NULL;
        d_nend = NULL;
        d_lrow = NULL;
        d_loff = NULL;
        num_normal = 0;
        num_long = 0;
        rows = 0;
    }
};

// =============================================================================
// CSR adaptive, paper-style row blocks.
// =============================================================================

#define PAPER_NNZ_PER_WG 1024
#define PAPER_THREADS_PER_WG 256
#define PAPER_SMALL_VALUE 2

__global__ void csr_adaptive_paper_kernel(CSR_Matrix mat,
                                          const int *__restrict__ row_blocks,
                                          const float *__restrict__ x,
                                          float *__restrict__ y) {
    __shared__ float lds[PAPER_NNZ_PER_WG];

    const int start_row = row_blocks[blockIdx.x];
    const int next_start_row = row_blocks[blockIdx.x + 1];
    const int num_rows = next_start_row - start_row;
    const int local_tid = threadIdx.x;

    if (num_rows > PAPER_SMALL_VALUE) {
        const int first_col = mat.row_ptr[start_row];
        const int num_nonzeros = mat.row_ptr[next_start_row] - first_col;

        for (int i = local_tid; i < num_nonzeros; i += PAPER_THREADS_PER_WG) {
            lds[i] = mat.values[first_col + i] * x[mat.col_idx[first_col + i]];
        }
        __syncthreads();

        int rows_done = 0;
        while (rows_done < num_rows) {
            const int local_row = rows_done + local_tid;
            if (local_row < num_rows) {
                const int row = start_row + local_row;
                const int t_start = mat.row_ptr[row] - first_col;
                const int t_end = mat.row_ptr[row + 1] - first_col;
                float temp = 0.0f;
                for (int j = t_start; j < t_end; j++) {
                    temp += lds[j];
                }
                y[row] = temp;
            }
            rows_done += PAPER_THREADS_PER_WG;
        }
    } else {
        const int warp_id = local_tid >> 5;
        const int lane = local_tid & 31;

        if (warp_id < num_rows) {
            const int row = start_row + warp_id;
            const int start = mat.row_ptr[row];
            const int end = mat.row_ptr[row + 1];

            float sum = 0.0f;
            for (int j = start + lane; j < end; j += 32) {
                sum += mat.values[j] * x[mat.col_idx[j]];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0) {
                y[row] = sum;
            }
        }
    }
}

static int build_row_blocks_paper(const CSR_Matrix *csr, int **row_blocks_out) {
    const int total_rows = csr->rows;
    const int *row_delimiters = csr->row_ptr;

    int *row_blocks = (int *)malloc((size_t)(total_rows + 2) * sizeof(int));
    if (!row_blocks) {
        fprintf(stderr, "Error allocating row_blocks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    row_blocks[0] = 0;
    int sum = 0;
    int last_i = 0;
    int ctr = 1;

    for (int i = 1; i <= total_rows; i++) {
        sum += row_delimiters[i] - row_delimiters[i - 1];

        if (sum == PAPER_NNZ_PER_WG) {
            last_i = i;
            row_blocks[ctr++] = i;
            sum = 0;
        } else if (sum > PAPER_NNZ_PER_WG) {
            if (i - last_i > 1) {
                row_blocks[ctr++] = i - 1;
                i--;
            } else {
                row_blocks[ctr++] = i;
            }
            last_i = i;
            sum = 0;
        }
    }

    if (row_blocks[ctr - 1] != total_rows) {
        row_blocks[ctr++] = total_rows;
    }

    *row_blocks_out = row_blocks;
    return ctr - 1;
}

struct CsrAdaptivePaperRunner : BaseRunner {
    using BaseRunner::BaseRunner;

    int *d_row_blocks = NULL;
    int num_blocks = 0;

    const char *name() const override {
        return "adaptive-paper";
    }

    double prep(const CSR_Matrix &h_csr,
                const CSR_Matrix &,
                float *,
                float *) override {
        double start = now_seconds();
        int *h_row_blocks = NULL;
        num_blocks = build_row_blocks_paper(&h_csr, &h_row_blocks);
        if (num_blocks > 0) {
            CHECK_CUDA(cudaMalloc((void **)&d_row_blocks,
                                  (size_t)(num_blocks + 1) * sizeof(int)),
                       rank);
            CHECK_CUDA(cudaMemcpy(d_row_blocks, h_row_blocks,
                                  (size_t)(num_blocks + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice),
                       rank);
        }
        free(h_row_blocks);
        CHECK_CUDA(cudaDeviceSynchronize(), rank);
        return now_seconds() - start;
    }

    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) override {
        if (num_blocks > 0) {
            csr_adaptive_paper_kernel<<<num_blocks, PAPER_THREADS_PER_WG>>>(
                d_csr, d_row_blocks, d_x, d_y);
        }
    }

    void teardown() override {
        cudaFree(d_row_blocks);
        d_row_blocks = NULL;
        num_blocks = 0;
    }
};

// =============================================================================
// CSR partial overlap.
// =============================================================================

#define PARTIAL_MAX_BATCH_SIZE 1024
#define PARTIAL_THREADS_PER_BLOCK 128
#define PARTIAL_SWITCH_POINT_1 17
#define PARTIAL_SWITCH_POINT_2 34
#define PARTIAL_SWITCH_POINT_4 64
#define PARTIAL_SWITCH_POINT_8 122
#define PARTIAL_SWITCH_POINT_16 216

typedef struct {
    int start_row;
    int end_row;
    int start_nnz;
    int end_nnz;
} BatchInfo;

static int build_partial_batches(const CSR_Matrix *csr,
                                 BatchInfo **batches_out,
                                 int **long_rows_out,
                                 int *long_row_count_out) {
    int rows = csr->rows;

    int long_capacity = rows > 0 ? rows : 1;
    int batch_capacity = rows + 1;
    int *long_rows = (int *)malloc((size_t)long_capacity * sizeof(int));
    BatchInfo *batches =
        (BatchInfo *)malloc((size_t)batch_capacity * sizeof(BatchInfo));
    if (!long_rows || !batches) {
        fprintf(stderr, "Error allocating batch arrays\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int long_count = 0;
    int batch_count = 0;
    int cur_start_row = -1;
    int cur_nnz_sum = 0;

    for (int row = 0; row < rows; row++) {
        int row_nnz = csr->row_ptr[row + 1] - csr->row_ptr[row];

        if (row_nnz > PARTIAL_MAX_BATCH_SIZE) {
            if (cur_start_row >= 0) {
                batches[batch_count].start_row = cur_start_row;
                batches[batch_count].end_row = row;
                batches[batch_count].start_nnz = csr->row_ptr[cur_start_row];
                batches[batch_count].end_nnz = csr->row_ptr[row];
                batch_count++;
                cur_start_row = -1;
                cur_nnz_sum = 0;
            }
            long_rows[long_count++] = row;
            continue;
        }

        if (cur_start_row < 0) {
            cur_start_row = row;
            cur_nnz_sum = row_nnz;
        } else if (cur_nnz_sum + row_nnz > PARTIAL_MAX_BATCH_SIZE) {
            batches[batch_count].start_row = cur_start_row;
            batches[batch_count].end_row = row;
            batches[batch_count].start_nnz = csr->row_ptr[cur_start_row];
            batches[batch_count].end_nnz = csr->row_ptr[row];
            batch_count++;
            cur_start_row = row;
            cur_nnz_sum = row_nnz;
        } else {
            cur_nnz_sum += row_nnz;
        }
    }

    if (cur_start_row >= 0) {
        batches[batch_count].start_row = cur_start_row;
        batches[batch_count].end_row = rows;
        batches[batch_count].start_nnz = csr->row_ptr[cur_start_row];
        batches[batch_count].end_nnz = csr->row_ptr[rows];
        batch_count++;
    }

    *batches_out = batches;
    *long_rows_out = long_rows;
    *long_row_count_out = long_count;
    return batch_count;
}

template <int VECTOR_SIZE>
__device__ inline void partial_vector_compute(const BatchInfo batch,
                                              const int *__restrict__ row_ptr,
                                              const int *__restrict__ scol,
                                              const float *__restrict__ sval,
                                              const float *__restrict__ x,
                                              float *__restrict__ y) {
    int tid = threadIdx.x;
    int vector_id = tid / VECTOR_SIZE;
    int vector_num = blockDim.x / VECTOR_SIZE;
    int lane_id = tid & (VECTOR_SIZE - 1);

    int batch_start_nnz = batch.start_nnz;

    for (int row = batch.start_row + vector_id; row < batch.end_row;
         row += vector_num) {
        int nz_start = row_ptr[row];
        int nz_end = row_ptr[row + 1];

        float sum = 0.0f;
        for (int j = nz_start + lane_id; j < nz_end; j += VECTOR_SIZE) {
            int local = j - batch_start_nnz;
            sum += sval[local] * x[scol[local]];
        }

        if (VECTOR_SIZE > 1) {
            unsigned mask = __activemask();
#pragma unroll
            for (int off = VECTOR_SIZE >> 1; off > 0; off >>= 1) {
                sum += __shfl_down_sync(mask, sum, off, VECTOR_SIZE);
            }
        }

        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

__device__ inline void partial_dispatch_compute(const BatchInfo batch,
                                                const int *__restrict__ row_ptr,
                                                const int *__restrict__ scol,
                                                const float *__restrict__ sval,
                                                const float *__restrict__ x,
                                                float *__restrict__ y) {
    int row_num = batch.end_row - batch.start_row;
    int nnz_num = batch.end_nnz - batch.start_nnz;
    int mean = row_num > 0 ? (nnz_num / row_num) : 0;

    if (mean < PARTIAL_SWITCH_POINT_1) {
        partial_vector_compute<1>(batch, row_ptr, scol, sval, x, y);
    } else if (mean < PARTIAL_SWITCH_POINT_2) {
        partial_vector_compute<2>(batch, row_ptr, scol, sval, x, y);
    } else if (mean < PARTIAL_SWITCH_POINT_4) {
        partial_vector_compute<4>(batch, row_ptr, scol, sval, x, y);
    } else if (mean < PARTIAL_SWITCH_POINT_8) {
        partial_vector_compute<8>(batch, row_ptr, scol, sval, x, y);
    } else if (mean < PARTIAL_SWITCH_POINT_16) {
        partial_vector_compute<16>(batch, row_ptr, scol, sval, x, y);
    } else {
        partial_vector_compute<32>(batch, row_ptr, scol, sval, x, y);
    }
}

__global__ void csr_partial_overlap_kernel(const BatchInfo *__restrict__ batches,
                                           int total_batches,
                                           int block_batch_num,
                                           const int *__restrict__ row_ptr,
                                           const int *__restrict__ col_idx,
                                           const float *__restrict__ values,
                                           const float *__restrict__ x,
                                           float *__restrict__ y) {
    __shared__ float sval[PARTIAL_MAX_BATCH_SIZE * 2];
    __shared__ int scol[PARTIAL_MAX_BATCH_SIZE * 2];

    int start_batch = blockIdx.x * block_batch_num;
    int end_batch = start_batch + block_batch_num;
    if (end_batch > total_batches) {
        end_batch = total_batches;
    }
    if (end_batch <= start_batch) {
        return;
    }

    int my_batch_num = end_batch - start_batch;

    auto block = cg::this_thread_block();
    alignas(cuda::pipeline_shared_state<cuda::thread_scope_block, 2>)
        __shared__ unsigned char pss_storage
            [sizeof(cuda::pipeline_shared_state<cuda::thread_scope_block, 2>)];
    auto pss =
        reinterpret_cast<cuda::pipeline_shared_state<cuda::thread_scope_block, 2> *>(
            pss_storage);
    if (threadIdx.x == 0) {
        new (pss) cuda::pipeline_shared_state<cuda::thread_scope_block, 2>();
    }
    __syncthreads();
    auto pipeline = cuda::make_pipeline(block, pss);

    const int shared_off[2] = {0, PARTIAL_MAX_BATCH_SIZE};

    int fetch_idx = 0;
    {
        BatchInfo b = batches[start_batch + fetch_idx];
        int len = b.end_nnz - b.start_nnz;
        int off = shared_off[fetch_idx & 1];

        pipeline.producer_acquire();
        cuda::memcpy_async(block, sval + off, values + b.start_nnz,
                           sizeof(float) * len, pipeline);
        cuda::memcpy_async(block, scol + off, col_idx + b.start_nnz,
                           sizeof(int) * len, pipeline);
        pipeline.producer_commit();
        fetch_idx++;
    }

    for (int compute_idx = 0; compute_idx < my_batch_num; compute_idx++) {
        if (fetch_idx < my_batch_num) {
            BatchInfo b = batches[start_batch + fetch_idx];
            int len = b.end_nnz - b.start_nnz;
            int off = shared_off[fetch_idx & 1];

            pipeline.producer_acquire();
            cuda::memcpy_async(block, sval + off, values + b.start_nnz,
                               sizeof(float) * len, pipeline);
            cuda::memcpy_async(block, scol + off, col_idx + b.start_nnz,
                               sizeof(int) * len, pipeline);
            pipeline.producer_commit();
            fetch_idx++;
        }

        pipeline.consumer_wait();
        block.sync();

        BatchInfo cur = batches[start_batch + compute_idx];
        int off = shared_off[compute_idx & 1];
        partial_dispatch_compute(cur, row_ptr, scol + off, sval + off, x, y);

        block.sync();
        pipeline.consumer_release();
    }
}

__global__ void partial_long_row_kernel(const int *__restrict__ long_rows,
                                        int long_row_num,
                                        const int *__restrict__ row_ptr,
                                        const int *__restrict__ col_idx,
                                        const float *__restrict__ values,
                                        const float *__restrict__ x,
                                        float *__restrict__ y) {
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int warps = blockDim.x >> 5;

    for (int idx = blockIdx.x; idx < long_row_num; idx += gridDim.x) {
        int row = long_rows[idx];
        int nz_start = row_ptr[row];
        int nz_end = row_ptr[row + 1];

        float sum = 0.0f;
        for (int j = nz_start + tid; j < nz_end; j += blockDim.x) {
            sum += values[j] * x[col_idx[j]];
        }

        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, off);
        }

        if (lane == 0) {
            warp_sums[warp_id] = sum;
        }
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane < warps) ? warp_sums[lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffff, v, off);
            }
            if (lane == 0) {
                y[row] = v;
            }
        }
        __syncthreads();
    }
}

struct CsrPartialOverlapRunner : BaseRunner {
    explicit CsrPartialOverlapRunner(int rank_) : BaseRunner(rank_) {}

    BatchInfo *d_batches = NULL;
    int *d_long_rows = NULL;
    int total_batches = 0;
    int long_row_count = 0;
    int block_batch_num = 0;
    int blocks = 0;
    int sm_count = 0;

    const char *name() const override {
        return "partial";
    }

    double prep(const CSR_Matrix &h_csr,
                const CSR_Matrix &,
                float *,
                float *) override {
        double start = now_seconds();

        cudaDeviceProp prop;
        int device = 0;
        CHECK_CUDA(cudaGetDevice(&device), rank);
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device), rank);
        sm_count = prop.multiProcessorCount;

        BatchInfo *h_batches = NULL;
        int *h_long = NULL;
        total_batches =
            build_partial_batches(&h_csr, &h_batches, &h_long, &long_row_count);

        const int target_block_batch_num = 4;
        if (total_batches > 0) {
            blocks = (total_batches + target_block_batch_num - 1) /
                     target_block_batch_num;
            const int block_floor = 2 * sm_count;
            if (blocks < block_floor) {
                blocks = block_floor;
            }
            if (blocks > total_batches) {
                blocks = total_batches;
            }
            block_batch_num = (total_batches + blocks - 1) / blocks;

            CHECK_CUDA(cudaMalloc((void **)&d_batches,
                                  (size_t)total_batches * sizeof(BatchInfo)),
                       rank);
            CHECK_CUDA(cudaMemcpy(d_batches, h_batches,
                                  (size_t)total_batches * sizeof(BatchInfo),
                                  cudaMemcpyHostToDevice),
                       rank);
        }

        if (long_row_count > 0) {
            CHECK_CUDA(cudaMalloc((void **)&d_long_rows,
                                  (size_t)long_row_count * sizeof(int)),
                       rank);
            CHECK_CUDA(cudaMemcpy(d_long_rows, h_long,
                                  (size_t)long_row_count * sizeof(int),
                                  cudaMemcpyHostToDevice),
                       rank);
        }

        free(h_batches);
        free(h_long);
        CHECK_CUDA(cudaDeviceSynchronize(), rank);
        return now_seconds() - start;
    }

    void launch(const CSR_Matrix &d_csr, const float *d_x, float *d_y) override {
        if (total_batches > 0) {
            csr_partial_overlap_kernel<<<blocks, PARTIAL_THREADS_PER_BLOCK>>>(
                d_batches, total_batches, block_batch_num,
                d_csr.row_ptr, d_csr.col_idx, d_csr.values, d_x, d_y);
        }
        if (long_row_count > 0) {
            int lblocks = long_row_count < sm_count ? long_row_count : sm_count;
            if (lblocks > 0) {
                partial_long_row_kernel<<<lblocks, 256>>>(
                    d_long_rows, long_row_count,
                    d_csr.row_ptr, d_csr.col_idx, d_csr.values, d_x, d_y);
            }
        }
    }

    void teardown() override {
        cudaFree(d_batches);
        cudaFree(d_long_rows);
        d_batches = NULL;
        d_long_rows = NULL;
        total_batches = 0;
        long_row_count = 0;
        block_batch_num = 0;
        blocks = 0;
        sm_count = 0;
    }
};

// =============================================================================
// cuSPARSE.
// =============================================================================

struct CuSparseRunner : BaseRunner {
    explicit CuSparseRunner(int rank_) : BaseRunner(rank_) {
        CHECK_CUSPARSE(cusparseCreate(&handle), rank);
    }

    ~CuSparseRunner() override {
        teardown();
        if (handle) {
            cusparseDestroy(handle);
            handle = NULL;
        }
    }

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t mat_a = NULL;
    cusparseDnVecDescr_t vec_x = NULL;
    cusparseDnVecDescr_t vec_y = NULL;
    void *d_buffer = NULL;
    float alpha = 1.0f;
    float beta = 0.0f;

    const char *name() const override {
        return "cusparse";
    }

    double prep(const CSR_Matrix &,
                const CSR_Matrix &d_csr,
                float *d_x,
                float *d_y) override {
        double start = now_seconds();

        CHECK_CUSPARSE(cusparseCreateCsr(
                           &mat_a,
                           d_csr.rows, d_csr.cols, d_csr.nnz,
                           d_csr.row_ptr, d_csr.col_idx, d_csr.values,
                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
                       rank);
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_x, d_csr.cols, d_x, CUDA_R_32F),
                       rank);
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec_y, d_csr.rows, d_y, CUDA_R_32F),
                       rank);

        size_t buffer_size = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
                           handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, mat_a, vec_x, &beta, vec_y,
                           CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &buffer_size),
                       rank);
        if (buffer_size > 0) {
            CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size), rank);
        }

        CHECK_CUDA(cudaDeviceSynchronize(), rank);
        return now_seconds() - start;
    }

    void launch(const CSR_Matrix &, const float *, float *) override {
        CHECK_CUSPARSE(cusparseSpMV(
                           handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha, mat_a, vec_x, &beta, vec_y,
                           CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, d_buffer),
                       rank);
    }

    void teardown() override {
        if (mat_a) {
            cusparseDestroySpMat(mat_a);
            mat_a = NULL;
        }
        if (vec_x) {
            cusparseDestroyDnVec(vec_x);
            vec_x = NULL;
        }
        if (vec_y) {
            cusparseDestroyDnVec(vec_y);
            vec_y = NULL;
        }
        if (d_buffer) {
            cudaFree(d_buffer);
            d_buffer = NULL;
        }
    }
};

const char *spmv_kernel_kind_name(SpmvKernelKind kind) {
    switch (kind) {
        case SPMV_KERNEL_SCALAR:
            return "scalar";
        case SPMV_KERNEL_VECTOR:
            return "vector";
        case SPMV_KERNEL_ADAPTIVE:
            return "adaptive";
        case SPMV_KERNEL_ADAPTIVE_PAPER:
            return "adaptive-paper";
        case SPMV_KERNEL_PARTIAL:
            return "partial";
        case SPMV_KERNEL_CUSPARSE:
            return "cusparse";
    }
    return "unknown";
}

int parse_spmv_kernel_kind(const char *name, SpmvKernelKind *kind) {
    if (strcmp(name, "scalar") == 0) {
        *kind = SPMV_KERNEL_SCALAR;
    } else if (strcmp(name, "vector") == 0) {
        *kind = SPMV_KERNEL_VECTOR;
    } else if (strcmp(name, "adaptive") == 0) {
        *kind = SPMV_KERNEL_ADAPTIVE;
    } else if (strcmp(name, "adaptive-paper") == 0) {
        *kind = SPMV_KERNEL_ADAPTIVE_PAPER;
    } else if (strcmp(name, "partial") == 0 ||
               strcmp(name, "partial-overlap") == 0) {
        *kind = SPMV_KERNEL_PARTIAL;
    } else if (strcmp(name, "cusparse") == 0) {
        *kind = SPMV_KERNEL_CUSPARSE;
    } else {
        return 1;
    }
    return 0;
}

SpmvKernelRunner *create_spmv_kernel_runner(SpmvKernelKind kind, int rank) {
    switch (kind) {
        case SPMV_KERNEL_SCALAR:
            return new CsrScalarRunner(rank);
        case SPMV_KERNEL_VECTOR:
            return new CsrVectorRunner(rank);
        case SPMV_KERNEL_ADAPTIVE:
            return new CsrAdaptiveRunner(rank);
        case SPMV_KERNEL_ADAPTIVE_PAPER:
            return new CsrAdaptivePaperRunner(rank);
        case SPMV_KERNEL_PARTIAL:
            return new CsrPartialOverlapRunner(rank);
        case SPMV_KERNEL_CUSPARSE:
            return new CuSparseRunner(rank);
    }
    return NULL;
}

void destroy_spmv_kernel_runner(SpmvKernelRunner *runner) {
    delete runner;
}

void print_spmv_kernel_choices(FILE *stream) {
    fprintf(stream,
            "Available kernels: scalar, vector, adaptive, adaptive-paper, "
            "partial, cusparse, all\n");
}
