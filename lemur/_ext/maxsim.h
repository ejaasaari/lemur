#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__AVX512F__)
#define LEMUR_HAS_AVX512 1
#else
#define LEMUR_HAS_AVX512 0
#endif

#if defined(__AVX2__) && defined(__FMA__) && !LEMUR_HAS_AVX512
#define LEMUR_HAS_AVX2 1
#else
#define LEMUR_HAS_AVX2 0
#endif

#if LEMUR_HAS_AVX512 || LEMUR_HAS_AVX2
#include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define LEMUR_HAS_NEON 1
#else
#define LEMUR_HAS_NEON 0
#endif

#if defined(__clang__)
#define LEMUR_UNROLL_FULL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define LEMUR_UNROLL_FULL _Pragma("GCC unroll 16")
#else
#define LEMUR_UNROLL_FULL
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

class MaxSim {
private:
  const float *train_;
  const int32_t *train_counts_;
  int vec_dim_;
  int num_train_points_;

  std::vector<int32_t> train_offsets_;
  std::vector<const float *> train_ptrs_;

  static constexpr int kFastDim48 = 48;
  static constexpr int kFastDim64 = 64;
  static constexpr int kFastDim96 = 96;
  static constexpr int kFastDim128 = 128;
  static constexpr int kMinSimdQueryVecCount = 1;
  static constexpr int kMaxSimdQueryVecCount = 48;
  static constexpr int kSimdScratchFloats = kFastDim128 * kMaxSimdQueryVecCount;

  static inline bool is_supported_simd_query_count(int32_t query_vec_count) {
    return query_vec_count >= kMinSimdQueryVecCount &&
           query_vec_count <= kMaxSimdQueryVecCount;
  }

  template <typename Runner>
  static inline void dispatch_query_bucket_48(int32_t query_vec_count,
                                              Runner &&runner) {
    if (query_vec_count <= 4) {
      runner(std::integral_constant<int, 4>{});
    } else if (query_vec_count <= 8) {
      runner(std::integral_constant<int, 8>{});
    } else if (query_vec_count <= 12) {
      runner(std::integral_constant<int, 12>{});
    } else if (query_vec_count <= 16) {
      runner(std::integral_constant<int, 16>{});
    } else if (query_vec_count <= 20) {
      runner(std::integral_constant<int, 20>{});
    } else if (query_vec_count <= 24) {
      runner(std::integral_constant<int, 24>{});
    } else if (query_vec_count <= 28) {
      runner(std::integral_constant<int, 28>{});
    } else if (query_vec_count <= 32) {
      runner(std::integral_constant<int, 32>{});
    } else if (query_vec_count <= 36) {
      runner(std::integral_constant<int, 36>{});
    } else if (query_vec_count <= 40) {
      runner(std::integral_constant<int, 40>{});
    } else if (query_vec_count <= 44) {
      runner(std::integral_constant<int, 44>{});
    } else {
      runner(std::integral_constant<int, 48>{});
    }
  }

  template <typename Runner>
  inline bool dispatch_fast_dim(Runner &&runner) const {
    switch (vec_dim_) {
    case kFastDim48:
      runner(std::integral_constant<int, kFastDim48>{});
      return true;
    case kFastDim64:
      runner(std::integral_constant<int, kFastDim64>{});
      return true;
    case kFastDim96:
      runner(std::integral_constant<int, kFastDim96>{});
      return true;
    case kFastDim128:
      runner(std::integral_constant<int, kFastDim128>{});
      return true;
    default:
      return false;
    }
  }

  enum class CandidatePrefetch { kNone, kX86, kNeon };

  template <CandidatePrefetch Prefetch, typename ScoreOne>
  inline void score_candidates_loop(const int *__restrict row, int num_indices,
                                    std::pair<float, int> *__restrict cand,
                                    ScoreOne &&score_one) const {
    for (int j = 0; j < num_indices; ++j) {
      const int idx = row[j];
      if (j + 1 < num_indices) {
#if LEMUR_HAS_AVX512 || LEMUR_HAS_AVX2
        if constexpr (Prefetch == CandidatePrefetch::kX86)
          _mm_prefetch((const char *)train_ptrs_[row[j + 1]], _MM_HINT_T0);
#endif
#if LEMUR_HAS_NEON
        if constexpr (Prefetch == CandidatePrefetch::kNeon)
          __builtin_prefetch(train_ptrs_[row[j + 1]], 0, 3);
#endif
      }
      cand[(size_t)j] = {score_one(idx), idx};
    }
  }

public:
  static inline float dot_product(const float *x1, const float *x2,
                                  size_t length) {
#if LEMUR_HAS_AVX512
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= length; i += 16) {
      __m512 v1 = _mm512_loadu_ps(x1 + i);
      __m512 v2 = _mm512_loadu_ps(x2 + i);
      sum = _mm512_fmadd_ps(v1, v2, sum);
    }
    if (i < length) {
      const uint32_t rem = static_cast<uint32_t>(length - i);
      const __mmask16 m = static_cast<__mmask16>((1u << rem) - 1u);
      __m512 v1 = _mm512_maskz_loadu_ps(m, x1 + i);
      __m512 v2 = _mm512_maskz_loadu_ps(m, x2 + i);
      sum = _mm512_fmadd_ps(v1, v2, sum);
    }

    auto sumh = _mm256_add_ps(_mm512_castps512_ps256(sum),
                              _mm512_extractf32x8_ps(sum, 1));
    auto sumhh = _mm_add_ps(_mm256_castps256_ps128(sumh),
                            _mm256_extractf128_ps(sumh, 1));
    auto tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
    auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
    return _mm_cvtss_f32(tmp2);
#else
    float sum = 0.0f;
    for (size_t i = 0; i < length; ++i)
      sum += x1[i] * x2[i];
    return sum;
#endif
  }

private:
#if LEMUR_HAS_AVX512
  static inline float hsum512_ps(__m512 v) {
    __m256 sumh =
        _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
    __m128 sumhh = _mm_add_ps(_mm256_castps256_ps128(sumh),
                              _mm256_extractf128_ps(sumh, 1));
    __m128 tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
    __m128 tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
    return _mm_cvtss_f32(tmp2);
  }

  template <int D>
  static inline void transpose_query_32(const float *__restrict query,
                                        float *__restrict qT) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_assume_aligned(qT, 64);
#endif
    for (int d = 0; d < D; ++d) {
      const float *__restrict col = query + d;
      float *__restrict out = qT + d * 32;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 32
#endif
      for (int q = 0; q < 32; ++q) {
        out[q] = col[(int64_t)q * D];
      }
    }
  }

  template <int D>
  static inline float maxsim_Dx32_qT_2x(const float *__restrict qT,
                                        const float *__restrict train_block,
                                        int count) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    __m512 best0 = _mm512_set1_ps(neg_inf);
    __m512 best1 = _mm512_set1_ps(neg_inf);

#if defined(__GNUC__) || defined(__clang__)
    __builtin_assume_aligned(qT, 64);
#endif

    int tv = 0;
    for (; tv + 1 < count; tv += 2) {
      const float *__restrict x0 = train_block + (int64_t)tv * D;
      const float *__restrict x1 = x0 + D;

      __m512 acc00 = _mm512_setzero_ps();
      __m512 acc01 = _mm512_setzero_ps();
      __m512 acc10 = _mm512_setzero_ps();
      __m512 acc11 = _mm512_setzero_ps();

      for (int d = 0; d < D; d += 4) {
        __m512 q00 = _mm512_load_ps(qT + (d + 0) * 32);
        __m512 q01 = _mm512_load_ps(qT + (d + 0) * 32 + 16);
        __m512 bx00 = _mm512_set1_ps(x0[d + 0]);
        __m512 bx10 = _mm512_set1_ps(x1[d + 0]);
        acc00 = _mm512_fmadd_ps(q00, bx00, acc00);
        acc01 = _mm512_fmadd_ps(q01, bx00, acc01);
        acc10 = _mm512_fmadd_ps(q00, bx10, acc10);
        acc11 = _mm512_fmadd_ps(q01, bx10, acc11);

        __m512 q10 = _mm512_load_ps(qT + (d + 1) * 32);
        __m512 q11 = _mm512_load_ps(qT + (d + 1) * 32 + 16);
        __m512 bx01 = _mm512_set1_ps(x0[d + 1]);
        __m512 bx11 = _mm512_set1_ps(x1[d + 1]);
        acc00 = _mm512_fmadd_ps(q10, bx01, acc00);
        acc01 = _mm512_fmadd_ps(q11, bx01, acc01);
        acc10 = _mm512_fmadd_ps(q10, bx11, acc10);
        acc11 = _mm512_fmadd_ps(q11, bx11, acc11);

        __m512 q20 = _mm512_load_ps(qT + (d + 2) * 32);
        __m512 q21 = _mm512_load_ps(qT + (d + 2) * 32 + 16);
        __m512 bx02 = _mm512_set1_ps(x0[d + 2]);
        __m512 bx12 = _mm512_set1_ps(x1[d + 2]);
        acc00 = _mm512_fmadd_ps(q20, bx02, acc00);
        acc01 = _mm512_fmadd_ps(q21, bx02, acc01);
        acc10 = _mm512_fmadd_ps(q20, bx12, acc10);
        acc11 = _mm512_fmadd_ps(q21, bx12, acc11);

        __m512 q30 = _mm512_load_ps(qT + (d + 3) * 32);
        __m512 q31 = _mm512_load_ps(qT + (d + 3) * 32 + 16);
        __m512 bx03 = _mm512_set1_ps(x0[d + 3]);
        __m512 bx13 = _mm512_set1_ps(x1[d + 3]);
        acc00 = _mm512_fmadd_ps(q30, bx03, acc00);
        acc01 = _mm512_fmadd_ps(q31, bx03, acc01);
        acc10 = _mm512_fmadd_ps(q30, bx13, acc10);
        acc11 = _mm512_fmadd_ps(q31, bx13, acc11);
      }

      best0 = _mm512_max_ps(best0, acc00);
      best1 = _mm512_max_ps(best1, acc01);
      best0 = _mm512_max_ps(best0, acc10);
      best1 = _mm512_max_ps(best1, acc11);
    }

    if (tv < count) {
      const float *__restrict x0 = train_block + (int64_t)tv * D;
      __m512 acc00 = _mm512_setzero_ps();
      __m512 acc01 = _mm512_setzero_ps();

      for (int d = 0; d < D; d += 4) {
        __m512 q00 = _mm512_load_ps(qT + (d + 0) * 32);
        __m512 q01 = _mm512_load_ps(qT + (d + 0) * 32 + 16);
        __m512 bx00 = _mm512_set1_ps(x0[d + 0]);
        acc00 = _mm512_fmadd_ps(q00, bx00, acc00);
        acc01 = _mm512_fmadd_ps(q01, bx00, acc01);

        __m512 q10 = _mm512_load_ps(qT + (d + 1) * 32);
        __m512 q11 = _mm512_load_ps(qT + (d + 1) * 32 + 16);
        __m512 bx01 = _mm512_set1_ps(x0[d + 1]);
        acc00 = _mm512_fmadd_ps(q10, bx01, acc00);
        acc01 = _mm512_fmadd_ps(q11, bx01, acc01);

        __m512 q20 = _mm512_load_ps(qT + (d + 2) * 32);
        __m512 q21 = _mm512_load_ps(qT + (d + 2) * 32 + 16);
        __m512 bx02 = _mm512_set1_ps(x0[d + 2]);
        acc00 = _mm512_fmadd_ps(q20, bx02, acc00);
        acc01 = _mm512_fmadd_ps(q21, bx02, acc01);

        __m512 q30 = _mm512_load_ps(qT + (d + 3) * 32);
        __m512 q31 = _mm512_load_ps(qT + (d + 3) * 32 + 16);
        __m512 bx03 = _mm512_set1_ps(x0[d + 3]);
        acc00 = _mm512_fmadd_ps(q30, bx03, acc00);
        acc01 = _mm512_fmadd_ps(q31, bx03, acc01);
      }

      best0 = _mm512_max_ps(best0, acc00);
      best1 = _mm512_max_ps(best1, acc01);
    }

    return hsum512_ps(best0) + hsum512_ps(best1);
  }

  static inline __mmask16 avx512_lane_mask(int lanes) {
    if (lanes >= 16)
      return static_cast<__mmask16>(0xffffu);
    if (lanes <= 0)
      return static_cast<__mmask16>(0);
    return static_cast<__mmask16>((1u << lanes) - 1u);
  }

  template <int QB>
  static inline __m512 load_qT_block_avx512(const float *__restrict qT,
                                            int64_t offset, int b) {
    const int lanes = QB - b * 16;
    return _mm512_maskz_loadu_ps(avx512_lane_mask(lanes), qT + offset + b * 16);
  }

  template <int D, int QB>
  static inline void transpose_query_avx512(const float *__restrict query,
                                            int32_t query_vec_count,
                                            float *__restrict qT) {
    for (int d = 0; d < D; ++d) {
      const float *__restrict col = query + d;
      float *__restrict out = qT + (int64_t)d * QB;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 48
#endif
      for (int32_t q = 0; q < query_vec_count; ++q)
        out[q] = col[(int64_t)q * D];
      for (int32_t q = query_vec_count; q < QB; ++q)
        out[q] = 0.0f;
    }
  }

  template <int B>
  static inline void zero_avx512_accumulators(__m512 (&acc)[B]) {
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) acc[b] = _mm512_setzero_ps();
  }

  template <int B> static inline void init_avx512_best(__m512 (&best)[B]) {
    const __m512 neg_inf =
        _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) best[b] = neg_inf;
  }

  template <int D, int QB>
  static inline float
  maxsim_transposed_2x_avx512(const float *__restrict qT,
                              int32_t query_vec_count,
                              const float *__restrict train_block, int count) {
    static_assert(D % 4 == 0,
                  "AVX-512 MaxSim dimensions must be divisible by 4");
    static_assert(QB % 4 == 0, "AVX-512 query bucket must be divisible by 4");
    constexpr int B = (QB + 15) / 16;
    __m512 best[B];
    init_avx512_best(best);

    int tv = 0;
    for (; tv + 1 < count; tv += 2) {
      const float *__restrict x0 = train_block + (int64_t)tv * D;
      const float *__restrict x1 = x0 + D;
      __m512 acc0[B];
      __m512 acc1[B];
      zero_avx512_accumulators(acc0);
      zero_avx512_accumulators(acc1);

      for (int d = 0; d < D; d += 4) {
#define LEMUR_AVX512_FMA_TRANSPOSED_PAIR(DOFF)                                 \
  do {                                                                         \
    const __m512 x0d = _mm512_set1_ps(x0[d + DOFF]);                           \
    const __m512 x1d = _mm512_set1_ps(x1[d + DOFF]);                           \
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) {                            \
      const __m512 qv =                                                        \
          load_qT_block_avx512<QB>(qT, (int64_t)(d + DOFF) * QB, b);           \
      acc0[b] = _mm512_fmadd_ps(qv, x0d, acc0[b]);                             \
      acc1[b] = _mm512_fmadd_ps(qv, x1d, acc1[b]);                             \
    }                                                                          \
  } while (false)
        LEMUR_AVX512_FMA_TRANSPOSED_PAIR(0);
        LEMUR_AVX512_FMA_TRANSPOSED_PAIR(1);
        LEMUR_AVX512_FMA_TRANSPOSED_PAIR(2);
        LEMUR_AVX512_FMA_TRANSPOSED_PAIR(3);
#undef LEMUR_AVX512_FMA_TRANSPOSED_PAIR
      }

      LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) best[b] =
          _mm512_max_ps(best[b], _mm512_max_ps(acc0[b], acc1[b]));
    }

    if (tv < count) {
      const float *__restrict x = train_block + (int64_t)tv * D;
      __m512 acc[B];
      zero_avx512_accumulators(acc);

      for (int d = 0; d < D; d += 4) {
#define LEMUR_AVX512_FMA_TRANSPOSED_SINGLE(DOFF)                               \
  do {                                                                         \
    const __m512 xd = _mm512_set1_ps(x[d + DOFF]);                             \
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) {                            \
      const __m512 qv =                                                        \
          load_qT_block_avx512<QB>(qT, (int64_t)(d + DOFF) * QB, b);           \
      acc[b] = _mm512_fmadd_ps(qv, xd, acc[b]);                                \
    }                                                                          \
  } while (false)
        LEMUR_AVX512_FMA_TRANSPOSED_SINGLE(0);
        LEMUR_AVX512_FMA_TRANSPOSED_SINGLE(1);
        LEMUR_AVX512_FMA_TRANSPOSED_SINGLE(2);
        LEMUR_AVX512_FMA_TRANSPOSED_SINGLE(3);
#undef LEMUR_AVX512_FMA_TRANSPOSED_SINGLE
      }

      LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) best[b] =
          _mm512_max_ps(best[b], acc[b]);
    }

    alignas(64) float values[B * 16];
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b)
        _mm512_store_ps(values + b * 16, best[b]);
    float total = 0.0f;
    for (int32_t q = 0; q < query_vec_count; ++q)
      total += values[q];
    return total;
  }

  template <int D>
  inline void
  score_candidates_Dx32_avx512(const float *__restrict query,
                               const int *__restrict row, int num_indices,
                               std::pair<float, int> *__restrict cand,
                               float *__restrict qT_scratch) const {
    transpose_query_32<D>(query, qT_scratch);
    score_candidates_loop<CandidatePrefetch::kX86>(
        row, num_indices, cand, [&](int idx) {
          return maxsim_Dx32_qT_2x<D>(qT_scratch, train_ptrs_[idx],
                                      train_counts_[idx]);
        });
  }

  template <int D, int QB>
  inline void score_candidates_transposed_2x_avx512(
      const float *__restrict query, int32_t query_vec_count,
      const int *__restrict row, int num_indices,
      std::pair<float, int> *__restrict cand,
      float *__restrict qT_scratch) const {
    transpose_query_avx512<D, QB>(query, query_vec_count, qT_scratch);
    score_candidates_loop<CandidatePrefetch::kX86>(
        row, num_indices, cand, [&](int idx) {
          return maxsim_transposed_2x_avx512<D, QB>(qT_scratch, query_vec_count,
                                                    train_ptrs_[idx],
                                                    train_counts_[idx]);
        });
  }

  template <int D>
  inline void
  dispatch_transposed_2x_avx512(const float *__restrict query,
                                int32_t query_vec_count,
                                const int *__restrict row, int num_indices,
                                std::pair<float, int> *__restrict cand,
                                float *__restrict qT_scratch) const {
    dispatch_query_bucket_48(query_vec_count, [&](auto bucket) {
      constexpr int QB = decltype(bucket)::value;
      score_candidates_transposed_2x_avx512<D, QB>(
          query, query_vec_count, row, num_indices, cand, qT_scratch);
    });
  }

  template <int D>
  inline void
  score_candidates_avx512_dim(const float *__restrict query,
                              int32_t query_vec_count,
                              const int *__restrict row, int num_indices,
                              std::pair<float, int> *__restrict cand,
                              float *__restrict qT_scratch) const {
    if (query_vec_count == 32)
      score_candidates_Dx32_avx512<D>(query, row, num_indices, cand,
                                      qT_scratch);
    else
      dispatch_transposed_2x_avx512<D>(query, query_vec_count, row, num_indices,
                                       cand, qT_scratch);
  }

  inline bool score_candidates_avx512(const float *__restrict query,
                                      int32_t query_vec_count,
                                      const int *__restrict row,
                                      int num_indices,
                                      std::pair<float, int> *__restrict cand,
                                      float *__restrict qT_scratch) const {
    if (!is_supported_simd_query_count(query_vec_count))
      return false;

    return dispatch_fast_dim([&](auto dim) {
      constexpr int D = decltype(dim)::value;
      score_candidates_avx512_dim<D>(query, query_vec_count, row, num_indices,
                                     cand, qT_scratch);
    });
  }
#endif

#if LEMUR_HAS_AVX2
  template <int D, int QB>
  static inline void transpose_query_avx2(const float *__restrict query,
                                          int32_t query_vec_count,
                                          float *__restrict qT) {
    static_assert(QB % 4 == 0, "AVX2 query bucket must be divisible by 4");
    constexpr int B = (QB + 7) / 8;
    constexpr int QS = B * 8;
    static_assert(QS <= 48, "AVX2 query stride must fit scratch buffer");
#if defined(__GNUC__) || defined(__clang__)
    __builtin_assume_aligned(qT, 32);
#endif
    for (int d = 0; d < D; ++d) {
      const float *__restrict col = query + d;
      float *__restrict out = qT + (int64_t)d * QS;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 48
#endif
      for (int32_t q = 0; q < query_vec_count; ++q)
        out[q] = col[(int64_t)q * D];
      for (int32_t q = query_vec_count; q < QS; ++q)
        out[q] = 0.0f;
    }
  }

  template <int B> static inline void zero_avx2_accumulators(__m256 (&acc)[B]) {
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) acc[b] = _mm256_setzero_ps();
  }

  template <int B> static inline void init_avx2_best(__m256 (&best)[B]) {
    const __m256 neg_inf =
        _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) best[b] = neg_inf;
  }

  template <int D, int QB>
  static inline float
  maxsim_transposed_2x_avx2(const float *__restrict qT, int32_t query_vec_count,
                            const float *__restrict train_block, int count) {
    static_assert(D % 4 == 0, "AVX2 MaxSim dimensions must be divisible by 4");
    static_assert(QB % 4 == 0, "AVX2 query bucket must be divisible by 4");
    constexpr int B = (QB + 7) / 8;
    constexpr int QS = B * 8;
    __m256 best[B];
    init_avx2_best(best);

#if defined(__GNUC__) || defined(__clang__)
    __builtin_assume_aligned(qT, 32);
#endif

    int tv = 0;
    for (; tv + 1 < count; tv += 2) {
      const float *__restrict x0 = train_block + (int64_t)tv * D;
      const float *__restrict x1 = x0 + D;
      __m256 acc0[B];
      __m256 acc1[B];
      zero_avx2_accumulators(acc0);
      zero_avx2_accumulators(acc1);

      for (int d = 0; d < D; d += 4) {
#define LEMUR_AVX2_FMA_TRANSPOSED_PAIR(DOFF)                                   \
  do {                                                                         \
    const __m256 x0d = _mm256_set1_ps(x0[d + DOFF]);                           \
    const __m256 x1d = _mm256_set1_ps(x1[d + DOFF]);                           \
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) {                            \
      const __m256 qv = _mm256_load_ps(qT + (int64_t)(d + DOFF) * QS + b * 8); \
      acc0[b] = _mm256_fmadd_ps(qv, x0d, acc0[b]);                             \
      acc1[b] = _mm256_fmadd_ps(qv, x1d, acc1[b]);                             \
    }                                                                          \
  } while (false)
        LEMUR_AVX2_FMA_TRANSPOSED_PAIR(0);
        LEMUR_AVX2_FMA_TRANSPOSED_PAIR(1);
        LEMUR_AVX2_FMA_TRANSPOSED_PAIR(2);
        LEMUR_AVX2_FMA_TRANSPOSED_PAIR(3);
#undef LEMUR_AVX2_FMA_TRANSPOSED_PAIR
      }

      LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) best[b] =
          _mm256_max_ps(best[b], _mm256_max_ps(acc0[b], acc1[b]));
    }

    if (tv < count) {
      const float *__restrict x = train_block + (int64_t)tv * D;
      __m256 acc[B];
      zero_avx2_accumulators(acc);

      for (int d = 0; d < D; d += 4) {
#define LEMUR_AVX2_FMA_TRANSPOSED_SINGLE(DOFF)                                 \
  do {                                                                         \
    const __m256 xd = _mm256_set1_ps(x[d + DOFF]);                             \
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) {                            \
      const __m256 qv = _mm256_load_ps(qT + (int64_t)(d + DOFF) * QS + b * 8); \
      acc[b] = _mm256_fmadd_ps(qv, xd, acc[b]);                                \
    }                                                                          \
  } while (false)
        LEMUR_AVX2_FMA_TRANSPOSED_SINGLE(0);
        LEMUR_AVX2_FMA_TRANSPOSED_SINGLE(1);
        LEMUR_AVX2_FMA_TRANSPOSED_SINGLE(2);
        LEMUR_AVX2_FMA_TRANSPOSED_SINGLE(3);
#undef LEMUR_AVX2_FMA_TRANSPOSED_SINGLE
      }

      LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) best[b] =
          _mm256_max_ps(best[b], acc[b]);
    }

    alignas(32) float values[B * 8];
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b)
        _mm256_store_ps(values + b * 8, best[b]);
    float total = 0.0f;
    for (int32_t q = 0; q < query_vec_count; ++q)
      total += values[q];
    return total;
  }

  template <int D, int QB>
  inline void score_candidates_transposed_2x_avx2(
      const float *__restrict query, int32_t query_vec_count,
      const int *__restrict row, int num_indices,
      std::pair<float, int> *__restrict cand,
      float *__restrict qT_scratch) const {
    transpose_query_avx2<D, QB>(query, query_vec_count, qT_scratch);
    score_candidates_loop<CandidatePrefetch::kX86>(
        row, num_indices, cand, [&](int idx) {
          return maxsim_transposed_2x_avx2<D, QB>(qT_scratch, query_vec_count,
                                                  train_ptrs_[idx],
                                                  train_counts_[idx]);
        });
  }

  template <int D>
  inline void
  dispatch_transposed_2x_avx2(const float *__restrict query,
                              int32_t query_vec_count,
                              const int *__restrict row, int num_indices,
                              std::pair<float, int> *__restrict cand,
                              float *__restrict qT_scratch) const {
    dispatch_query_bucket_48(query_vec_count, [&](auto bucket) {
      constexpr int QB = decltype(bucket)::value;
      score_candidates_transposed_2x_avx2<D, QB>(query, query_vec_count, row,
                                                 num_indices, cand, qT_scratch);
    });
  }

  template <int D>
  inline void score_candidates_avx2_dim(const float *__restrict query,
                                        int32_t query_vec_count,
                                        const int *__restrict row,
                                        int num_indices,
                                        std::pair<float, int> *__restrict cand,
                                        float *__restrict qT_scratch) const {
    dispatch_transposed_2x_avx2<D>(query, query_vec_count, row, num_indices,
                                   cand, qT_scratch);
  }

  inline bool score_candidates_avx2(const float *__restrict query,
                                    int32_t query_vec_count,
                                    const int *__restrict row, int num_indices,
                                    std::pair<float, int> *__restrict cand,
                                    float *__restrict qT_scratch) const {
    if (!is_supported_simd_query_count(query_vec_count))
      return false;

    return dispatch_fast_dim([&](auto dim) {
      constexpr int D = decltype(dim)::value;
      score_candidates_avx2_dim<D>(query, query_vec_count, row, num_indices,
                                   cand, qT_scratch);
    });
  }
#endif

#if LEMUR_HAS_NEON
  template <int D, int QB>
  static inline void transpose_query_neon(const float *__restrict query,
                                          int32_t query_vec_count,
                                          float *__restrict qT) {
    int32_t q = 0;
    for (; q + 4 <= query_vec_count; q += 4) {
      const float *__restrict r0 = query + (int64_t)(q + 0) * D;
      const float *__restrict r1 = query + (int64_t)(q + 1) * D;
      const float *__restrict r2 = query + (int64_t)(q + 2) * D;
      const float *__restrict r3 = query + (int64_t)(q + 3) * D;

      for (int d = 0; d < D; d += 4) {
        const float32x4_t a = vld1q_f32(r0 + d);
        const float32x4_t b = vld1q_f32(r1 + d);
        const float32x4_t c = vld1q_f32(r2 + d);
        const float32x4_t e = vld1q_f32(r3 + d);
        const float32x4_t ab0 = vtrn1q_f32(a, b);
        const float32x4_t ab1 = vtrn2q_f32(a, b);
        const float32x4_t ce0 = vtrn1q_f32(c, e);
        const float32x4_t ce1 = vtrn2q_f32(c, e);

        vst1q_f32(qT + (int64_t)(d + 0) * QB + q,
                  vcombine_f32(vget_low_f32(ab0), vget_low_f32(ce0)));
        vst1q_f32(qT + (int64_t)(d + 1) * QB + q,
                  vcombine_f32(vget_low_f32(ab1), vget_low_f32(ce1)));
        vst1q_f32(qT + (int64_t)(d + 2) * QB + q,
                  vcombine_f32(vget_high_f32(ab0), vget_high_f32(ce0)));
        vst1q_f32(qT + (int64_t)(d + 3) * QB + q,
                  vcombine_f32(vget_high_f32(ab1), vget_high_f32(ce1)));
      }
    }

    for (; q < query_vec_count; ++q) {
      const float *__restrict row = query + (int64_t)q * D;
      for (int d = 0; d < D; ++d)
        qT[(int64_t)d * QB + q] = row[d];
    }

    for (int d = 0; d < D; ++d)
      for (int32_t pad = query_vec_count; pad < QB; ++pad)
        qT[(int64_t)d * QB + pad] = 0.0f;
  }

  template <int B>
  static inline void zero_neon_accumulators(float32x4_t (&acc)[B]) {
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (int b = 0; b < B; ++b)
      acc[b] = vdupq_n_f32(0.0f);
  }

  template <int B> static inline void init_neon_best(float32x4_t (&best)[B]) {
    const float32x4_t neg_inf =
        vdupq_n_f32(-std::numeric_limits<float>::infinity());
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (int b = 0; b < B; ++b)
      best[b] = neg_inf;
  }

  template <int D, int QB, int DimUnroll>
  static inline float
  maxsim_transposed_2x_neon(const float *__restrict qT, int32_t query_vec_count,
                            const float *__restrict train_block, int count) {
    constexpr int B = QB / 4;
    float32x4_t best[B];
    init_neon_best(best);

    int tv = 0;
    for (; tv + 1 < count; tv += 2) {
      const float *__restrict x0 = train_block + (int64_t)tv * D;
      const float *__restrict x1 = x0 + D;
      float32x4_t acc0[B];
      float32x4_t acc1[B];
      zero_neon_accumulators(acc0);
      zero_neon_accumulators(acc1);

      if constexpr (DimUnroll == 4) {
        for (int d = 0; d < D; d += 4) {
#define LEMUR_NEON_FMA_TRANSPOSED_PAIR(DOFF)                                   \
  do {                                                                         \
    const float x0d = x0[d + DOFF];                                            \
    const float x1d = x1[d + DOFF];                                            \
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) {                            \
      const float32x4_t qv = vld1q_f32(qT + (int64_t)(d + DOFF) * QB + b * 4); \
      acc0[b] = vfmaq_n_f32(acc0[b], qv, x0d);                                 \
      acc1[b] = vfmaq_n_f32(acc1[b], qv, x1d);                                 \
    }                                                                          \
  } while (false)
          LEMUR_NEON_FMA_TRANSPOSED_PAIR(0);
          LEMUR_NEON_FMA_TRANSPOSED_PAIR(1);
          LEMUR_NEON_FMA_TRANSPOSED_PAIR(2);
          LEMUR_NEON_FMA_TRANSPOSED_PAIR(3);
#undef LEMUR_NEON_FMA_TRANSPOSED_PAIR
        }
      } else {
        for (int d = 0; d < D; ++d) {
          const float x0d = x0[d];
          const float x1d = x1[d];
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
          for (int b = 0; b < B; ++b) {
            const float32x4_t qv = vld1q_f32(qT + (int64_t)d * QB + b * 4);
            acc0[b] = vfmaq_n_f32(acc0[b], qv, x0d);
            acc1[b] = vfmaq_n_f32(acc1[b], qv, x1d);
          }
        }
      }

#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
      for (int b = 0; b < B; ++b)
        best[b] = vmaxq_f32(best[b], vmaxq_f32(acc0[b], acc1[b]));
    }

    if (tv < count) {
      const float *__restrict x = train_block + (int64_t)tv * D;
      float32x4_t acc[B];
      zero_neon_accumulators(acc);
      for (int d = 0; d < D; ++d) {
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
        for (int b = 0; b < B; ++b)
          acc[b] = vfmaq_n_f32(acc[b], vld1q_f32(qT + (int64_t)d * QB + b * 4),
                               x[d]);
      }
      for (int b = 0; b < B; ++b)
        best[b] = vmaxq_f32(best[b], acc[b]);
    }

    alignas(16) float values[QB];
    for (int b = 0; b < B; ++b)
      vst1q_f32(values + b * 4, best[b]);
    float total = 0.0f;
    for (int32_t q = 0; q < query_vec_count; ++q)
      total += values[q];
    return total;
  }

  template <int D, int QB>
  static inline float
  maxsim_transposed_4x_neon(const float *__restrict qT, int32_t query_vec_count,
                            const float *__restrict train_block, int count) {
    constexpr int B = QB / 4;
    static_assert(B <= 6,
                  "four-vector NEON kernel supports at most 24 queries");
    float32x4_t best[B];
    init_neon_best(best);

    int tv = 0;
    for (; tv + 3 < count; tv += 4) {
      const float *__restrict x0 = train_block + (int64_t)(tv + 0) * D;
      const float *__restrict x1 = train_block + (int64_t)(tv + 1) * D;
      const float *__restrict x2 = train_block + (int64_t)(tv + 2) * D;
      const float *__restrict x3 = train_block + (int64_t)(tv + 3) * D;
      float32x4_t acc0[B];
      float32x4_t acc1[B];
      float32x4_t acc2[B];
      float32x4_t acc3[B];
      zero_neon_accumulators(acc0);
      zero_neon_accumulators(acc1);
      zero_neon_accumulators(acc2);
      zero_neon_accumulators(acc3);

      for (int d = 0; d < D; d += 4) {
#define LEMUR_NEON_FMA_TRANSPOSED_QUAD(DOFF)                                   \
  do {                                                                         \
    const float x0d = x0[d + DOFF];                                            \
    const float x1d = x1[d + DOFF];                                            \
    const float x2d = x2[d + DOFF];                                            \
    const float x3d = x3[d + DOFF];                                            \
    LEMUR_UNROLL_FULL for (int b = 0; b < B; ++b) {                            \
      const float32x4_t qv = vld1q_f32(qT + (int64_t)(d + DOFF) * QB + b * 4); \
      acc0[b] = vfmaq_n_f32(acc0[b], qv, x0d);                                 \
      acc1[b] = vfmaq_n_f32(acc1[b], qv, x1d);                                 \
      acc2[b] = vfmaq_n_f32(acc2[b], qv, x2d);                                 \
      acc3[b] = vfmaq_n_f32(acc3[b], qv, x3d);                                 \
    }                                                                          \
  } while (false)
        LEMUR_NEON_FMA_TRANSPOSED_QUAD(0);
        LEMUR_NEON_FMA_TRANSPOSED_QUAD(1);
        LEMUR_NEON_FMA_TRANSPOSED_QUAD(2);
        LEMUR_NEON_FMA_TRANSPOSED_QUAD(3);
#undef LEMUR_NEON_FMA_TRANSPOSED_QUAD
      }

      for (int b = 0; b < B; ++b) {
        const float32x4_t pair0 = vmaxq_f32(acc0[b], acc1[b]);
        const float32x4_t pair1 = vmaxq_f32(acc2[b], acc3[b]);
        best[b] = vmaxq_f32(best[b], vmaxq_f32(pair0, pair1));
      }
    }

    for (; tv < count; ++tv) {
      const float *__restrict x = train_block + (int64_t)tv * D;
      float32x4_t acc[B];
      zero_neon_accumulators(acc);
      for (int d = 0; d < D; ++d) {
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif
        for (int b = 0; b < B; ++b)
          acc[b] = vfmaq_n_f32(acc[b], vld1q_f32(qT + (int64_t)d * QB + b * 4),
                               x[d]);
      }
      for (int b = 0; b < B; ++b)
        best[b] = vmaxq_f32(best[b], acc[b]);
    }

    alignas(16) float values[QB];
    for (int b = 0; b < B; ++b)
      vst1q_f32(values + b * 4, best[b]);
    float total = 0.0f;
    for (int32_t q = 0; q < query_vec_count; ++q)
      total += values[q];
    return total;
  }

  template <int D, int QB, int DimUnroll>
  inline void score_candidates_transposed_2x_neon(
      const float *__restrict query, int32_t query_vec_count,
      const int *__restrict row, int num_indices,
      std::pair<float, int> *__restrict cand,
      float *__restrict qT_scratch) const {
    transpose_query_neon<D, QB>(query, query_vec_count, qT_scratch);
    score_candidates_loop<CandidatePrefetch::kNeon>(
        row, num_indices, cand, [&](int idx) {
          return maxsim_transposed_2x_neon<D, QB, DimUnroll>(
              qT_scratch, query_vec_count, train_ptrs_[idx],
              train_counts_[idx]);
        });
  }

  template <int D, int DimUnroll>
  inline void
  dispatch_transposed_2x_neon(const float *__restrict query,
                              int32_t query_vec_count,
                              const int *__restrict row, int num_indices,
                              std::pair<float, int> *__restrict cand,
                              float *__restrict qT_scratch) const {
    dispatch_query_bucket_48(query_vec_count, [&](auto bucket) {
      constexpr int QB = decltype(bucket)::value;
      score_candidates_transposed_2x_neon<D, QB, DimUnroll>(
          query, query_vec_count, row, num_indices, cand, qT_scratch);
    });
  }

  template <int D, int QB>
  inline void score_candidates_transposed_4x_neon(
      const float *__restrict query, int32_t query_vec_count,
      const int *__restrict row, int num_indices,
      std::pair<float, int> *__restrict cand,
      float *__restrict qT_scratch) const {
    transpose_query_neon<D, QB>(query, query_vec_count, qT_scratch);
    score_candidates_loop<CandidatePrefetch::kNeon>(
        row, num_indices, cand, [&](int idx) {
          return maxsim_transposed_4x_neon<D, QB>(qT_scratch, query_vec_count,
                                                  train_ptrs_[idx],
                                                  train_counts_[idx]);
        });
  }

  template <int D>
  inline void
  dispatch_transposed_4x_neon(const float *__restrict query,
                              int32_t query_vec_count,
                              const int *__restrict row, int num_indices,
                              std::pair<float, int> *__restrict cand,
                              float *__restrict qT_scratch) const {
    dispatch_query_bucket_48(query_vec_count, [&](auto bucket) {
      constexpr int QB = decltype(bucket)::value;
      if constexpr (QB <= 24) {
        score_candidates_transposed_4x_neon<D, QB>(
            query, query_vec_count, row, num_indices, cand, qT_scratch);
      }
    });
  }

  template <int D>
  inline void score_candidates_neon_dim(const float *__restrict query,
                                        int32_t query_vec_count,
                                        const int *__restrict row,
                                        int num_indices,
                                        std::pair<float, int> *__restrict cand,
                                        float *__restrict qT_scratch) const {
    if (query_vec_count <= 24) {
      dispatch_transposed_4x_neon<D>(query, query_vec_count, row, num_indices,
                                     cand, qT_scratch);
    } else if constexpr (D <= 64) {
      dispatch_transposed_2x_neon<D, 4>(query, query_vec_count, row,
                                        num_indices, cand, qT_scratch);
    } else {
      dispatch_transposed_2x_neon<D, 1>(query, query_vec_count, row,
                                        num_indices, cand, qT_scratch);
    }
  }

  inline bool score_candidates_neon(const float *__restrict query,
                                    int32_t query_vec_count,
                                    const int *__restrict row, int num_indices,
                                    std::pair<float, int> *__restrict cand,
                                    float *__restrict qT_scratch) const {
    if (!is_supported_simd_query_count(query_vec_count))
      return false;

    return dispatch_fast_dim([&](auto dim) {
      constexpr int D = decltype(dim)::value;
      score_candidates_neon_dim<D>(query, query_vec_count, row, num_indices,
                                   cand, qT_scratch);
    });
  }
#endif

  inline float maxsim_generic(const float *__restrict query,
                              int32_t query_vec_count, int train_index) const {
    const int32_t count = train_counts_[train_index];
    const float *__restrict train_block = train_ptrs_[train_index];

    float total = 0.0f;
    for (int32_t qv = 0; qv < query_vec_count; ++qv) {
      const float *__restrict q = query + (int64_t)qv * vec_dim_;
      float best = -std::numeric_limits<float>::infinity();

      const float *__restrict x = train_block;
      for (int32_t tv = 0; tv < count; ++tv, x += vec_dim_) {
        const float s = dot_product(q, x, (size_t)vec_dim_);
        best = (s > best) ? s : best;
      }
      total += best;
    }
    return total;
  }

  inline void
  score_candidates_generic(const float *__restrict query,
                           int32_t query_vec_count, const int *__restrict row,
                           int num_indices,
                           std::pair<float, int> *__restrict cand) const {
    for (int j = 0; j < num_indices; ++j) {
      const int idx = row[j];
      cand[(size_t)j] = {maxsim_generic(query, query_vec_count, idx), idx};
    }
  }

  inline float maxsim_one_query_one_train(const float *__restrict query,
                                          int32_t query_vec_count,
                                          float *__restrict qT_scratch,
                                          int train_index) const {
#if LEMUR_HAS_AVX512
    const int32_t count = train_counts_[train_index];
    const float *__restrict train_block = train_ptrs_[train_index];

    if (query_vec_count == 32) {
      float result = 0.0f;
      if (dispatch_fast_dim([&](auto dim) {
            constexpr int D = decltype(dim)::value;
            transpose_query_32<D>(query, qT_scratch);
            result = maxsim_Dx32_qT_2x<D>(qT_scratch, train_block, count);
          })) {
        return result;
      }
    }
#else
    (void)qT_scratch;
#endif

    return maxsim_generic(query, query_vec_count, train_index);
  }

public:
  MaxSim(const float *train, const int32_t *train_counts, int vec_dim,
         int num_train_points)
      : train_(train), train_counts_(train_counts), vec_dim_(vec_dim),
        num_train_points_(num_train_points),
        train_offsets_((size_t)num_train_points + 1, 0),
        train_ptrs_((size_t)num_train_points, nullptr) {
    int32_t off = 0;
    train_offsets_[0] = 0;
    for (int i = 0; i < num_train_points_; ++i) {
      train_ptrs_[i] = train_ + (int64_t)off * vec_dim_;
      off += train_counts_[i];
      train_offsets_[i + 1] = off;
    }
  }

  std::vector<std::vector<int>>
  batch_query_subset_fixed(const float *__restrict queries, int num_queries,
                           int32_t query_vec_count, int k,
                           const int *__restrict indices_matrix,
                           int num_indices, int num_threads) const {
    std::vector<std::vector<int>> results((size_t)num_queries);
    if (num_queries <= 0 || query_vec_count <= 0 || vec_dim_ <= 0 ||
        num_indices <= 0 || k <= 0) {
      return results;
    }

    const int kk = std::min(k, num_indices);
    for (int i = 0; i < num_queries; ++i)
      results[i].resize((size_t)kk);

    const int threads =
        (num_threads == -1) ? omp_get_max_threads() : std::max(1, num_threads);

#pragma omp parallel num_threads(threads)
    {
      std::vector<std::pair<float, int>> cand((size_t)num_indices);
#if LEMUR_HAS_AVX512
      alignas(64) float qT[kSimdScratchFloats];
#elif LEMUR_HAS_AVX2
      alignas(32) float qT[kSimdScratchFloats];
#elif LEMUR_HAS_NEON
      alignas(16) float qT[kSimdScratchFloats];
#endif

      auto cmp = [](const std::pair<float, int> &a,
                    const std::pair<float, int> &b) {
        if (a.first != b.first)
          return a.first > b.first;
        return a.second < b.second;
      };

#pragma omp for schedule(static)
      for (int qi = 0; qi < num_queries; ++qi) {
        const float *__restrict query = queries + (int64_t)qi *
                                                      (int64_t)query_vec_count *
                                                      (int64_t)vec_dim_;
        const int *__restrict row =
            indices_matrix + (int64_t)qi * (int64_t)num_indices;

#if LEMUR_HAS_AVX512
        if (score_candidates_avx512(query, query_vec_count, row, num_indices,
                                    cand.data(), qT)) {
        } else
#elif LEMUR_HAS_AVX2
        if (score_candidates_avx2(query, query_vec_count, row, num_indices,
                                  cand.data(), qT)) {
        } else
#elif LEMUR_HAS_NEON
        if (score_candidates_neon(query, query_vec_count, row, num_indices,
                                  cand.data(), qT)) {
        } else
#endif
        {
          score_candidates_generic(query, query_vec_count, row, num_indices,
                                   cand.data());
        }

        if (kk < num_indices) {
          std::nth_element(cand.begin(), cand.begin() + kk, cand.end(), cmp);
          std::sort(cand.begin(), cand.begin() + kk, cmp);
        } else {
          std::sort(cand.begin(), cand.end(), cmp);
        }

        auto &out = results[qi];
        for (int t = 0; t < kk; ++t)
          out[(size_t)t] = cand[(size_t)t].second;
      }
    }

    return results;
  }

  std::vector<std::vector<int>>
  batch_query_subset(const float *__restrict queries, int num_queries,
                     const int32_t *__restrict query_vec_counts, int k,
                     const int *__restrict indices_matrix, int num_indices,
                     int num_threads) const {
    std::vector<std::vector<int>> results((size_t)num_queries);
    if (num_queries <= 0 || vec_dim_ <= 0 || num_indices <= 0 || k <= 0) {
      return results;
    }

    const int kk = std::min(k, num_indices);
    for (int i = 0; i < num_queries; ++i)
      results[i].resize((size_t)kk);

    int requested =
        (num_threads == -1) ? omp_get_max_threads() : std::max(1, num_threads);
    int T = std::min(requested, num_queries);
    if (T <= 0)
      T = 1;

    std::vector<int> q_begin((size_t)T + 1);
    std::vector<int> q_end((size_t)T);
    {
      int base = num_queries / T;
      int rem = num_queries % T;
      int s = 0;
      for (int t = 0; t < T; ++t) {
        int take = base + (t < rem ? 1 : 0);
        q_begin[t] = s;
        q_end[t] = s + take;
        s += take;
      }
      q_begin[T] = num_queries;
    }

    std::vector<int64_t> vec_off((size_t)T + 1, 0);
    {
      int boundary_t = 1;
      int next_boundary = (T >= 1) ? q_begin[1] : num_queries;
      int64_t cum = 0;
      vec_off[0] = 0;

      for (int qi = 0; qi < num_queries; ++qi) {
        if (boundary_t <= T && qi == next_boundary) {
          vec_off[boundary_t] = cum;
          ++boundary_t;
          next_boundary = (boundary_t <= T) ? q_begin[boundary_t] : num_queries;
        }
        cum += (int64_t)query_vec_counts[qi];
      }
      vec_off[T] = cum;
    }

#pragma omp parallel num_threads(T)
    {
      const int tid = omp_get_thread_num();
      const int qb = q_begin[tid];
      const int qe = q_end[tid];

      const float *__restrict qptr = queries + vec_off[tid] * (int64_t)vec_dim_;

      std::vector<std::pair<float, int>> cand((size_t)num_indices);
#if LEMUR_HAS_AVX512
      alignas(64) float qT[kSimdScratchFloats];
#elif LEMUR_HAS_AVX2
      alignas(32) float qT[kSimdScratchFloats];
#elif LEMUR_HAS_NEON
      alignas(16) float qT[kSimdScratchFloats];
#endif

      auto cmp = [](const std::pair<float, int> &a,
                    const std::pair<float, int> &b) {
        if (a.first != b.first)
          return a.first > b.first;
        return a.second < b.second;
      };

      for (int qi = qb; qi < qe; ++qi) {
        const int32_t qcount = query_vec_counts[qi];
        const float *__restrict query = qptr;
        qptr += (int64_t)qcount * (int64_t)vec_dim_;

        const int *__restrict row =
            indices_matrix + (int64_t)qi * (int64_t)num_indices;

        if (qcount <= 0) {
          auto &out = results[qi];
          for (int t = 0; t < kk; ++t)
            out[(size_t)t] = row[t];
          continue;
        }

#if LEMUR_HAS_AVX512
        if (score_candidates_avx512(query, qcount, row, num_indices,
                                    cand.data(), qT)) {
        } else
#elif LEMUR_HAS_AVX2
        if (score_candidates_avx2(query, qcount, row, num_indices, cand.data(),
                                  qT)) {
        } else
#elif LEMUR_HAS_NEON
        if (score_candidates_neon(query, qcount, row, num_indices, cand.data(),
                                  qT)) {
        } else
#endif
        {
          score_candidates_generic(query, qcount, row, num_indices,
                                   cand.data());
        }

        if (kk < num_indices) {
          std::nth_element(cand.begin(), cand.begin() + kk, cand.end(), cmp);
          std::sort(cand.begin(), cand.begin() + kk, cmp);
        } else {
          std::sort(cand.begin(), cand.end(), cmp);
        }

        auto &out = results[qi];
        for (int t = 0; t < kk; ++t)
          out[(size_t)t] = cand[(size_t)t].second;
      }
    }

    return results;
  }
};
