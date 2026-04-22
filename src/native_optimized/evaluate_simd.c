/*
 * evaluate_simd.c — weighted E+T cost kernel optimised for modern ISAs.
 *
 * Compile with -DN=<n_jobs>. Selects AVX-512F+DQ, AVX2, or scalar at compile time.
 *
 * Exported symbol:
 *   int64_t evaluate(const int64_t *p, const int64_t *a, const int64_t *b,
 *                    const int64_t *schedule, int64_t d);
 */

#ifndef N
#  error "Compile with -DN=<number_of_jobs>  (e.g. -DN=50)"
#endif

#include <stdint.h>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#  include <immintrin.h>
#  define USE_AVX512 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define USE_AVX2 1
#endif

#if defined(USE_AVX512)
#  pragma message "evaluate_simd.c: AVX-512F+DQ path selected"
#elif defined(USE_AVX2)
#  pragma message "evaluate_simd.c: AVX2 path selected"
#else
#  pragma message "evaluate_simd.c: scalar fallback"
#endif

#ifdef USE_AVX2
static __attribute__((always_inline)) inline __m256i
_avx2_mullo_epi64(__m256i a, __m256i b)
{
    __m256i lo    = _mm256_mul_epu32(a, b);
    __m256i a_hi  = _mm256_srli_epi64(a, 32);
    __m256i b_hi  = _mm256_srli_epi64(b, 32);
    __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(a,    b_hi),
                                      _mm256_mul_epu32(a_hi, b));
    return _mm256_add_epi64(lo, _mm256_slli_epi64(cross, 32));
}
#endif

static __attribute__((always_inline)) inline void
build_ct(const int64_t *restrict p,
         const int64_t *restrict sched,
         int64_t ct[N])
{
    int64_t c = 0;
    for (int k = 0; k < N; k++) {
        c     += p[sched[k]];
        ct[k]  = c;
    }
}

#ifdef USE_AVX512
static __attribute__((hot, target("avx512f,avx512dq"))) int64_t
cost_avx512(const int64_t *restrict a,
            const int64_t *restrict b,
            const int64_t *restrict sched,
            const int64_t ct[N],
            int64_t d)
{
    const __m512i vd    = _mm512_set1_epi64(d);
    const __m512i vzero = _mm512_setzero_si512();
    __m512i vacc        = _mm512_setzero_si512();
    int     k           = 0;
    int64_t cost        = 0;

    for (; k + 8 <= N; k += 8) {
        __m512i vct  = _mm512_loadu_si512((const __m512i *)(ct    + k));
        __m512i vidx = _mm512_loadu_si512((const __m512i *)(sched + k));

        __m512i va = _mm512_i64gather_epi64(vidx, (const long long *)a, 8);
        __m512i vb = _mm512_i64gather_epi64(vidx, (const long long *)b, 8);

        __m512i vgap = _mm512_sub_epi64(vct, vd);         /* ct - d    */
        __m512i vneg = _mm512_sub_epi64(vzero, vgap);     /* d  - ct   */

        __mmask8 ml = _mm512_cmpgt_epi64_mask(vgap, vzero); /* gap > 0 */
        __mmask8 me = _mm512_cmpgt_epi64_mask(vneg,  vzero);/* -gap> 0 */

        vacc = _mm512_add_epi64(vacc, _mm512_maskz_mullo_epi64(ml, vb, vgap));
        vacc = _mm512_add_epi64(vacc, _mm512_maskz_mullo_epi64(me, va, vneg));
    }

    /* Horizontal reduce: fold 512→256→128→scalar. */
    __m256i lo256 = _mm512_castsi512_si256(vacc);
    __m256i hi256 = _mm512_extracti64x4_epi64(vacc, 1);
    __m256i sum256 = _mm256_add_epi64(lo256, hi256);
    __m128i lo128  = _mm256_castsi256_si128(sum256);
    __m128i hi128  = _mm256_extracti128_si256(sum256, 1);
    __m128i s128   = _mm_add_epi64(lo128, hi128);
    cost += (int64_t)(_mm_extract_epi64(s128, 0) + _mm_extract_epi64(s128, 1));

    /* Scalar tail (0–7 elements; statically 0 for N % 8 == 0). */
    for (; k < N; k++) {
        int64_t j   = sched[k];
        int64_t gap = ct[k] - d;
        if (gap < 0) cost -= a[j] * gap;
        else if (gap > 0) cost += b[j] * gap;
    }
    return cost;
}
#endif /* USE_AVX512 */

#ifdef USE_AVX2
static __attribute__((hot, target("avx2"))) int64_t
cost_avx2(const int64_t *restrict a,
          const int64_t *restrict b,
          const int64_t *restrict sched,
          const int64_t ct[N],
          int64_t d)
{
    const __m256i vd    = _mm256_set1_epi64x(d);
    const __m256i vzero = _mm256_setzero_si256();
    __m256i vacc        = _mm256_setzero_si256();
    int     k           = 0;
    int64_t cost        = 0;

    for (; k + 4 <= N; k += 4) {
        __m256i vct  = _mm256_loadu_si256((const __m256i *)(ct    + k));
        __m256i vidx = _mm256_loadu_si256((const __m256i *)(sched + k));

        __m256i va = _mm256_i64gather_epi64((const long long *)a, vidx, 8);
        __m256i vb = _mm256_i64gather_epi64((const long long *)b, vidx, 8);

        __m256i vgap = _mm256_sub_epi64(vct, vd);
        __m256i vneg = _mm256_sub_epi64(vzero, vgap);
        __m256i ml   = _mm256_cmpgt_epi64(vgap, vzero);
        __m256i me   = _mm256_cmpgt_epi64(vneg,  vzero);

        vacc = _mm256_add_epi64(vacc,
                   _mm256_and_si256(ml, _avx2_mullo_epi64(vb, vgap)));
        vacc = _mm256_add_epi64(vacc,
                   _mm256_and_si256(me, _avx2_mullo_epi64(va, vneg)));
    }

    __m128i lo = _mm256_castsi256_si128(vacc);
    __m128i hi = _mm256_extracti128_si256(vacc, 1);
    __m128i s  = _mm_add_epi64(lo, hi);
    cost += (int64_t)(_mm_extract_epi64(s, 0) + _mm_extract_epi64(s, 1));

    for (; k < N; k++) {
        int64_t j   = sched[k];
        int64_t gap = ct[k] - d;
        if (gap < 0) cost -= a[j] * gap;
        else if (gap > 0) cost += b[j] * gap;
    }
    return cost;
}
#endif /* USE_AVX2 */

static __attribute__((hot)) int64_t
cost_scalar(const int64_t *restrict a,
            const int64_t *restrict b,
            const int64_t *restrict sched,
            const int64_t ct[N],
            int64_t d)
{
    int64_t cost = 0;
    for (int k = 0; k < N; k++) {
        int64_t j   = sched[k];
        int64_t gap = ct[k] - d;
        if (__builtin_expect(gap < 0, 0)) cost -= a[j] * gap;
        else if (__builtin_expect(gap > 0, 1)) cost += b[j] * gap;
    }
    return cost;
}

int64_t
evaluate(const int64_t *restrict p,
         const int64_t *restrict a,
         const int64_t *restrict b,
         const int64_t *restrict sched,
         int64_t d)
{
    int64_t ct[N];
    build_ct(p, sched, ct);
#if   defined(USE_AVX512)
    return cost_avx512(a, b, sched, ct, d);
#elif defined(USE_AVX2)
    return cost_avx2  (a, b, sched, ct, d);
#else
    return cost_scalar(a, b, sched, ct, d);
#endif
}
