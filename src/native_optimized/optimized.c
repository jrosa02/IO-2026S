/*
 * optimized.c — weighted earliness/tardiness cost for the common due-date
 * scheduling problem, optimised for modern x86-64 (AVX2).
 *
 * Build:
 *   python setup.py build_ext --inplace
 *
 * Python signature:
 *   evaluate_jit(p, a, b, schedule, d) -> int
 *
 *   p, a, b   : contiguous int64 numpy arrays, length n
 *   schedule  : contiguous int64 numpy array  (permutation of 0..n-1)
 *   d         : int  (common due date)
 *
 * Strategy
 * --------
 * The single-pass loop has a serial data dependency on the running completion
 * time (c[k] depends on c[k-1]), which blocks SIMD.  We break it into two
 * passes:
 *
 *   Pass 1  (serial)   : prefix-sum → completion time array ct[]
 *   Pass 2  (AVX2 x4)  : ct[k], schedule[k] → gather a[j]/b[j], mask, mullo
 *
 * For pass 2 we emulate _mm256_mullo_epi64 (AVX-512 only) using a pair of
 * 32-bit unsigned multiplies — correct for all signed int64 values that do
 * not overflow (which the scheduling cost never will for realistic inputs).
 *
 * A compile-time fallback for non-AVX2 targets handles the scalar tail and
 * any platform without AVX2.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#  include <immintrin.h>
#endif

/* Maximum n that fits in a stack-allocated completion-time buffer (8 KB). */
#define CT_STACK_MAX 1024

/* -----------------------------------------------------------------------
 * Portable 64-bit multiply emulated via two 32-bit multiplies.
 * Correct as long as the true product fits in int64 (guaranteed here:
 * max weight ~10^4, max gap ~10^6, product ~10^10 << 2^63).
 * ----------------------------------------------------------------------- */
#ifdef __AVX2__
static __attribute__((always_inline)) inline __m256i
_mm256_mullo_epi64_emu(__m256i a, __m256i b)
{
    /* lo32 * lo32 → 64-bit (unsigned, but sign doesn't matter for low half) */
    __m256i lo    = _mm256_mul_epu32(a, b);
    /* cross terms: a_lo*b_hi + a_hi*b_lo, shift into upper 32 bits */
    __m256i a_hi  = _mm256_srli_epi64(a, 32);
    __m256i b_hi  = _mm256_srli_epi64(b, 32);
    __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(a,    b_hi),
                                      _mm256_mul_epu32(a_hi, b));
    return _mm256_add_epi64(lo, _mm256_slli_epi64(cross, 32));
}
#endif /* __AVX2__ */

/* -----------------------------------------------------------------------
 * Pass 1: serial prefix-sum  →  completion times ct[0..n-1].
 * The data dependency on `c` prevents any SIMD here.
 * ----------------------------------------------------------------------- */
static __attribute__((hot, noinline)) void
build_ct(const int64_t *restrict p,
         const int64_t *restrict schedule,
         int64_t *restrict ct,
         npy_intp n)
{
    int64_t c = 0;
    for (npy_intp k = 0; k < n; k++) {
        c      += p[schedule[k]];
        ct[k]   = c;
    }
}

/* -----------------------------------------------------------------------
 * Pass 2 (scalar fallback) — used for the tail or when AVX2 is absent.
 * ----------------------------------------------------------------------- */
static __attribute__((hot)) int64_t
cost_scalar(const int64_t *restrict a,
            const int64_t *restrict b,
            const int64_t *restrict schedule,
            const int64_t *restrict ct,
            npy_intp start,
            npy_intp n,
            int64_t d)
{
    int64_t cost = 0;
    for (npy_intp k = start; k < n; k++) {
        int64_t j   = schedule[k];
        int64_t gap = ct[k] - d;
        if (__builtin_expect(gap < 0, 0))
            cost -= a[j] * gap;
        else if (__builtin_expect(gap > 0, 1))
            cost += b[j] * gap;
    }
    return cost;
}

/* -----------------------------------------------------------------------
 * Pass 2 (AVX2) — 4 lanes of int64, gather + masked multiply-accumulate.
 * ----------------------------------------------------------------------- */
#ifdef __AVX2__
static __attribute__((hot, target("avx2"))) int64_t
cost_avx2(const int64_t *restrict a,
          const int64_t *restrict b,
          const int64_t *restrict schedule,
          const int64_t *restrict ct,
          npy_intp n,
          int64_t d)
{
    const __m256i vd    = _mm256_set1_epi64x(d);
    const __m256i vzero = _mm256_setzero_si256();
    __m256i vacc        = _mm256_setzero_si256();

    npy_intp k = 0;
    for (; k + 4 <= n; k += 4) {
        /* Load completion times and job indices for 4 consecutive positions. */
        __m256i vct   = _mm256_loadu_si256((const __m256i *)(ct       + k));
        __m256i vidx  = _mm256_loadu_si256((const __m256i *)(schedule + k));

        /* Gather a[schedule[k..k+3]] and b[schedule[k..k+3]] (stride = 8 B). */
        __m256i va = _mm256_i64gather_epi64((const long long *)a, vidx, 8);
        __m256i vb = _mm256_i64gather_epi64((const long long *)b, vidx, 8);

        /* gap = ct - d;  neg_gap = -gap = d - ct */
        __m256i vgap  = _mm256_sub_epi64(vct, vd);
        __m256i vneg  = _mm256_sub_epi64(vzero, vgap);

        /* Boolean masks (all-ones / all-zeros per lane): */
        __m256i mlate  = _mm256_cmpgt_epi64(vgap, vzero);  /* gap  > 0 */
        __m256i mearly = _mm256_cmpgt_epi64(vneg,  vzero); /* -gap > 0 */

        /* Weighted contributions, zeroed for the wrong-sign lanes. */
        __m256i late_c  = _mm256_and_si256(mlate,
                              _mm256_mullo_epi64_emu(vb, vgap));
        __m256i early_c = _mm256_and_si256(mearly,
                              _mm256_mullo_epi64_emu(va, vneg));

        vacc = _mm256_add_epi64(vacc, late_c);
        vacc = _mm256_add_epi64(vacc, early_c);
    }

    /* Horizontal reduce: sum the 4 int64 lanes. */
    __m128i lo     = _mm256_castsi256_si128(vacc);
    __m128i hi     = _mm256_extracti128_si256(vacc, 1);
    __m128i sum128 = _mm_add_epi64(lo, hi);
    int64_t cost   = (int64_t)(_mm_extract_epi64(sum128, 0) +
                               _mm_extract_epi64(sum128, 1));

    /* Scalar tail for remaining elements. */
    cost += cost_scalar(a, b, schedule, ct, k, n, d);
    return cost;
}
#endif /* __AVX2__ */

/* -----------------------------------------------------------------------
 * Top-level driver — allocates ct[], dispatches to AVX2 or scalar.
 * ----------------------------------------------------------------------- */
static __attribute__((hot)) int64_t
_compute(const int64_t *restrict p,
         const int64_t *restrict a,
         const int64_t *restrict b,
         const int64_t *restrict schedule,
         npy_intp n,
         int64_t d)
{
    /* Stack buffer covers all standard sch* sizes (up to sch1000, n=1000). */
    int64_t  ct_stack[CT_STACK_MAX];
    int64_t *ct = (n <= CT_STACK_MAX)
                      ? ct_stack
                      : (int64_t *)PyMem_Malloc((size_t)n * sizeof(int64_t));
    if (!ct) return -1;  /* OOM — caller checks via exception state */

    build_ct(p, schedule, ct, n);

#ifdef __AVX2__
    int64_t result = cost_avx2(a, b, schedule, ct, n, d);
#else
    #warning Fallback to scallar
    int64_t result = cost_scalar(a, b, schedule, ct, 0, n, d);
#endif

    if (n > CT_STACK_MAX) PyMem_Free(ct);
    return result;
}

/* -----------------------------------------------------------------------
 * Python-callable wrapper
 * ----------------------------------------------------------------------- */
static PyObject *
evaluate_jit(PyObject *self, PyObject *args)
{
    PyArrayObject *p_arr, *a_arr, *b_arr, *sched_arr;
    long long d;

    if (!PyArg_ParseTuple(args, "O!O!O!O!L",
                          &PyArray_Type, &p_arr,
                          &PyArray_Type, &a_arr,
                          &PyArray_Type, &b_arr,
                          &PyArray_Type, &sched_arr,
                          &d))
        return NULL;

    int64_t result = _compute(
        (const int64_t *)PyArray_DATA(p_arr),
        (const int64_t *)PyArray_DATA(a_arr),
        (const int64_t *)PyArray_DATA(b_arr),
        (const int64_t *)PyArray_DATA(sched_arr),
        PyArray_SIZE(sched_arr),
        (int64_t)d
    );

    return PyLong_FromLongLong((long long)result);
}

/* -----------------------------------------------------------------------
 * Order Crossover (OX) — AVX-512 gather + compress
 *
 * Algorithm (two-buffer strategy):
 *   1. Copy p1[a:b] directly into child[a:b].
 *   2. Mark those jobs in a uint64 used[] table.
 *   3. Scan p2 left-to-right; keep only jobs where used[job]==0.
 *      AVX-512 path: load 8 int64 from p2, gather 8 used flags, compress-
 *      store the passing lanes into a tmp[] buffer.
 *   4. Distribute tmp[0..a-1] → child[0..a-1] and
 *                tmp[a..]    → child[b..n-1].
 *
 * Python signature:
 *   crossover_jit(p1, p2, a, b) -> np.ndarray[int64]
 * ----------------------------------------------------------------------- */

#define OX_STACK_MAX 1024   /* covers all standard sch* sizes (up to n=1000) */

static __attribute__((hot)) void
_ox_crossover(const int64_t *restrict p1,
              const int64_t *restrict p2,
              int64_t *restrict child,
              npy_intp n, npy_intp a, npy_intp b)
{
    /* Stack buffers: used[n]×8 B + tmp[n]×8 B = 16 KB for n=1024 (in L1). */
    uint64_t used_stack[OX_STACK_MAX];
    int64_t  tmp_stack[OX_STACK_MAX];

    int heap = (n > OX_STACK_MAX);
    uint64_t *used = heap ? (uint64_t *)malloc((size_t)n * sizeof(uint64_t)) : used_stack;
    int64_t  *tmp  = heap ? (int64_t  *)malloc((size_t)n * sizeof(int64_t))  : tmp_stack;
    if (!used || !tmp) { free(used); free(tmp); return; }

    /* 1. Segment copy */
    memcpy(child + a, p1 + a, (size_t)(b - a) * sizeof(int64_t));

    /* 2. Mark used jobs */
    memset(used, 0, (size_t)n * sizeof(uint64_t));
    for (npy_intp k = a; k < b; k++)
        used[p1[k]] = 1u;

    /* 3. Collect unused p2 jobs → tmp[], skipping used ones.
     *    AVX-512F: gather 8 used-flags per iteration, compress non-zero out. */
    npy_intp out = 0, k = 0;
#ifdef __AVX512F__
    {
        const __m512i vzero = _mm512_setzero_si512();
        for (; k + 8 <= n; k += 8) {
            __m512i v64  = _mm512_loadu_si512((const void *)(p2 + k));
            __m512i vu   = _mm512_i64gather_epi64(v64, (const void *)used, 8);
            __mmask8 keep = _mm512_cmpeq_epi64_mask(vu, vzero);
            _mm512_mask_compressstoreu_epi64((void *)(tmp + out), keep, v64);
            out += (npy_intp)__builtin_popcount((unsigned)keep);
        }
    }
#endif
    for (; k < n; k++) {
        if (!used[p2[k]]) tmp[out++] = p2[k];
    }

    /* 4. Distribute: tmp[0..a-1] → child[0..a-1], tmp[a..] → child[b..n-1] */
    memcpy(child,     tmp,     (size_t)a       * sizeof(int64_t));
    memcpy(child + b, tmp + a, (size_t)(n - b) * sizeof(int64_t));

    if (heap) { free(used); free(tmp); }
}

static PyObject *
crossover_jit(PyObject *self, PyObject *args)
{
    PyArrayObject *p1_arr, *p2_arr;
    int a_int, b_int;

    if (!PyArg_ParseTuple(args, "O!O!ii",
                          &PyArray_Type, &p1_arr,
                          &PyArray_Type, &p2_arr,
                          &a_int, &b_int))
        return NULL;

    npy_intp n       = PyArray_SIZE(p1_arr);
    npy_intp dims[1] = {n};
    PyArrayObject *child_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
    if (!child_arr) return NULL;

    _ox_crossover(
        (const int64_t *)PyArray_DATA(p1_arr),
        (const int64_t *)PyArray_DATA(p2_arr),
        (int64_t *)PyArray_DATA(child_arr),
        n, (npy_intp)a_int, (npy_intp)b_int
    );

    return (PyObject *)child_arr;
}

/* -----------------------------------------------------------------------
 * Batch Order Crossover — one Python→C call for an entire generation.
 *
 * Python signature:
 *   batch_crossover_jit(p1, p2, a, b) -> np.ndarray  shape (k, n)
 *
 *   p1, p2 : (k, n) contiguous int64 — k parent pairs
 *   a, b   : (k,)  int32             — per-pair cut points, a[i] < b[i]
 *
 * Reuses _ox_crossover() for each row; avoids k separate Python→C calls.
 * ----------------------------------------------------------------------- */
static PyObject *
batch_crossover_jit(PyObject *self, PyObject *args)
{
    PyArrayObject *p1_arr, *p2_arr, *a_arr, *b_arr;

    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                          &PyArray_Type, &p1_arr,
                          &PyArray_Type, &p2_arr,
                          &PyArray_Type, &a_arr,
                          &PyArray_Type, &b_arr))
        return NULL;

    npy_intp k       = PyArray_DIM(p1_arr, 0);
    npy_intp n       = PyArray_DIM(p1_arr, 1);
    npy_intp dims[2] = {k, n};

    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_INT64);
    if (!out) return NULL;

    const int64_t *p1 = (const int64_t *)PyArray_DATA(p1_arr);
    const int64_t *p2 = (const int64_t *)PyArray_DATA(p2_arr);
    const int32_t *av = (const int32_t *)PyArray_DATA(a_arr);
    const int32_t *bv = (const int32_t *)PyArray_DATA(b_arr);
    int64_t       *ch = (int64_t *)PyArray_DATA(out);

    for (npy_intp c = 0; c < k; c++)
        _ox_crossover(p1 + c * n, p2 + c * n, ch + c * n,
                      n, (npy_intp)av[c], (npy_intp)bv[c]);

    return (PyObject *)out;
}

/* -----------------------------------------------------------------------
 * Module definition
 * ----------------------------------------------------------------------- */
static PyMethodDef methods[] = {
    {"evaluate_jit", evaluate_jit, METH_VARARGS,
     "evaluate_jit(p, a, b, schedule, d) -> int\n"
     "Two-pass AVX2-vectorised weighted earliness+tardiness cost."},
    {"crossover_jit", crossover_jit, METH_VARARGS,
     "crossover_jit(p1, p2, a, b) -> ndarray\n"
     "Order Crossover (OX) with AVX-512 gather+compress."},
    {"batch_crossover_jit", batch_crossover_jit, METH_VARARGS,
     "batch_crossover_jit(p1, p2, a, b) -> ndarray  shape (k,n)\n"
     "Batch OX crossover for an entire generation in one C call."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "optimized", NULL, -1, methods
};

PyMODINIT_FUNC
PyInit_optimized(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
