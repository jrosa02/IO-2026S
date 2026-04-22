/*
 * evaluate_opt.c — weighted earliness/tardiness cost for the common due-date
 * scheduling problem, optimised for modern x86-64 (AVX2).
 *
 * Python signature:
 *   evaluate_opt(p, a, b, schedule, d) -> int
 *
 *   p, a, b   : contiguous int64 numpy arrays, length n
 *   schedule  : contiguous int64 numpy array  (permutation of 0..n-1)
 *   d         : int  (common due date)
 *
 * AVX2 uses emulated _mm256_mullo_epi64 (not in standard AVX2) via
 * two 32-bit unsigned multiplies, correct for scheduling costs.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>

#ifdef __AVX2__
#  include <immintrin.h>
#  pragma message "evaluate_opt.c: AVX2 path selected"
#else
#  pragma message "evaluate_opt.c: scalar fallback"
#endif

#ifdef __AVX2__
static __attribute__((always_inline)) inline __m256i
_mm256_mullo_epi64_emu(__m256i a, __m256i b)
{
    __m256i lo    = _mm256_mul_epu32(a, b);
    __m256i a_hi  = _mm256_srli_epi64(a, 32);
    __m256i b_hi  = _mm256_srli_epi64(b, 32);
    __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(a,    b_hi),
                                      _mm256_mul_epu32(a_hi, b));
    return _mm256_add_epi64(lo, _mm256_slli_epi64(cross, 32));
}
#endif

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
        __m256i vct   = _mm256_loadu_si256((const __m256i *)(ct       + k));
        __m256i vidx  = _mm256_loadu_si256((const __m256i *)(schedule + k));

        __m256i va = _mm256_i64gather_epi64((const long long *)a, vidx, 8);
        __m256i vb = _mm256_i64gather_epi64((const long long *)b, vidx, 8);

        __m256i vgap  = _mm256_sub_epi64(vct, vd);
        __m256i vneg  = _mm256_sub_epi64(vzero, vgap);

        __m256i mlate  = _mm256_cmpgt_epi64(vgap, vzero);
        __m256i mearly = _mm256_cmpgt_epi64(vneg,  vzero);

        __m256i late_c  = _mm256_and_si256(mlate,
                              _mm256_mullo_epi64_emu(vb, vgap));
        __m256i early_c = _mm256_and_si256(mearly,
                              _mm256_mullo_epi64_emu(va, vneg));

        vacc = _mm256_add_epi64(vacc, late_c);
        vacc = _mm256_add_epi64(vacc, early_c);
    }

    __m128i lo     = _mm256_castsi256_si128(vacc);
    __m128i hi     = _mm256_extracti128_si256(vacc, 1);
    __m128i sum128 = _mm_add_epi64(lo, hi);
    int64_t cost   = (int64_t)(_mm_extract_epi64(sum128, 0) +
                               _mm_extract_epi64(sum128, 1));

    for (; k < n; k++) {
        int64_t j   = schedule[k];
        int64_t gap = ct[k] - d;
        if (gap < 0)
            cost -= a[j] * gap;
        else if (gap > 0)
            cost += b[j] * gap;
    }
    return cost;
}
#else
static __attribute__((hot)) int64_t
cost_scalar(const int64_t *restrict a,
            const int64_t *restrict b,
            const int64_t *restrict schedule,
            const int64_t *restrict ct,
            npy_intp n,
            int64_t d)
{
    int64_t cost = 0;
    for (npy_intp k = 0; k < n; k++) {
        int64_t j   = schedule[k];
        int64_t gap = ct[k] - d;
        if (gap < 0)
            cost -= a[j] * gap;
        else if (gap > 0)
            cost += b[j] * gap;
    }
    return cost;
}
#endif

static PyObject *
evaluate_opt(PyObject *self, PyObject *args)
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

    npy_intp n = PyArray_SIZE(sched_arr);
    const int64_t *p      = (const int64_t *)PyArray_DATA(p_arr);
    const int64_t *a      = (const int64_t *)PyArray_DATA(a_arr);
    const int64_t *b      = (const int64_t *)PyArray_DATA(b_arr);
    const int64_t *sched  = (const int64_t *)PyArray_DATA(sched_arr);

    int64_t *ct = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (!ct) return PyErr_NoMemory();

    build_ct(p, sched, ct, n);

#ifdef __AVX2__
    int64_t result = cost_avx2(a, b, sched, ct, n, (int64_t)d);
#else
    int64_t result = cost_scalar(a, b, sched, ct, n, (int64_t)d);
#endif

    free(ct);
    return PyLong_FromLongLong((long long)result);
}
static PyMethodDef methods[] = {
    {"evaluate_opt", evaluate_opt, METH_VARARGS,
     "evaluate_opt(p, a, b, schedule, d) -> int\n"
     "AVX2-vectorised weighted earliness+tardiness cost."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "evaluate_opt", NULL, -1, methods
};

PyMODINIT_FUNC
PyInit_evaluate_opt(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
