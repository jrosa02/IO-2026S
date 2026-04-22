/*
 * crossover_opt.c — Order Crossover (OX) for job scheduling.
 *
 * Python signature:
 *   batch_crossover_opt(p1, p2, a, b) -> np.ndarray  shape (k, n)
 *
 * Strategy: copy p1[a:b] to child[a:b], mark used jobs, scan p2 for
 * unused jobs and distribute to child[0..a-1] and child[b..n-1].
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__)
#  include <immintrin.h>
#endif

#if defined(__AVX512F__)
#  pragma message "crossover_opt.c: AVX-512 path selected"
#else
#  pragma message "crossover_opt.c: scalar fallback"
#endif

#define OX_STACK_MAX 1024

static __attribute__((hot)) void
_ox_crossover(const int64_t *restrict p1,
              const int64_t *restrict p2,
              int64_t *restrict child,
              npy_intp n, npy_intp a, npy_intp b)
{
    uint64_t used_stack[OX_STACK_MAX];
    int64_t  tmp_stack[OX_STACK_MAX];

    int heap = (n > OX_STACK_MAX);
    uint64_t *used = heap ? (uint64_t *)malloc((size_t)n * sizeof(uint64_t)) : used_stack;
    int64_t  *tmp  = heap ? (int64_t  *)malloc((size_t)n * sizeof(int64_t))  : tmp_stack;
    if (!used || !tmp) { free(used); free(tmp); return; }

    memcpy(child + a, p1 + a, (size_t)(b - a) * sizeof(int64_t));

    memset(used, 0, (size_t)n * sizeof(uint64_t));
    for (npy_intp k = a; k < b; k++)
        used[p1[k]] = 1u;

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

    memcpy(child,     tmp,     (size_t)a       * sizeof(int64_t));
    memcpy(child + b, tmp + a, (size_t)(n - b) * sizeof(int64_t));

    if (heap) { free(used); free(tmp); }
}

static PyObject *
crossover_opt(PyObject *self, PyObject *args)
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

static PyObject *
batch_crossover_opt(PyObject *self, PyObject *args)
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

static PyMethodDef methods[] = {
    {"crossover_opt", crossover_opt, METH_VARARGS,
     "crossover_opt(p1, p2, a, b) -> ndarray\n"
     "Order Crossover (OX)."},
    {"batch_crossover_opt", batch_crossover_opt, METH_VARARGS,
     "batch_crossover_opt(p1, p2, a, b) -> ndarray  shape (k,n)\n"
     "Batch OX crossover for an entire generation in one C call."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "crossover_opt", NULL, -1, methods
};

PyMODINIT_FUNC
PyInit_crossover_opt(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
