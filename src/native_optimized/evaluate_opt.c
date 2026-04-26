/*
 * evaluate_opt.c — weighted earliness/tardiness cost for the common due-date
 * scheduling problem.
 *
 * Python signatures:
 * evaluate_opt(p, a, b, schedule, d) -> int
 * evaluate_swap_opt(p, a, b, schedule, d, i, j) -> int
 * evaluate_batch_opt(p, a, b, schedules, d) -> np.ndarray shape (k,) int64
 * evaluate_batch_swap_opt(p, a, b, schedule, d, i_arr, j_arr) -> np.ndarray shape (k,) int64
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <stdlib.h>

/* ==========================================================================
 * LAYER 2 — CORE MATH (no PyObject*, pure C)
 * ========================================================================== */

static __attribute__((hot, noinline)) void
build_completion_times(const int64_t *restrict processing_times,
                       const int64_t *restrict schedule,
                       int64_t *restrict completion_times,
                       npy_intp n_jobs)
{
    int64_t cumulative_time = 0;
    for (npy_intp k = 0; k < n_jobs; k++) {
        cumulative_time     += processing_times[schedule[k]];
        completion_times[k]  = cumulative_time;
    }
}

static __attribute__((hot)) int64_t
cost_scalar(const int64_t *restrict earliness_weights,
            const int64_t *restrict tardiness_weights,
            const int64_t *restrict schedule,
            const int64_t *restrict completion_times,
            npy_intp n_jobs,
            int64_t due_date)
{
    int64_t cost = 0;
    for (npy_intp k = 0; k < n_jobs; k++) {
        int64_t job_idx  = schedule[k];
        int64_t time_gap = completion_times[k] - due_date;

        if (time_gap < 0)
            cost -= earliness_weights[job_idx] * time_gap;
        else if (time_gap > 0)
            cost += tardiness_weights[job_idx] * time_gap;
    }
    return cost;
}

typedef struct {
    const int64_t *p;
    const int64_t *a;
    const int64_t *b;
} ProbData;

static inline int64_t
_evaluate(const ProbData *pd, const int64_t *s, npy_intp n_jobs, int64_t due_date)
{
    int64_t *ct = (int64_t *)malloc((size_t)n_jobs * sizeof(int64_t));
    if (!ct) return -1;
    build_completion_times(pd->p, s, ct, n_jobs);
    int64_t result = cost_scalar(pd->a, pd->b, s, ct, n_jobs, due_date);
    free(ct);
    return result;
}

static void
_evaluate_batch(const ProbData *pd, const int64_t *s,
                npy_intp k, npy_intp n_jobs, int64_t due_date,
                int64_t *out)
{
    int64_t *ct = (int64_t *)malloc((size_t)n_jobs * sizeof(int64_t));
    if (!ct) return;
    for (npy_intp row = 0; row < k; row++) {
        const int64_t *sched = s + row * n_jobs;
        build_completion_times(pd->p, sched, ct, n_jobs);
        out[row] = cost_scalar(pd->a, pd->b, sched, ct, n_jobs, due_date);
    }
    free(ct);
}

/* ==========================================================================
 * LAYER 1 — PYTHON C-API WRAPPERS (unpack/pack only, no logic)
 * ========================================================================== */

static inline ProbData
extract_prob_data(PyArrayObject *p_arr, PyArrayObject *a_arr, PyArrayObject *b_arr)
{
    ProbData d;
    d.p = (const int64_t *)PyArray_DATA(p_arr);
    d.a = (const int64_t *)PyArray_DATA(a_arr);
    d.b = (const int64_t *)PyArray_DATA(b_arr);
    return d;
}

static PyObject *
evaluate_opt(PyObject *self, PyObject *args)
{
    PyArrayObject *p_arr, *a_arr, *b_arr, *sched_arr;
    long long due_date;

    if (!PyArg_ParseTuple(args, "O!O!O!O!L",
                          &PyArray_Type, &p_arr, &PyArray_Type, &a_arr,
                          &PyArray_Type, &b_arr, &PyArray_Type, &sched_arr, &due_date))
        return NULL;

    npy_intp n_jobs  = PyArray_SIZE(sched_arr);
    ProbData pd      = extract_prob_data(p_arr, a_arr, b_arr);
    const int64_t *s = (const int64_t *)PyArray_DATA(sched_arr);

    int64_t result = _evaluate(&pd, s, n_jobs, (int64_t)due_date);
    if (result == -1 && PyErr_NoMemory()) return NULL;
    return PyLong_FromLongLong((long long)result);
}

static PyObject *
evaluate_batch_opt(PyObject *self, PyObject *args)
{
    PyArrayObject *p_arr, *a_arr, *b_arr, *scheds_arr;
    long long due_date;

    if (!PyArg_ParseTuple(args, "O!O!O!O!L",
                          &PyArray_Type, &p_arr, &PyArray_Type, &a_arr,
                          &PyArray_Type, &b_arr, &PyArray_Type, &scheds_arr, &due_date))
        return NULL;

    npy_intp k       = PyArray_DIM(scheds_arr, 0);
    npy_intp n_jobs  = PyArray_DIM(scheds_arr, 1);
    ProbData pd      = extract_prob_data(p_arr, a_arr, b_arr);
    const int64_t *s = (const int64_t *)PyArray_DATA(scheds_arr);

    npy_intp dims[1] = {k};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
    if (!out) return NULL;

    _evaluate_batch(&pd, s, k, n_jobs, (int64_t)due_date,
                    (int64_t *)PyArray_DATA(out));
    return (PyObject *)out;
}

static PyObject *
evaluate_swap_opt(PyObject *self, PyObject *args)
{
    PyArrayObject *p_arr, *a_arr, *b_arr, *sched_arr;
    long long due_date, pos_i, pos_j;

    if (!PyArg_ParseTuple(args, "O!O!O!O!LLL",
                          &PyArray_Type, &p_arr, &PyArray_Type, &a_arr,
                          &PyArray_Type, &b_arr, &PyArray_Type, &sched_arr,
                          &due_date, &pos_i, &pos_j))
        return NULL;

    npy_intp n_jobs = PyArray_SIZE(sched_arr);
    ProbData pd     = extract_prob_data(p_arr, a_arr, b_arr);
    int64_t *s      = (int64_t *)PyArray_DATA(sched_arr);

    int64_t temp_job = s[pos_i]; s[pos_i] = s[pos_j]; s[pos_j] = temp_job;
    int64_t result   = _evaluate(&pd, s, n_jobs, (int64_t)due_date);
    s[pos_j] = s[pos_i]; s[pos_i] = temp_job;

    if (result == -1 && PyErr_NoMemory()) return NULL;
    return PyLong_FromLongLong((long long)result);
}

static PyObject *
evaluate_batch_swap_opt(PyObject *self, PyObject *args)
{
    PyArrayObject *p_arr, *a_arr, *b_arr, *sched_arr, *i_arr, *j_arr;
    long long due_date;

    if (!PyArg_ParseTuple(args, "O!O!O!O!LO!O!",
                          &PyArray_Type, &p_arr,
                          &PyArray_Type, &a_arr,
                          &PyArray_Type, &b_arr,
                          &PyArray_Type, &sched_arr,
                          &due_date,
                          &PyArray_Type, &i_arr,
                          &PyArray_Type, &j_arr))
        return NULL;

    npy_intp k             = PyArray_SIZE(i_arr);
    npy_intp n_jobs        = PyArray_SIZE(sched_arr);
    ProbData pd            = extract_prob_data(p_arr, a_arr, b_arr);
    int64_t *s             = (int64_t *)PyArray_DATA(sched_arr);
    const int64_t *pos_is  = (const int64_t *)PyArray_DATA(i_arr);
    const int64_t *pos_js  = (const int64_t *)PyArray_DATA(j_arr);

    npy_intp dims[1] = {k};
    PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
    if (!out) return NULL;
    int64_t *costs = (int64_t *)PyArray_DATA(out);

    for (npy_intp b_idx = 0; b_idx < k; b_idx++) {
        int64_t i_idx    = pos_is[b_idx];
        int64_t j_idx    = pos_js[b_idx];
        int64_t temp_job = s[i_idx]; s[i_idx] = s[j_idx]; s[j_idx] = temp_job;
        costs[b_idx]     = _evaluate(&pd, s, n_jobs, (int64_t)due_date);
        s[j_idx] = s[i_idx]; s[i_idx] = temp_job;
    }

    return (PyObject *)out;
}

/* ==========================================================================
 * MODULE INIT
 * ========================================================================== */

static PyMethodDef methods[] = {
    {"evaluate_opt",            evaluate_opt,            METH_VARARGS,
     "evaluate_opt(p, a, b, schedule, d) -> int"},
    {"evaluate_batch_opt",      evaluate_batch_opt,      METH_VARARGS,
     "evaluate_batch_opt(p, a, b, schedules, d) -> np.ndarray shape (k,) int64"},
    {"evaluate_swap_opt",       evaluate_swap_opt,       METH_VARARGS,
     "evaluate_swap_opt(p, a, b, schedule, d, i, j) -> int"},
    {"evaluate_batch_swap_opt", evaluate_batch_swap_opt, METH_VARARGS,
     "evaluate_batch_swap_opt(p, a, b, schedule, d, i_arr, j_arr) -> np.ndarray shape (k,) int64"},
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