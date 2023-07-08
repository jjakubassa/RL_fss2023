#include "init.c"
#include "getargs.c"
#include "getargsfast.c"
#include "int_ops.c"
#include "str_ops.c"
#include "bytes_ops.c"
#include "list_ops.c"
#include "dict_ops.c"
#include "set_ops.c"
#include "tuple_ops.c"
#include "exc_ops.c"
#include "misc_ops.c"
#include "generic_ops.c"
#include "__native.h"
#include "__native_internal.h"
static PyMethodDef module_methods[] = {
    {"fib", (PyCFunction)CPyPy_fib, METH_FASTCALL | METH_KEYWORDS, NULL /* docstring */},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "fib",
    NULL, /* docstring */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    module_methods
};

PyMODINIT_FUNC PyInit_fib(void)
{
    PyObject* modname = NULL;
    if (CPyModule_fib_internal) {
        Py_INCREF(CPyModule_fib_internal);
        return CPyModule_fib_internal;
    }
    CPyModule_fib_internal = PyModule_Create(&module);
    if (unlikely(CPyModule_fib_internal == NULL))
        goto fail;
    modname = PyObject_GetAttrString((PyObject *)CPyModule_fib_internal, "__name__");
    CPyStatic_globals = PyModule_GetDict(CPyModule_fib_internal);
    if (unlikely(CPyStatic_globals == NULL))
        goto fail;
    if (CPyGlobalsInit() < 0)
        goto fail;
    char result = CPyDef___top_level__();
    if (result == 2)
        goto fail;
    Py_DECREF(modname);
    return CPyModule_fib_internal;
    fail:
    Py_CLEAR(CPyModule_fib_internal);
    Py_CLEAR(modname);
    return NULL;
}

CPyTagged CPyDef_fib(CPyTagged cpy_r_n) {
    int64_t cpy_r_r0;
    char cpy_r_r1;
    char cpy_r_r2;
    char cpy_r_r3;
    CPyTagged cpy_r_r4;
    CPyTagged cpy_r_r5;
    CPyTagged cpy_r_r6;
    CPyTagged cpy_r_r7;
    CPyTagged cpy_r_r8;
    CPyTagged cpy_r_r9;
CPyL0: ;
    cpy_r_r0 = cpy_r_n & 1;
    cpy_r_r1 = cpy_r_r0 != 0;
    if (!cpy_r_r1) goto CPyL2;
CPyL1: ;
    cpy_r_r2 = CPyTagged_IsLt_(2, cpy_r_n);
    if (cpy_r_r2) {
        goto CPyL4;
    } else
        goto CPyL3;
CPyL2: ;
    cpy_r_r3 = (Py_ssize_t)cpy_r_n <= (Py_ssize_t)2;
    if (!cpy_r_r3) goto CPyL4;
CPyL3: ;
    CPyTagged_INCREF(cpy_r_n);
    return cpy_r_n;
CPyL4: ;
    cpy_r_r4 = CPyTagged_Subtract(cpy_r_n, 4);
    cpy_r_r5 = CPyDef_fib(cpy_r_r4);
    CPyTagged_DECREF(cpy_r_r4);
    if (unlikely(cpy_r_r5 == CPY_INT_TAG)) {
        CPy_AddTraceback("fib.py", "fib", 8, CPyStatic_globals);
        goto CPyL7;
    }
CPyL5: ;
    cpy_r_r6 = CPyTagged_Subtract(cpy_r_n, 2);
    cpy_r_r7 = CPyDef_fib(cpy_r_r6);
    CPyTagged_DECREF(cpy_r_r6);
    if (unlikely(cpy_r_r7 == CPY_INT_TAG)) {
        CPy_AddTraceback("fib.py", "fib", 8, CPyStatic_globals);
        goto CPyL8;
    }
CPyL6: ;
    cpy_r_r8 = CPyTagged_Add(cpy_r_r5, cpy_r_r7);
    CPyTagged_DECREF(cpy_r_r5);
    CPyTagged_DECREF(cpy_r_r7);
    return cpy_r_r8;
CPyL7: ;
    cpy_r_r9 = CPY_INT_TAG;
    return cpy_r_r9;
CPyL8: ;
    CPyTagged_DecRef(cpy_r_r5);
    goto CPyL7;
}

PyObject *CPyPy_fib(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames) {
    static const char * const kwlist[] = {"n", 0};
    static CPyArg_Parser parser = {"O:fib", kwlist, 0};
    PyObject *obj_n;
    if (!CPyArg_ParseStackAndKeywordsOneArg(args, nargs, kwnames, &parser, &obj_n)) {
        return NULL;
    }
    CPyTagged arg_n;
    if (likely(PyLong_Check(obj_n)))
        arg_n = CPyTagged_BorrowFromObject(obj_n);
    else {
        CPy_TypeError("int", obj_n); goto fail;
    }
    CPyTagged retval = CPyDef_fib(arg_n);
    if (retval == CPY_INT_TAG) {
        return NULL;
    }
    PyObject *retbox = CPyTagged_StealAsObject(retval);
    return retbox;
fail: ;
    CPy_AddTraceback("fib.py", "fib", 4, CPyStatic_globals);
    return NULL;
}

char CPyDef___top_level__(void) {
    PyObject *cpy_r_r0;
    PyObject *cpy_r_r1;
    char cpy_r_r2;
    PyObject *cpy_r_r3;
    PyObject *cpy_r_r4;
    PyObject *cpy_r_r5;
    PyObject *cpy_r_r6;
    PyObject *cpy_r_r7;
    char cpy_r_r8;
    PyObject *cpy_r_r9;
    PyObject *cpy_r_r10;
    PyObject *cpy_r_r11;
    PyObject *cpy_r_r12;
    PyObject *cpy_r_r13;
    PyObject *cpy_r_r14;
    int32_t cpy_r_r15;
    char cpy_r_r16;
    PyObject *cpy_r_r17;
    PyObject *cpy_r_r18;
    PyObject *cpy_r_r19;
    PyObject *cpy_r_r20;
    PyObject *cpy_r_r21;
    PyObject *cpy_r_r22;
    PyObject *cpy_r_r23;
    int32_t cpy_r_r24;
    char cpy_r_r25;
    CPyTagged cpy_r_r26;
    PyObject *cpy_r_r27;
    PyObject *cpy_r_r28;
    PyObject *cpy_r_r29;
    PyObject *cpy_r_r30;
    PyObject *cpy_r_r31;
    PyObject *cpy_r_r32;
    PyObject *cpy_r_r33;
    PyObject *cpy_r_r34;
    PyObject *cpy_r_r35;
    PyObject *cpy_r_r36;
    PyObject *cpy_r_r37;
    PyObject *cpy_r_r38;
    PyObject *cpy_r_r39;
    PyObject *cpy_r_r40;
    PyObject **cpy_r_r42;
    PyObject *cpy_r_r43;
    char cpy_r_r44;
CPyL0: ;
    cpy_r_r0 = CPyModule_builtins;
    cpy_r_r1 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r2 = cpy_r_r0 != cpy_r_r1;
    if (cpy_r_r2) goto CPyL3;
CPyL1: ;
    cpy_r_r3 = CPyStatics[3]; /* 'builtins' */
    cpy_r_r4 = PyImport_Import(cpy_r_r3);
    if (unlikely(cpy_r_r4 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", -1, CPyStatic_globals);
        goto CPyL23;
    }
CPyL2: ;
    CPyModule_builtins = cpy_r_r4;
    CPy_INCREF(CPyModule_builtins);
    CPy_DECREF(cpy_r_r4);
CPyL3: ;
    cpy_r_r5 = CPyStatic_globals;
    cpy_r_r6 = CPyModule_time;
    cpy_r_r7 = (PyObject *)&_Py_NoneStruct;
    cpy_r_r8 = cpy_r_r6 != cpy_r_r7;
    if (cpy_r_r8) goto CPyL6;
CPyL4: ;
    cpy_r_r9 = CPyStatics[4]; /* 'time' */
    cpy_r_r10 = PyImport_Import(cpy_r_r9);
    if (unlikely(cpy_r_r10 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", 1, CPyStatic_globals);
        goto CPyL23;
    }
CPyL5: ;
    CPyModule_time = cpy_r_r10;
    CPy_INCREF(CPyModule_time);
    CPy_DECREF(cpy_r_r10);
CPyL6: ;
    cpy_r_r11 = PyImport_GetModuleDict();
    cpy_r_r12 = CPyStatics[4]; /* 'time' */
    cpy_r_r13 = CPyDict_GetItem(cpy_r_r11, cpy_r_r12);
    if (unlikely(cpy_r_r13 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", 1, CPyStatic_globals);
        goto CPyL23;
    }
CPyL7: ;
    cpy_r_r14 = CPyStatics[4]; /* 'time' */
    cpy_r_r15 = CPyDict_SetItem(cpy_r_r5, cpy_r_r14, cpy_r_r13);
    CPy_DECREF(cpy_r_r13);
    cpy_r_r16 = cpy_r_r15 >= 0;
    if (unlikely(!cpy_r_r16)) {
        CPy_AddTraceback("fib.py", "<module>", 1, CPyStatic_globals);
        goto CPyL23;
    }
CPyL8: ;
    cpy_r_r17 = CPyModule_time;
    cpy_r_r18 = CPyStatics[4]; /* 'time' */
    cpy_r_r19 = CPyObject_GetAttr(cpy_r_r17, cpy_r_r18);
    if (unlikely(cpy_r_r19 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", 11, CPyStatic_globals);
        goto CPyL23;
    }
CPyL9: ;
    cpy_r_r20 = _PyObject_Vectorcall(cpy_r_r19, 0, 0, 0);
    CPy_DECREF(cpy_r_r19);
    if (unlikely(cpy_r_r20 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", 11, CPyStatic_globals);
        goto CPyL23;
    }
CPyL10: ;
    if (likely(CPyFloat_Check(cpy_r_r20)))
        cpy_r_r21 = cpy_r_r20;
    else {
        CPy_TypeErrorTraceback("fib.py", "<module>", 11, CPyStatic_globals, "float", cpy_r_r20);
        goto CPyL23;
    }
CPyL11: ;
    cpy_r_r22 = CPyStatic_globals;
    cpy_r_r23 = CPyStatics[5]; /* 't0' */
    cpy_r_r24 = CPyDict_SetItem(cpy_r_r22, cpy_r_r23, cpy_r_r21);
    CPy_DECREF(cpy_r_r21);
    cpy_r_r25 = cpy_r_r24 >= 0;
    if (unlikely(!cpy_r_r25)) {
        CPy_AddTraceback("fib.py", "<module>", 11, CPyStatic_globals);
        goto CPyL23;
    }
CPyL12: ;
    cpy_r_r26 = CPyDef_fib(64);
    if (unlikely(cpy_r_r26 == CPY_INT_TAG)) {
        CPy_AddTraceback("fib.py", "<module>", 12, CPyStatic_globals);
        goto CPyL23;
    } else
        goto CPyL24;
CPyL13: ;
    cpy_r_r27 = CPyModule_time;
    cpy_r_r28 = CPyStatics[4]; /* 'time' */
    cpy_r_r29 = CPyObject_GetAttr(cpy_r_r27, cpy_r_r28);
    if (unlikely(cpy_r_r29 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", 13, CPyStatic_globals);
        goto CPyL23;
    }
CPyL14: ;
    cpy_r_r30 = _PyObject_Vectorcall(cpy_r_r29, 0, 0, 0);
    CPy_DECREF(cpy_r_r29);
    if (unlikely(cpy_r_r30 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", 13, CPyStatic_globals);
        goto CPyL23;
    }
CPyL15: ;
    if (likely(CPyFloat_Check(cpy_r_r30)))
        cpy_r_r31 = cpy_r_r30;
    else {
        CPy_TypeErrorTraceback("fib.py", "<module>", 13, CPyStatic_globals, "float", cpy_r_r30);
        goto CPyL23;
    }
CPyL16: ;
    cpy_r_r32 = CPyStatic_globals;
    cpy_r_r33 = CPyStatics[5]; /* 't0' */
    cpy_r_r34 = CPyDict_GetItem(cpy_r_r32, cpy_r_r33);
    if (unlikely(cpy_r_r34 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", 13, CPyStatic_globals);
        goto CPyL25;
    }
CPyL17: ;
    if (likely(CPyFloat_Check(cpy_r_r34)))
        cpy_r_r35 = cpy_r_r34;
    else {
        CPy_TypeErrorTraceback("fib.py", "<module>", 13, CPyStatic_globals, "float", cpy_r_r34);
        goto CPyL25;
    }
CPyL18: ;
    cpy_r_r36 = PyNumber_Subtract(cpy_r_r31, cpy_r_r35);
    CPy_DECREF(cpy_r_r31);
    CPy_DECREF(cpy_r_r35);
    if (unlikely(cpy_r_r36 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", 13, CPyStatic_globals);
        goto CPyL23;
    }
CPyL19: ;
    if (likely(CPyFloat_Check(cpy_r_r36)))
        cpy_r_r37 = cpy_r_r36;
    else {
        CPy_TypeErrorTraceback("fib.py", "<module>", 13, CPyStatic_globals, "float", cpy_r_r36);
        goto CPyL23;
    }
CPyL20: ;
    cpy_r_r38 = CPyModule_builtins;
    cpy_r_r39 = CPyStatics[6]; /* 'print' */
    cpy_r_r40 = CPyObject_GetAttr(cpy_r_r38, cpy_r_r39);
    if (unlikely(cpy_r_r40 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", 13, CPyStatic_globals);
        goto CPyL26;
    }
CPyL21: ;
    PyObject *cpy_r_r41[1] = {cpy_r_r37};
    cpy_r_r42 = (PyObject **)&cpy_r_r41;
    cpy_r_r43 = _PyObject_Vectorcall(cpy_r_r40, cpy_r_r42, 1, 0);
    CPy_DECREF(cpy_r_r40);
    if (unlikely(cpy_r_r43 == NULL)) {
        CPy_AddTraceback("fib.py", "<module>", 13, CPyStatic_globals);
        goto CPyL26;
    } else
        goto CPyL27;
CPyL22: ;
    CPy_DECREF(cpy_r_r37);
    return 1;
CPyL23: ;
    cpy_r_r44 = 2;
    return cpy_r_r44;
CPyL24: ;
    CPyTagged_DECREF(cpy_r_r26);
    goto CPyL13;
CPyL25: ;
    CPy_DecRef(cpy_r_r31);
    goto CPyL23;
CPyL26: ;
    CPy_DecRef(cpy_r_r37);
    goto CPyL23;
CPyL27: ;
    CPy_DECREF(cpy_r_r43);
    goto CPyL22;
}

int CPyGlobalsInit(void)
{
    static int is_initialized = 0;
    if (is_initialized) return 0;
    
    CPy_Init();
    CPyModule_fib = Py_None;
    CPyModule_builtins = Py_None;
    CPyModule_time = Py_None;
    if (CPyStatics_Initialize(CPyStatics, CPyLit_Str, CPyLit_Bytes, CPyLit_Int, CPyLit_Float, CPyLit_Complex, CPyLit_Tuple, CPyLit_FrozenSet) < 0) {
        return -1;
    }
    is_initialized = 1;
    return 0;
}

PyObject *CPyStatics[7];
const char * const CPyLit_Str[] = {
    "\004\bbuiltins\004time\002t0\005print",
    "",
};
const char * const CPyLit_Bytes[] = {
    "",
};
const char * const CPyLit_Int[] = {
    "",
};
const double CPyLit_Float[] = {0};
const double CPyLit_Complex[] = {0};
const int CPyLit_Tuple[] = {0};
const int CPyLit_FrozenSet[] = {0};
CPyModule *CPyModule_fib_internal = NULL;
CPyModule *CPyModule_fib;
PyObject *CPyStatic_globals;
CPyModule *CPyModule_builtins_internal = NULL;
CPyModule *CPyModule_builtins;
CPyModule *CPyModule_time_internal = NULL;
CPyModule *CPyModule_time;
CPyTagged CPyDef_fib(CPyTagged cpy_r_n);
PyObject *CPyPy_fib(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
char CPyDef___top_level__(void);
