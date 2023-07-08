#ifndef MYPYC_NATIVE_INTERNAL_H
#define MYPYC_NATIVE_INTERNAL_H
#include <Python.h>
#include <CPy.h>
#include "__native.h"

int CPyGlobalsInit(void);

extern PyObject *CPyStatics[7];
extern const char * const CPyLit_Str[];
extern const char * const CPyLit_Bytes[];
extern const char * const CPyLit_Int[];
extern const double CPyLit_Float[];
extern const double CPyLit_Complex[];
extern const int CPyLit_Tuple[];
extern const int CPyLit_FrozenSet[];
extern CPyModule *CPyModule_fib_internal;
extern CPyModule *CPyModule_fib;
extern PyObject *CPyStatic_globals;
extern CPyModule *CPyModule_builtins_internal;
extern CPyModule *CPyModule_builtins;
extern CPyModule *CPyModule_time_internal;
extern CPyModule *CPyModule_time;
extern CPyTagged CPyDef_fib(CPyTagged cpy_r_n);
extern PyObject *CPyPy_fib(PyObject *self, PyObject *const *args, size_t nargs, PyObject *kwnames);
extern char CPyDef___top_level__(void);
#endif
