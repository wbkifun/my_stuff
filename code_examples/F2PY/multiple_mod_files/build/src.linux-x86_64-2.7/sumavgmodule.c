/* File: sumavgmodule.c
 * This file is auto-generated with f2py (version:2).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * See http://cens.ioc.ee/projects/f2py2e/
 * Generation date: Wed Sep 26 01:58:45 2012
 * $Revision:$
 * $Date:$
 * Do not edit this file directly unless you know what you are doing!!!
 */
#ifdef __cplusplus
extern "C" {
#endif

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include "fortranobject.h"
/*need_includes0*/

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *sumavg_error;
static PyObject *sumavg_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
/*need_typedefs*/

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
  PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
  fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif

#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int double_from_pyobj(double* v,PyObject *obj,const char *errmess) {
  PyObject* tmp = NULL;
  if (PyFloat_Check(obj)) {
#ifdef __sgi
    *v = PyFloat_AsDouble(obj);
#else
    *v = PyFloat_AS_DOUBLE(obj);
#endif
    return 1;
  }
  tmp = PyNumber_Float(obj);
  if (tmp) {
#ifdef __sgi
    *v = PyFloat_AsDouble(tmp);
#else
    *v = PyFloat_AS_DOUBLE(tmp);
#endif
    Py_DECREF(tmp);
    return 1;
  }
  if (PyComplex_Check(obj))
    tmp = PyObject_GetAttrString(obj,"real");
  else if (PyString_Check(obj) || PyUnicode_Check(obj))
    /*pass*/;
  else if (PySequence_Check(obj))
    tmp = PySequence_GetItem(obj,0);
  if (tmp) {
    PyErr_Clear();
    if (double_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
    Py_DECREF(tmp);
  }
  {
    PyObject* err = PyErr_Occurred();
    if (err==NULL) err = sumavg_error;
    PyErr_SetString(err,errmess);
  }
  return 0;
}

static int float_from_pyobj(float* v,PyObject *obj,const char *errmess) {
  double d=0.0;
  if (double_from_pyobj(&d,obj,errmess)) {
    *v = (float)d;
    return 1;
  }
  return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/************************************ sum ************************************/
static char doc_f2py_rout_sumavg_summod_sum[] = "\
Function signature:\n\
  sum = sum(a,b,c)\n\
Required arguments:\n"
"  a : input float\n"
"  b : input float\n"
"  c : input float\n"
"Return objects:\n"
"  sum : float";
/* #declfortranroutine# */
static PyObject *f2py_rout_sumavg_summod_sum(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(float*,float*,float*,float*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  float sum = 0;
  float a = 0;
  PyObject *a_capi = Py_None;
  float b = 0;
  PyObject *b_capi = Py_None;
  float c = 0;
  PyObject *c_capi = Py_None;
  static char *capi_kwlist[] = {"a","b","c",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOO:sumavg.summod.sum",\
    capi_kwlist,&a_capi,&b_capi,&c_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable a */
    f2py_success = float_from_pyobj(&a,a_capi,"sumavg.summod.sum() 1st argument (a) can't be converted to float");
  if (f2py_success) {
  /* Processing variable sum */
  /* Processing variable b */
    f2py_success = float_from_pyobj(&b,b_capi,"sumavg.summod.sum() 2nd argument (b) can't be converted to float");
  if (f2py_success) {
  /* Processing variable c */
    f2py_success = float_from_pyobj(&c,c_capi,"sumavg.summod.sum() 3rd argument (c) can't be converted to float");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  (*f2py_func)(&sum,&a,&b,&c);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("f",sum);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*if (f2py_success) of c*/
  /* End of cleaning variable c */
  } /*if (f2py_success) of b*/
  /* End of cleaning variable b */
  /* End of cleaning variable sum */
  } /*if (f2py_success) of a*/
  /* End of cleaning variable a */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/********************************* end of sum *********************************/

/************************************ avg ************************************/
static char doc_f2py_rout_sumavg_avgmod_avg[] = "\
Function signature:\n\
  avg = avg(a,b,c)\n\
Required arguments:\n"
"  a : input float\n"
"  b : input float\n"
"  c : input float\n"
"Return objects:\n"
"  avg : float";
/* #declfortranroutine# */
static PyObject *f2py_rout_sumavg_avgmod_avg(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(float*,float*,float*,float*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  float avg = 0;
  float a = 0;
  PyObject *a_capi = Py_None;
  float b = 0;
  PyObject *b_capi = Py_None;
  float c = 0;
  PyObject *c_capi = Py_None;
  static char *capi_kwlist[] = {"a","b","c",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOO:sumavg.avgmod.avg",\
    capi_kwlist,&a_capi,&b_capi,&c_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable a */
    f2py_success = float_from_pyobj(&a,a_capi,"sumavg.avgmod.avg() 1st argument (a) can't be converted to float");
  if (f2py_success) {
  /* Processing variable c */
    f2py_success = float_from_pyobj(&c,c_capi,"sumavg.avgmod.avg() 3rd argument (c) can't be converted to float");
  if (f2py_success) {
  /* Processing variable avg */
  /* Processing variable b */
    f2py_success = float_from_pyobj(&b,b_capi,"sumavg.avgmod.avg() 2nd argument (b) can't be converted to float");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  (*f2py_func)(&avg,&a,&b,&c);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("f",avg);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*if (f2py_success) of b*/
  /* End of cleaning variable b */
  /* End of cleaning variable avg */
  } /*if (f2py_success) of c*/
  /* End of cleaning variable c */
  } /*if (f2py_success) of a*/
  /* End of cleaning variable a */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/********************************* end of avg *********************************/

/*********************************** sumavg ***********************************/
static char doc_f2py_rout_sumavg_sumavgmod_sumavg[] = "\
Function signature:\n\
  sumavg(a,b,c)\n\
Required arguments:\n"
"  a : input float\n"
"  b : input float\n"
"  c : input float";
/*  */
static PyObject *f2py_rout_sumavg_sumavgmod_sumavg(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(float*,float*,float*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  float a = 0;
  PyObject *a_capi = Py_None;
  float b = 0;
  PyObject *b_capi = Py_None;
  float c = 0;
  PyObject *c_capi = Py_None;
  static char *capi_kwlist[] = {"a","b","c",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOO:sumavg.sumavgmod.sumavg",\
    capi_kwlist,&a_capi,&b_capi,&c_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable a */
    f2py_success = float_from_pyobj(&a,a_capi,"sumavg.sumavgmod.sumavg() 1st argument (a) can't be converted to float");
  if (f2py_success) {
  /* Processing variable c */
    f2py_success = float_from_pyobj(&c,c_capi,"sumavg.sumavgmod.sumavg() 3rd argument (c) can't be converted to float");
  if (f2py_success) {
  /* Processing variable b */
    f2py_success = float_from_pyobj(&b,b_capi,"sumavg.sumavgmod.sumavg() 2nd argument (b) can't be converted to float");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(&a,&b,&c);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("");
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*if (f2py_success) of b*/
  /* End of cleaning variable b */
  } /*if (f2py_success) of c*/
  /* End of cleaning variable c */
  } /*if (f2py_success) of a*/
  /* End of cleaning variable a */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/******************************* end of sumavg *******************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/

static FortranDataDef f2py_summod_def[] = {
  {"sum",-1,{{-1}},0,NULL,(void *)f2py_rout_sumavg_summod_sum,doc_f2py_rout_sumavg_summod_sum},
  {NULL}
};

static void f2py_setup_summod(char *sum) {
  int i_f2py=0;
  f2py_summod_def[i_f2py++].data = sum;
}
extern void F_FUNC(f2pyinitsummod,F2PYINITSUMMOD)(void (*)(char *));
static void f2py_init_summod(void) {
  F_FUNC(f2pyinitsummod,F2PYINITSUMMOD)(f2py_setup_summod);
}


static FortranDataDef f2py_avgmod_def[] = {
  {"avg",-1,{{-1}},0,NULL,(void *)f2py_rout_sumavg_avgmod_avg,doc_f2py_rout_sumavg_avgmod_avg},
  {NULL}
};

static void f2py_setup_avgmod(char *avg) {
  int i_f2py=0;
  f2py_avgmod_def[i_f2py++].data = avg;
}
extern void F_FUNC(f2pyinitavgmod,F2PYINITAVGMOD)(void (*)(char *));
static void f2py_init_avgmod(void) {
  F_FUNC(f2pyinitavgmod,F2PYINITAVGMOD)(f2py_setup_avgmod);
}


static FortranDataDef f2py_sumavgmod_def[] = {
  {"sumavg",-1,{{-1}},0,NULL,(void *)f2py_rout_sumavg_sumavgmod_sumavg,doc_f2py_rout_sumavg_sumavgmod_sumavg},
  {NULL}
};

static void f2py_setup_sumavgmod(char *sumavg) {
  int i_f2py=0;
  f2py_sumavgmod_def[i_f2py++].data = sumavg;
}
extern void F_FUNC(f2pyinitsumavgmod,F2PYINITSUMAVGMOD)(void (*)(char *));
static void f2py_init_sumavgmod(void) {
  F_FUNC(f2pyinitsumavgmod,F2PYINITSUMAVGMOD)(f2py_setup_sumavgmod);
}

/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "sumavg",
  NULL,
  -1,
  f2py_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};
#endif

#if PY_VERSION_HEX >= 0x03000000
#define RETVAL m
PyObject *PyInit_sumavg(void) {
#else
#define RETVAL
PyMODINIT_FUNC initsumavg(void) {
#endif
  int i;
  PyObject *m,*d, *s;
#if PY_VERSION_HEX >= 0x03000000
  m = sumavg_module = PyModule_Create(&moduledef);
#else
  m = sumavg_module = Py_InitModule("sumavg", f2py_module_methods);
#endif
  Py_TYPE(&PyFortran_Type) = &PyType_Type;
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module sumavg (failed to import numpy)"); return RETVAL;}
  d = PyModule_GetDict(m);
  s = PyString_FromString("$Revision: $");
  PyDict_SetItemString(d, "__version__", s);
#if PY_VERSION_HEX >= 0x03000000
  s = PyUnicode_FromString(
#else
  s = PyString_FromString(
#endif
    "This module 'sumavg' is auto-generated with f2py (version:2).\nFunctions:\n"
"Fortran 90/95 modules:\n""  summod --- sum()""  avgmod --- avg()""  sumavgmod --- sumavg()"".");
  PyDict_SetItemString(d, "__doc__", s);
  sumavg_error = PyErr_NewException ("sumavg.error", NULL, NULL);
  Py_DECREF(s);
  for(i=0;f2py_routine_defs[i].name!=NULL;i++)
    PyDict_SetItemString(d, f2py_routine_defs[i].name,PyFortranObject_NewAsAttr(&f2py_routine_defs[i]));



/*eof initf2pywraphooks*/
  PyDict_SetItemString(d, "sumavgmod", PyFortranObject_New(f2py_sumavgmod_def,f2py_init_sumavgmod));
  PyDict_SetItemString(d, "avgmod", PyFortranObject_New(f2py_avgmod_def,f2py_init_avgmod));
  PyDict_SetItemString(d, "summod", PyFortranObject_New(f2py_summod_def,f2py_init_summod));
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
  if (! PyErr_Occurred())
    on_exit(f2py_report_on_exit,(void*)"sumavg");
#endif

  return RETVAL;
}
#ifdef __cplusplus
}
#endif
