#include <Python.h>

extern PyObject* initFlagosModule(void);

PyMODINIT_FUNC PyInit__C(void) {
  return initFlagosModule();
}
