
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "compute_entropy.hh"

namespace py = pybind11;

namespace erick {
PYBIND11_MODULE(compute_entropy_python, m) {
  m.def("compute_entropy", &compute_entropy);
}
} // namespace erick
