
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "compute_entropy.hh"

namespace py = pybind11;

namespace erick {
PYBIND11_MODULE(compute_entropy_python, m) {
  py::enum_<detail::Category>(m, "Category")
    .value("WRONG", detail::WRONG)
    .value("IN_WORD", detail::IN_WORD)
    .value("RIGHT", detail::RIGHT);
  m.def("compute_entropy", &compute_entropy);
  m.def("_compute_counts", &detail::compute_counts);
  m.def("_get_categories_for_index", [](const int idx, const int size){
    std::vector<detail::Category> out;
    detail::get_categories_for_index(idx, size, out);
    return out;
  });
}
} // namespace erick
