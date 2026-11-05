#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of Python lists to std::vector

namespace py = pybind11;

// Your fast C++ data processing function
std::vector<double> fast_process(const std::vector<double>& data, double factor) {
    std::vector<double> result;
    result.reserve(data.size());
    for (double val : data) {
        result.push_back(val * factor + 1.0); // Example computation
    }
    return result;
}

// Bind the function to Python
PYBIND11_MODULE(_core, m) {
    m.doc() = "BaguetteAI C++ core"; // optional module docstring
    m.def("fast_process", &fast_process, "Process data with C++ speed");
}
