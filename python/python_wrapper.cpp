#include <sDbscan.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace python {

namespace py = pybind11;

PYBIND11_MODULE(sDbscan, m) { // Must be the same name with class Dbscan
    py::class_<sDbscan>(m, "sDbscan")
        .def(py::init<const int&, const int&>(),  py::arg("n_points"), py::arg("n_features"))
        .def("set_params", &sDbscan::set_params,
            py::arg("n_proj") = 1024, py::arg("topK") = 5, py::arg("topM") = 50,
            py::arg("distance") = "Cosine", py::arg("ker_n_features") = 1024, py::arg("ker_sigma") = 1.0,
            py::arg("ker_intervalSampling") = 0.4, py::arg("samplingProb") = 0.01,  py::arg("clusterNoise") = 0,
            py::arg("verbose") = false, py::arg("n_threads") = -1, py::arg("random_seed") = -1, py::arg("output") = ""
        )
        .def("set_sngParams", &sDbscan::set_sngParams,
        py::arg("distance") = "Cosine", py::arg("samplingProb") = 0.01, py::arg("clusterNoise") = 0,
        py::arg("verbose") = false, py::arg("n_threads") = -1, py::arg("random_seed") = -1, py::arg("output") = ""
        )
        .def_readonly("labels_", &sDbscan::labels_) // must be def_readonly
        .def_readonly("n_clusters_", &sDbscan::n_clusters_) // must be def_readonly
        .def_readonly("ordering_", &sDbscan::ordering_) // must be def_readonly
        .def_readonly("reachability_", &sDbscan::reachability_) // must be def_readonly
        .def("set_topM", &sDbscan::set_topM, py::arg("topM"))
        .def("set_threads", &sDbscan::set_threads, py::arg("n_threads"))
        .def("set_clusterNoise", &sDbscan::set_clusterNoise, py::arg("clusterNoise"))
        .def("clear", &sDbscan::clear)
        .def("set_topK", &sDbscan::set_topK, py::arg("topK"))

        // sDbscan
        .def("fit_sDbscan", &sDbscan::fit_sDbscan, py::arg("X"), py::arg("eps"), py::arg("minPts"))
        .def("load_fit_sDbscan", &sDbscan::load_fit_sDbscan, py::arg("dataset"), py::arg("eps"), py::arg("minPts"))
        .def("test_sDbscan", &sDbscan::test_sDbscan, py::arg("X"), py::arg("eps"), py::arg("rangeEps"), py::arg("minPts")) // not released in Python
        .def("load_test_sDbscan", &sDbscan::load_test_sDbscan, py::arg("dataset"), py::arg("eps"), py::arg("rangeEps"), py::arg("minPts"))

        // sOptics
        .def("fit_sOptics", &sDbscan::fit_sOptics, py::arg("X"), py::arg("eps"), py::arg("minPts"))
        .def("load_fit_sOptics", &sDbscan::load_fit_sOptics, py::arg("dataset"), py::arg("eps"), py::arg("minPts"))

        // sngDbscan
        .def("fit_sngDbscan", &sDbscan::fit_sngDbscan, py::arg("X"), py::arg("eps"), py::arg("minPts"))
        // do not support "load_fit_sngDbscan" as it runs very slow
        .def("test_sngDbscan", &sDbscan::test_sngDbscan, py::arg("X"), py::arg("eps"), py::arg("rangeEps"), py::arg("minPts")) // not released in Python
        .def("load_test_sngDbscan", &sDbscan::load_test_sngDbscan, py::arg("dataset"), py::arg("eps"), py::arg("rangeEps"), py::arg("minPts"))

        // sngOptics
        .def("fit_sngOptics", &sDbscan::fit_sngOptics, py::arg("X"), py::arg("eps"), py::arg("minPts"))
        .def("load_fit_sngOptics", &sDbscan::load_fit_sngOptics, py::arg("dataset"), py::arg("eps"), py::arg("minPts"))
        ;

} // namespace sDbscan
} // namespace python
