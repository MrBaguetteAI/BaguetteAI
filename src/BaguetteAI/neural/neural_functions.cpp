#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using vec = std::vector<double>;
using mat = std::vector<std::vector<double>>;

// ---------------------------------------------------------------------------
// Small linear-algebra helpers
// ---------------------------------------------------------------------------
static vec matvec(const mat& W, const vec& v) {
    if (W.empty() || v.empty()) throw std::invalid_argument("matvec: empty");
    const size_t r = W.size(), c = W[0].size();
    if (c != v.size()) throw std::invalid_argument("matvec: dimension mismatch");
    vec out(r, 0.0);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            out[i] += W[i][j] * v[j];
    return out;
}

static vec vadd(const vec& a, const vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("vadd: mismatch");
    vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

static vec vsub(const vec& a, const vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("vsub: mismatch");
    vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}

static vec vmul(const vec& a, const vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("vmul: mismatch");
    vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] * b[i];
    return r;
}

// W^T @ d   — with W being (n_out, n_in), returns length n_in
static vec matT_vec(const mat& W, const vec& d) {
    if (W.empty() || d.empty()) throw std::invalid_argument("matT_vec: empty");
    const size_t n_out = W.size(), n_in = W[0].size();
    if (n_out != d.size()) throw std::invalid_argument("matT_vec: mismatch");
    vec r(n_in, 0.0);
    for (size_t i = 0; i < n_out; ++i)
        for (size_t j = 0; j < n_in; ++j)
            r[j] += W[i][j] * d[i];
    return r;
}

// ---------------------------------------------------------------------------
// Activations — one entry per supported name
// ---------------------------------------------------------------------------
static vec act_forward(const std::string& name, const vec& z) {
    vec r(z.size());
    if (name == "sigmoid") {
        for (size_t i = 0; i < z.size(); ++i) {
            const double x = z[i];
            r[i] = (x >= 0) ? 1.0 / (1.0 + std::exp(-x))
                            : std::exp(x) / (1.0 + std::exp(x));
        }
    } else if (name == "relu") {
        for (size_t i = 0; i < z.size(); ++i) r[i] = z[i] > 0 ? z[i] : 0.0;
    } else if (name == "tanh") {
        for (size_t i = 0; i < z.size(); ++i) r[i] = std::tanh(z[i]);
    } else if (name == "linear") {
        r = z;
    } else if (name == "softmax") {
        double m = *std::max_element(z.begin(), z.end());
        double s = 0.0;
        for (size_t i = 0; i < z.size(); ++i) { r[i] = std::exp(z[i] - m); s += r[i]; }
        for (auto& v : r) v /= s;
    } else {
        throw std::invalid_argument("Unknown activation: " + name);
    }
    return r;
}

// Derivative given BOTH z and a (the activation of z). Most activations need only one,
// but softmax's derivative is handled in the loss-specific delta, so it returns ones here.
static vec act_backward(const std::string& name, const vec& z, const vec& a) {
    vec r(a.size());
    if (name == "sigmoid") {
        for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] * (1.0 - a[i]);
    } else if (name == "relu") {
        for (size_t i = 0; i < z.size(); ++i) r[i] = z[i] > 0 ? 1.0 : 0.0;
    } else if (name == "tanh") {
        for (size_t i = 0; i < a.size(); ++i) r[i] = 1.0 - a[i] * a[i];
    } else if (name == "linear" || name == "softmax") {
        for (auto& v : r) v = 1.0;
    } else {
        throw std::invalid_argument("Unknown activation: " + name);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------
struct Layer {
    mat W;              // (n_out, n_in)
    vec b;              // (n_out,)
    std::string activation;
};

struct Network {
    std::vector<Layer> layers;
    std::string loss;   // "mse" | "cross_entropy"
};

struct Cache {
    std::vector<vec> a; // a[0]=input, a[L]=output
    std::vector<vec> z; // z[0] unused; z[l] for l=1..L
};

// ---------------------------------------------------------------------------
// Factory
//   layer_specs: list of (n_neurons, activation) for EVERY layer, input first.
//   e.g. [(784,"linear"), (128,"relu"), (10,"softmax")]
//   loss: "mse" | "cross_entropy"
// ---------------------------------------------------------------------------
Network Construct(const std::vector<std::pair<int, std::string>>& layer_specs,
                  const std::string& loss,
                  int seed = 42)
{
    if (layer_specs.size() < 2)
        throw std::invalid_argument("Construct: need >=2 layers (input + output)");
    if (loss != "mse" && loss != "cross_entropy")
        throw std::invalid_argument("Construct: loss must be 'mse' or 'cross_entropy'");

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> distr(-0.8, 0.8);

    Network net;
    net.loss = loss;

    // The "input layer" is only a shape declaration; no weights attach to it.
    for (size_t l = 0; l + 1 < layer_specs.size(); ++l) {
        const int n_in  = layer_specs[l].first;
        const int n_out = layer_specs[l + 1].first;
        Layer L;
        L.W = mat(n_out, vec(n_in));
        for (auto& row : L.W) for (auto& v : row) v = distr(gen);
        L.b = vec(n_out);
        for (auto& v : L.b) v = distr(gen);
        L.activation = layer_specs[l + 1].second;
        net.layers.push_back(std::move(L));
    }
    return net;
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
Cache Forward(const Network& net, const vec& x) {
    Cache c;
    const size_t L = net.layers.size();
    c.a.resize(L + 1);
    c.z.resize(L + 1);
    c.a[0] = x;
    for (size_t l = 0; l < L; ++l) {
        c.z[l + 1] = vadd(matvec(net.layers[l].W, c.a[l]), net.layers[l].b);
        c.a[l + 1] = act_forward(net.layers[l].activation, c.z[l + 1]);
    }
    return c;
}

// ---------------------------------------------------------------------------
// Loss + output delta
// ---------------------------------------------------------------------------
static double loss_value(const std::string& name, const vec& a, const vec& t) {
    double s = 0.0;
    if (name == "mse") {
        for (size_t i = 0; i < a.size(); ++i) {
            const double e = a[i] - t[i];
            s += 0.5 * e * e;
        }
    } else if (name == "cross_entropy") {
        const double eps = 1e-12;
        for (size_t i = 0; i < a.size(); ++i)
            if (t[i] > 0.0) s -= t[i] * std::log(std::max(a[i], eps));
    } else {
        throw std::invalid_argument("Unknown loss: " + name);
    }
    return s;
}

// Delta at the output layer:
//   mse + sigmoid:           (a - t) ⊙ σ'(a)
//   mse + linear:            (a - t)
//   cross_entropy + softmax: (a - t)          <-- the classic simplification
//   cross_entropy + sigmoid: (a - t)          <-- per-output binary CE
static vec output_delta(const Network& net, const Cache& c, const vec& t) {
    const size_t L = net.layers.size();
    const std::string& out_act = net.layers[L - 1].activation;
    vec diff = vsub(c.a[L], t);
    if (net.loss == "mse" && out_act == "sigmoid") {
        return vmul(diff, act_backward("sigmoid", c.z[L], c.a[L]));
    }
    return diff;   // cross-entropy w/ softmax or sigmoid, or mse w/ linear
}

// ---------------------------------------------------------------------------
// One SGD step
// ---------------------------------------------------------------------------
double TrainStep(Network& net, const vec& x, const vec& t, double lr) {
    const int L = static_cast<int>(net.layers.size());
    Cache c = Forward(net, x);
    const double loss = loss_value(net.loss, c.a[L], t);

    // deltas
    std::vector<vec> d(L + 1);
    d[L] = output_delta(net, c, t);
    for (int l = L - 1; l >= 1; --l) {
        vec incoming = matT_vec(net.layers[l].W, d[l + 1]);
        d[l] = vmul(incoming,
                    act_backward(net.layers[l - 1].activation, c.z[l], c.a[l]));
    }

    // updates
    for (int l = 0; l < L; ++l) {
        Layer& lay = net.layers[l];
        for (size_t i = 0; i < lay.W.size(); ++i) {
            for (size_t j = 0; j < lay.W[i].size(); ++j)
                lay.W[i][j] -= lr * d[l + 1][i] * c.a[l][j];
            lay.b[i] -= lr * d[l + 1][i];
        }
    }
    return loss;
}

// ---------------------------------------------------------------------------
// Target encoding
// ---------------------------------------------------------------------------
static mat one_hot(const std::vector<int>& labels, int n_classes) {
    mat R(labels.size(), vec(n_classes, 0.0));
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] < 0 || labels[i] >= n_classes)
            throw std::invalid_argument("one_hot: label out of range");
        R[i][labels[i]] = 1.0;
    }
    return R;
}

// ---------------------------------------------------------------------------
// Training loop
//   X       : (n_samples, n_in)
//   Y_int   : (n_samples,) int labels      [used when use_int_labels == true]
//   Y_float : (n_samples, n_out) targets   [used otherwise]
// ---------------------------------------------------------------------------
double TrainLoop(Network& net,
                 const mat& X,
                 const std::vector<int>& Y_int,
                 const mat& Y_float,
                 bool use_int_labels,
                 int epochs,
                 double lr,
                 int batch_size = 1,
                 int report_every = 1000,
                 int seed = 0,
                 bool shuffle = true)
{
    if (X.empty()) throw std::invalid_argument("TrainLoop: empty X");
    const size_t n = X.size();

    mat Y;
    if (use_int_labels) {
        if (Y_int.size() != n) throw std::invalid_argument("TrainLoop: Y_int length mismatch");
        const int n_out = static_cast<int>(net.layers.back().W.size());
        Y = one_hot(Y_int, n_out);
    } else {
        if (Y_float.size() != n) throw std::invalid_argument("TrainLoop: Y_float length mismatch");
        Y = Y_float;
    }

    if (batch_size < 1) batch_size = 1;

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 gen(seed);

    double last_epoch_loss = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (shuffle) std::shuffle(idx.begin(), idx.end(), gen);

        double total_loss = 0.0;

        if (batch_size == 1) {
            for (int i : idx) total_loss += TrainStep(net, X[i], Y[i], lr);
        } else {
            // mini-batch: accumulate gradients, average, update once
            for (size_t start = 0; start < n; start += batch_size) {
                const size_t end = std::min(start + static_cast<size_t>(batch_size), n);
                // accumulate dW, db over the batch, then apply averaged update
                std::vector<mat> dW(net.layers.size());
                std::vector<vec> db(net.layers.size());
                for (size_t l = 0; l < net.layers.size(); ++l) {
                    dW[l] = mat(net.layers[l].W.size(),
                                vec(net.layers[l].W[0].size(), 0.0));
                    db[l] = vec(net.layers[l].b.size(), 0.0);
                }
                for (size_t k = start; k < end; ++k) {
                    const int i = idx[k];
                    Cache c = Forward(net, X[i]);
                    total_loss += loss_value(net.loss, c.a.back(), Y[i]);

                    const int L = static_cast<int>(net.layers.size());
                    std::vector<vec> d(L + 1);
                    d[L] = output_delta(net, c, Y[i]);
                    for (int l = L - 1; l >= 1; --l) {
                        vec incoming = matT_vec(net.layers[l].W, d[l + 1]);
                        d[l] = vmul(incoming,
                                    act_backward(net.layers[l - 1].activation,
                                                 c.z[l], c.a[l]));
                    }
                    for (int l = 0; l < L; ++l) {
                        for (size_t r = 0; r < net.layers[l].W.size(); ++r) {
                            for (size_t c2 = 0; c2 < net.layers[l].W[r].size(); ++c2)
                                dW[l][r][c2] += d[l + 1][r] * c.a[l][c2];
                            db[l][r] += d[l + 1][r];
                        }
                    }
                }
                const double inv = 1.0 / static_cast<double>(end - start);
                for (size_t l = 0; l < net.layers.size(); ++l) {
                    for (size_t r = 0; r < net.layers[l].W.size(); ++r) {
                        for (size_t c2 = 0; c2 < net.layers[l].W[r].size(); ++c2)
                            net.layers[l].W[r][c2] -= lr * inv * dW[l][r][c2];
                        net.layers[l].b[r] -= lr * inv * db[l][r];
                    }
                }
            }
        }

        last_epoch_loss = total_loss / static_cast<double>(n);
        if (report_every > 0 && (epoch % report_every == 0))
            std::cout << "epoch " << epoch
                      << "  loss " << last_epoch_loss << "\n";
    }
    return last_epoch_loss;
}

// ---------------------------------------------------------------------------
// Inference helpers
// ---------------------------------------------------------------------------
vec Predict(const Network& net, const vec& x) {
    return Forward(net, x).a.back();
}

int Predict_Class(const Network& net, const vec& x) {
    const vec p = Predict(net, x);
    return static_cast<int>(std::distance(p.begin(),
                                          std::max_element(p.begin(), p.end())));
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(neural_functions, m) {
    m.doc() = "General-purpose C++ neural network core";

    py::class_<Layer>(m, "Layer")
        .def_readwrite("W", &Layer::W)
        .def_readwrite("b", &Layer::b)
        .def_readwrite("activation", &Layer::activation);

    py::class_<Network>(m, "Network")
        .def_readwrite("layers", &Network::layers)
        .def_readwrite("loss",   &Network::loss);

    py::class_<Cache>(m, "Cache")
        .def_readwrite("a", &Cache::a)
        .def_readwrite("z", &Cache::z);

    // factory
    m.def("Construct", &Construct,
          "Build a network. layer_specs = [(n_neurons, activation), ...], "
          "activation ∈ {'sigmoid','relu','tanh','linear','softmax'}, "
          "loss ∈ {'mse','cross_entropy'}.",
          py::arg("layer_specs"), py::arg("loss"), py::arg("seed") = 42);

    // single-sample primitives
    m.def("Forward",   &Forward);
    m.def("TrainStep", &TrainStep);
    m.def("Predict",       &Predict);
    m.def("Predict_Class", &Predict_Class);

    // batch trainer
    m.def("TrainLoop", &TrainLoop,
          "X: (n, n_in). Y_int: (n,) labels (use_int_labels=True). "
          "Y_float: (n, n_out) (use_int_labels=False).",
          py::arg("net"), py::arg("X"),
          py::arg("Y_int"), py::arg("Y_float"), py::arg("use_int_labels"),
          py::arg("epochs"), py::arg("lr"),
          py::arg("batch_size") = 1,
          py::arg("report_every") = 1000,
          py::arg("seed") = 0,
          py::arg("shuffle") = true);

    // raw activation helper for users who want to experiment
    m.def("act_forward",  &act_forward);
    m.def("act_backward", &act_backward);
}
