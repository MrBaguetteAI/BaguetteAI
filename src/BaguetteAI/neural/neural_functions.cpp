#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of Python lists to std::vector

namespace py = pybind11;

using matrix1D = std::vector<double>;
using matrix2D = std::vector<std::vector<double>>;
using matrix3D = std::vector<std::vector<std::vector<double>>>;

struct Network {
  matrix3D W;   // W[l] : (sizes[l+1] × sizes[l])
  matrix3D b;   // b[l] : (sizes[l+1] × 1)
};

// forward: returns (activations per layer, z's per layer)
struct Cache {
    matrix3D a;   // a[0]=input, a[L]=output
    matrix3D z;   // z[1..L]
};


matrix2D Matrix_Multiply(const matrix2D& A, const matrix2D& B){
  if (A.empty() || B.empty()) {
    throw std::invalid_argument("Matrix_Multiply: Matrices cannot be empty");
  }
  int r1 = A.size();
  int c1 = A[0].size();
  int r2 = B.size();
  int c2 = B[0].size();

  if(c1==r2){
    matrix2D result(r1,matrix1D(c2,0)); // Initializing result matrix

    for(size_t i=0;i<r1;i++){
      for(size_t j=0;j<c2;j++){
        for(size_t k=0;k<c1;k++){
          result[i][j]+=A[i][k]*B[k][j];
        }
      }
    }
  return result;
  }

  else{
    throw std::invalid_argument("Matrix_Multiply: dimension mismatch");
  }
}


matrix2D Matrix_ADD(const matrix2D& A, const matrix2D& B){
  if (A.empty() || B.empty()) {
    throw std::invalid_argument("Matrix_ADD: Matrices cannot be empty");
  }

  int r1 = A.size();
  int c1 = A[0].size();
  int r2 = B.size();
  int c2 = B[0].size();

  if(c1==c2&&r1==r2){
    matrix2D result(r1,matrix1D(c2,0)); // Initializing result matrix
    for(size_t i=0;i<r1;i++){
      for(size_t j=0;j<c1;j++){
        result[i][j]=A[i][j]+B[i][j];
      }
    }
    return result;
  }
  else{
    throw std::invalid_argument("Matrix_ADD: dimension mismatch");
  }
}


matrix2D Matrix_Subtract(const matrix2D& A, const matrix2D& B){
  if (A.empty() || B.empty()) {
    throw std::invalid_argument("Matrix_Subtract: Matrices cannot be empty");
  }

  int r1 = A.size();
  int c1 = A[0].size();
  int r2 = B.size();
  int c2 = B[0].size();

  if(c1==c2&&r1==r2){
    matrix2D result(r1,matrix1D(c2,0)); // Initializing result matrix
    for(size_t i=0;i<r1;i++){
      for(size_t j=0;j<c1;j++){
        result[i][j]=A[i][j]-B[i][j];
      }
    }
    return result;
  }
  else{
    throw std::invalid_argument("Matrix_ADD: dimension mismatch");
  }
}


matrix2D Matrix_Transpose(const matrix2D& A){
  if (A.empty()) throw std::invalid_argument("Matrix_Transpose: empty");
  size_t r = A.size(), c = A[0].size();
  matrix2D T(c, matrix1D(r, 0.0));
  for (size_t i = 0; i < r; ++i)
    for (size_t j = 0; j < c; ++j)
      T[j][i] = A[i][j];
  return T;
}


matrix2D Matrix_Scale(const matrix2D& A, double s) {
    matrix2D R = A;
    for (auto& row : R) for (auto& v : row) v *= s;
    return R;
}


matrix2D sigmoid(const matrix2D& A){
  matrix2D R = A;
  for (size_t i = 0; i < R.size(); ++i)
    for (size_t j = 0; j < R[i].size(); ++j) {
      double x = R[i][j];
      if (x >= 0) R[i][j] = 1.0 / (1.0 + std::exp(-x));
      else        R[i][j] = std::exp(x) / (1.0 + std::exp(x));
    }
  return R;
}


matrix2D Hadamard(const matrix2D& A, const matrix2D& B){
  if (A.empty() || B.empty())
    throw std::invalid_argument("Hadamard: empty matrix");
  /*This function computes the Hadamard product — the element-wise multiplication of two matrices of identical shape.*/
  if (A.size() != B.size() || A[0].size() != B[0].size())
    throw std::invalid_argument("Hadamard: dimension mismatch");

  matrix2D R = A;

  for (size_t i = 0; i < A.size(); ++i)
    for (size_t j = 0; j < A[i].size(); ++j)
      R[i][j] = A[i][j] * B[i][j];
  return R;
}


matrix2D Sigmoid_Derivative(const matrix2D& a){
  // σ'(z) = a·(1−a)
  matrix2D R = a;
  for (size_t i = 0; i < R.size(); ++i)
    for (size_t j = 0; j < R[i].size(); ++j)
      R[i][j] = a[i][j] * (1.0 - a[i][j]);
  return R;
}

// Output layer: d^L = (a − t) ⊙ σ'(a)
matrix2D Backward_Output(const matrix2D& a, const matrix2D& t) {
  return Hadamard(Matrix_Subtract(a, t), Sigmoid_Derivative(a));
}


// Hidden layer: d^l = ((W^{l+1})ᵀ · d^{l+1}) ⊙ σ'(a^l)
matrix2D Backward_Hidden(const matrix2D& a,            // activation of THIS layer
                         const matrix2D& W_next,       // weights of NEXT layer
                         const matrix2D& d_next) {     // delta of NEXT layer
  matrix2D incoming = Matrix_Multiply(Matrix_Transpose(W_next), d_next);
  return Hadamard(incoming, Sigmoid_Derivative(a));
}


Network Construct_Neural_Net(const std::vector<int>& sizes) {
  // std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_int_distribution<> distr(-0.8,0.8);
  
  // sizes = {InputN, HiddenN, ..., OutputN}; length = TotalL + 1 if TotalL counts weighted layers
  if (sizes.size() < 2)
      throw std::invalid_argument("Construct_Neural_Net: need at least 2 layer sizes");

  Network net;
  size_t L = sizes.size() - 1;          // number of weighted layers
  net.W.resize(L);
  net.b.resize(L);

  for (size_t l = 0; l < L; ++l) {
    int n_in  = sizes[l];
    int n_out = sizes[l + 1];
    net.W[l] = matrix2D(n_out, matrix1D(n_in, distr(gen)));
    for (auto& row : net.W[l])
        for (auto& v : row)
            v = distr(gen); 
    net.b[l] = matrix2D(n_out, matrix1D(1,   distr(gen)));
  }
  return net;
}


Cache Forward(const Network& net, const matrix2D& x) {
  size_t L = net.W.size();
  Cache c;
  c.a.resize(L+1);
  c.z.resize(L+1);
  c.a[0] = x;

  for (size_t l = 0; l < L; ++l) {
    c.z[l+1] = Matrix_ADD(Matrix_Multiply(net.W[l], c.a[l]), net.b[l]);
    c.a[l+1] = sigmoid(c.z[l+1]);
  }
   return c;
}

double TrainStep(Network& net, const matrix2D& x,
                 const matrix2D& t, double lr) {
  int L = net.W.size();
  Cache c = Forward(net, x);

  // loss
  double loss = 0.0;
  for (size_t i = 0; i < c.a[L].size(); ++i) {
      double e = c.a[L][i][0] - t[i][0];
      loss += 0.5 * e * e;
  }

  // deltas
  std::vector<matrix2D> d(L+1);
  d[L] = Backward_Output(c.a[L], t);
  for (int l = L-1; l >= 1; --l)
      d[l] = Backward_Hidden(c.a[l], net.W[l], d[l+1]);

  // gradients + update
  for (size_t l = 0; l < L; ++l) {
      matrix2D dW = Matrix_Multiply(d[l+1], Matrix_Transpose(c.a[l]));
      net.W[l] = Matrix_Subtract(net.W[l], Matrix_Scale(dW, lr));   // add Matrix_Scale helper
      net.b[l] = Matrix_Subtract(net.b[l], Matrix_Scale(d[l+1], lr));
  }
  return loss;
}

double TrainLoop(Network& net,
                 const matrix3D& X,          // list of inputs, each a matrix2D column vector
                 const matrix3D& Y,          // list of targets, each a matrix2D column vector
                 int epochs,
                 double lr,
                 int report_every = 1000)
                 // int seed = 0,
                 // bool shuffle = true)
{
    if (X.size() != Y.size())
        throw std::invalid_argument("TrainLoop: X and Y must have the same length");
    if (X.empty())
        throw std::invalid_argument("TrainLoop: dataset is empty");

    const size_t n = X.size();

    // // index vector we shuffle each epoch
    // std::vector<int> idx(n);
    // std::iota(idx.begin(), idx.end(), 0);
    //
    // std::mt19937 gen(seed);
    double last_epoch_loss = 0.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // if (shuffle)
        //     std::shuffle(idx.begin(), idx.end(), gen);

        double total_loss = 0.0;
        for (int i = 0; i < static_cast<int>(n); ++i) {
            total_loss += TrainStep(net, X[i], Y[i], lr);
        }
        last_epoch_loss = total_loss / static_cast<double>(n);

        if (report_every > 0 && (epoch % report_every == 0)) {
            std::cout << "epoch " << epoch
                      << "  loss " << last_epoch_loss << "\n";
        }
    }
    return last_epoch_loss;
}


int main_run(){
  matrix2D input = {
    {1.0},
    {0.5}
  };

  matrix2D w1 = {
    {0.5,-0.3},
    {0.8,0.2}
  };

  matrix2D b1 = {
    {0.1},
    {-0.1}
  };

  matrix2D w2 = {
    {0.6,-0.4}
  };

  matrix2D b2 = {
    {0.05}
  };

  matrix2D target = {
    {1.0}
  };

  matrix2D z1_temp = Matrix_Multiply(w1,input);
  matrix2D z1 = Matrix_ADD(z1_temp, b1);
  matrix2D a1=sigmoid(z1);

  matrix2D z2_temp = Matrix_Multiply(w2,a1);
  matrix2D z2 = Matrix_ADD(z2_temp, b2);
  matrix2D a2 = sigmoid(z2);

  for(const auto& row : a2){
    for(auto val : row){
      std::cout << val <<" ";
    }
    std::cout << "\n";
  }
  return 0;
}

PYBIND11_MODULE(neural_functions, m) {
    m.doc() = "BaguetteAI neural network C++ core";

    // ---- bind structs first (required for the functions below) ----
    py::class_<Network>(m, "Network")
        .def_readwrite("W", &Network::W)
        .def_readwrite("b", &Network::b);

    py::class_<Cache>(m, "Cache")
        .def_readwrite("a", &Cache::a)
        .def_readwrite("z", &Cache::z);

    // ---- matrix primitives ----
    m.def("Matrix_Multiply",    &Matrix_Multiply,    "Multiply two 2-D matrices");
    m.def("Matrix_ADD",         &Matrix_ADD,         "Add two 2-D matrices");
    m.def("Matrix_Subtract",    &Matrix_Subtract,    "Subtract two 2-D matrices");
    m.def("Matrix_Transpose",   &Matrix_Transpose,   "Transpose the Matrix");
    m.def("Matrix_Scale",       &Matrix_Scale,       "Multiply each entry by a scalar");
    m.def("sigmoid",            &sigmoid,            "Element-wise sigmoid");
    m.def("Hadamard",           &Hadamard,           "Element-wise multiply");
    m.def("Sigmoid_Derivative", &Sigmoid_Derivative, "Sigmoid derivative (from activation)");

    // ---- network functions ----
    m.def("Backward_Output",      &Backward_Output,      "Output-layer delta");
    m.def("Backward_Hidden",      &Backward_Hidden,      "Hidden-layer delta");
    m.def("Construct_Neural_Net", &Construct_Neural_Net, "Build a network from layer sizes");
    m.def("Forward",              &Forward,              "Return Cache (activations + z)");
    m.def("TrainStep",            &TrainStep,            "One SGD step, mutates net");
    m.def("TrainLoop", &TrainLoop,
          "Run a full training loop over (X, Y). Mutates net in place.",
          py::arg("net"),
          py::arg("X"),
          py::arg("Y"),
          py::arg("epochs"),
          py::arg("lr"),
          py::arg("report_every") = 1000);
          // py::arg("seed")         = 0,
          // py::arg("shuffle")      = true);

    m.def("main_run",             &main_run,             "Demo network");
}
