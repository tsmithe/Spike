#pragma once

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include <sys/types.h>
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cctype>

#include <algorithm> 
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <random>
#include <mutex>
#include <thread>

class SpikeException : public std::exception {
public:
  SpikeException(std::string msg);
  const char* what() const noexcept override;
private:
  std::string _msg;
};

inline bool file_exists (const std::string& name) {
  if (FILE *file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}


/*** trim methods from https://stackoverflow.com/a/217605 ***/
// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}


template<typename T>
inline T infinity() { return std::numeric_limits<T>::infinity(); }

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef float FloatT;
typedef Eigen::Matrix<FloatT, Eigen::Dynamic, 1> EigenVector;
typedef Eigen::Matrix<FloatT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrix;
typedef Eigen::SparseMatrix<FloatT, Eigen::RowMajor> EigenSpMatrix;

inline void normalize_matrix_rows(EigenMatrix& R, FloatT scale=1) {
  //#pragma omp parallel for schedule(nonmonotonic:dynamic)
  for (int i = 0; i < R.rows(); ++i) {
    FloatT row_norm = R.row(i).norm();
    if (row_norm > 0)
      R.row(i) /= scale*row_norm;
  }
}

inline void normalize_matrix_rows(EigenSpMatrix& R, FloatT scale=1) {
  assert(R.rows() == R.outerSize());
  //#pragma omp parallel for schedule(nonmonotonic:dynamic)
  for (int i = 0; i < R.rows(); ++i) {
    FloatT row_norm = 0;
    for (EigenSpMatrix::InnerIterator it(R, i); it; ++it) {
      FloatT val = it.value();
      row_norm += val * val;
    }
    if (row_norm > 0) {
      for (EigenSpMatrix::InnerIterator it(R, i); it; ++it) {
        it.valueRef() /= scale*row_norm;
      }
    }
  }
}

namespace Eigen {

template<class Matrix>
inline void write_binary(const char* filename, const Matrix& matrix,
                         bool write_header = false){
  std::ofstream out(filename,
                    std::ios::out | std::ios::binary | std::ios::trunc);
  typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
  assert (rows > 0 && cols > 0);
  if (write_header) {
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
  }
  out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
  out.close();
}

template<class Matrix>
inline void read_binary(const char* filename, Matrix& matrix,
                        typename Matrix::Index rows=0,
                        typename Matrix::Index cols=0) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if (!in.good()) return;
  if (rows == 0 && cols == 0) {
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
  }
  matrix.resize(rows, cols);
  in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
  in.close();
}

// TODO: Fix RNG 
extern std::mt19937 global_random_generator;
inline EigenMatrix make_random_matrix(int J, int N, float scale=1,
                                      bool scale_by_norm=1, float sparseness=0,
                                      float mean=0, bool gaussian=0) {

  std::normal_distribution<> gauss;
  std::uniform_real_distribution<> U(0, 1);

  // J rows, each of N columns
  // Each row ~uniformly distributed on the N-sphere
  EigenMatrix R = EigenMatrix::Zero(J, N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < J; ++j) {
      if (sparseness > 0
          && U(global_random_generator) < sparseness) {
        R(j, i) = 0;
      } else {
        if (gaussian)
          R(j, i) = gauss(global_random_generator) + mean;
        else
          R(j, i) = U(global_random_generator) + mean;
      }
    }
  }

  if (scale_by_norm)
    normalize_matrix_rows(R, scale);
  else
    R.array() *= scale;

  return R;
}

} // namespace Eigen


template<typename FloatT>
class BetterTimer {
  unsigned _steps = 0;
  unsigned _seconds = 0;
  FloatT _fraction = 0;

public:
  void reset_timer() {
    _steps = 0; _seconds = 0; _fraction = 0;
  }

  void increment_time(FloatT dt) {
    _fraction += dt;
    if (_fraction > static_cast<FloatT>(1.0)) {
      ++_seconds;
      _fraction -= static_cast<FloatT>(1.0);
    }
    ++_steps;
  }

  FloatT current_time() const {
    return static_cast<FloatT>(_seconds) + _fraction;
  }

  unsigned const& timesteps() const {
    return _steps;
  }
};


struct EigenBuffer {
  std::ofstream file;

  inline void open(std::string fname) {
    assert(!is_open);
    filename = fname;
    file.open(filename, std::ofstream::out | std::ofstream::binary);
    is_open = true;
  }

  inline void lock() {
    buf_lock.lock();
  }

  inline void unlock() {
    buf_lock.unlock();
  }

  template<typename T>
  inline void push_back(int n, T const& b) {
    lock();
    buf.push_back(std::make_pair(n, b));
    unlock();
  }

  inline void pop_front() {
    lock();
    buf.pop_front();
    unlock();
  }

  inline auto& front() {
    return buf.front();
  }

  inline void clear() {
    lock();
    buf.clear();
    unlock();
  }

  inline int size() {
    lock();
    int size_ = buf.size();
    unlock();
    return size_;
  }

private:
  bool is_open = false;

  std::string filename;

  std::list<std::pair<int, EigenMatrix> > buf;
  std::mutex buf_lock;
};

class BufferWriter {
public:
  BufferWriter() = default;
  ~BufferWriter();

  void add_buffer(EigenBuffer* buf);

  void write_output();
  void write_loop();

  void start();
  void stop();

  void block_until_empty() const;

  std::vector<EigenBuffer*> buffers;
  FloatT time_since_last_flush = 0;
  std::thread othread; // TODO: Perhaps having too many
                       //       output threads will cause too much
                       //       seeking on disk, thus slowing things down?
                       // Perhaps better just to have one global thread?
                       // Or one thread per Electrodes?
private:
  bool running = false;
};

extern BufferWriter global_writer;
