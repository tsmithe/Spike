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
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <utility>
#include <vector>

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

template<typename T>
inline T infinity() { return std::numeric_limits<T>::infinity(); }

#include <Eigen/Dense>

typedef float FloatT;
typedef Eigen::VectorXf EigenVector;
typedef Eigen::MatrixXf EigenMatrix;

inline void normalize_matrix_rows(EigenMatrix& R, FloatT scale=1) {
  for (int j = 0; j < R.rows(); ++j) {
    FloatT row_norm = R.row(j).norm();
    if (row_norm > 0)
      R.row(j) /= scale*row_norm;
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

inline EigenMatrix make_random_matrix(int J, int N, float scale,
                                      bool scale_by_norm=1, float sparseness=0,
                                      float mean=0) {
  auto global_random_generator = std::mt19937();

  // J rows, each of N columns
  // Each row ~uniformly distributed on the N-sphere
  EigenMatrix R = EigenMatrix::Zero(J, N);
  std::normal_distribution<> gauss;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < J; ++j) {
      R(j, i) = gauss(global_random_generator) + mean;
    }
  }
  if (scale_by_norm)
    normalize_matrix_rows(R, scale);
  else
    R.array() *= scale;

  if (sparseness > 0) {
    std::uniform_real_distribution<> U(0, 1);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < J; ++j) {
        if (U(global_random_generator) < sparseness)
          R(j, i) = 0;
      }
    }
  }

  return R;
}

} // namespace Eigen

// Forward definitions:
// |---
class RateNeurons;
class RateNeuronGroup;
//class DummyRateNeurons;
//class InputDummyRateNeurons;
class RateSynapses;
class RateSynapseGroup;
//class RatePlasticity;
class RateElectrodes;
class RateModel;
// ---|

// TODO: Be more protective about friends!
//        -- That is, make more class members protected / private,
//           and specify friendship relationships to allow proper access.

namespace Backend {
  class RateSynapses : public virtual SpikeBackendBase {
  public:
    ~RateSynapses() override = default;
    SPIKE_ADD_BACKEND_FACTORY(RateSynapses);
    void prepare() override = 0;
    void reset_state() override = 0;

    virtual void add_group(RateSynapseGroup* group) = 0;

    // virtual void update_activation(FloatT dt) = 0;
    virtual const EigenVector& activation() = 0;
    virtual const EigenMatrix& weights() = 0;
    virtual void weights(EigenMatrix const& w) = 0;

    virtual void delay(unsigned int) = 0;
    virtual unsigned int delay() = 0;
  };

  /*
  class RatePlasticity : public virtual SpikeBackendBase {
  public:
    ~RatePlasticity() override = default;
    SPIKE_ADD_BACKEND_FACTORY(RatePlasticity);
    void prepare() override = 0;
    void reset_state() override = 0;

    virtual void apply_plasticity(FloatT dt) = 0;

    virtual void multipliers(EigenMatrix const&) = 0;
  };
  */

  /*
  class HebbPlasticity : public virtual RatePlasticity {
  public:
    SPIKE_ADD_BACKEND_FACTORY(HebbPlasticity);
  };
  */

  class RateNeurons : public virtual SpikeBackendBase {
  public:
    ~RateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(RateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;

    virtual void add_group(RateNeuronGroup* group) = 0;

    virtual void connect_input(RateSynapses* synapses/*,
                               RatePlasticity* plasticity*/) = 0;
    virtual bool staged_integrate_timestep(FloatT dt) = 0;
    virtual const EigenVector& rate() = 0;
  };

  /*
  class DummyRateNeurons : public virtual RateNeurons {
  public:
    ~DummyRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(DummyRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    virtual void connect_input(RateSynapses* synapses,
                               RatePlasticity* plasticity) = 0;
    virtual void add_rate(FloatT duration, EigenVector rates) = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    virtual const EigenVector& rate() = 0;
  };

  class InputDummyRateNeurons : public virtual DummyRateNeurons {
  public:
    ~InputDummyRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(InputDummyRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    virtual const EigenVector& rate() = 0;
  };
  */

  /*
  class RateElectrodes : public virtual SpikeBackendBase {
  public:
    ~RateElectrodes() override = default;
    SPIKE_ADD_BACKEND_FACTORY(RateElectrodes);
  };

  class RateModel : public virtual SpikeBackendBase {
  public:
    ~RateModel() override = default;
    SPIKE_ADD_BACKEND_FACTORY(RateModel);
  };
  */
}

static_assert(std::has_virtual_destructor<Backend::RateNeurons>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::RateSynapses>::value,
              "contract violated");
// static_assert(std::has_virtual_destructor<Backend::RatePlasticity>::value,
//               "contract violated");
/*
static_assert(std::has_virtual_destructor<Backend::RateModel>::value,
              "contract violated");
*/

struct EigenBuffer {
  std::list<std::pair<int, EigenMatrix> > buf;
  std::mutex lock;

  template<typename T>
  inline void push_back(int n, T const& b) {
    lock.lock();
    buf.push_back(std::make_pair(n, b));
    lock.unlock();
  }

  inline void clear() {
    lock.lock();
    buf.clear();
    lock.unlock();
  }

  inline int size() {
    lock.lock();
    int size_ = buf.size();
    lock.unlock();
    return size_;
  }
};

class BufferWriter {
public:
  BufferWriter(const std::string& filename_, EigenBuffer& buf_);
  ~BufferWriter();

  void write_output();
  void write_loop();

  void start();
  void stop();

  void block_until_empty() const;

  EigenBuffer& buffer;
  std::string filename;
  std::ofstream file;
  std::thread othread; // TODO: Perhaps having too many
                       //       output threads will cause too much
                       //       seeking on disk, thus slowing things down?
                       // Perhaps better just to have one global thread?
                       // Or one thread per Electrodes?
private:
  bool running = false;
};

struct RateNeuronGroup {
  std::string label;
  unsigned int size;
  FloatT alpha = 0.0;
  FloatT beta = 1.0;
  FloatT tau = 1.0;

  RateNeurons* parent = nullptr;
  unsigned int start;

  const EigenVector rate() const;
};

class RateNeurons : public virtual SpikeBase {
public:
  RateNeurons(Context* ctx, std::string _label);
  ~RateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateNeurons, SpikeBase);

  void reset_state() override;

  virtual int add_group(RateNeuronGroup* group);

  void assert_dendritic_consistency(RateSynapses* synapses/*,
                                    RatePlasticity* plasticity*/) const;
  void assert_dendritic_consistency() const;
  void connect_input(RateSynapses* synapses/*,
                     RatePlasticity* plasticity=nullptr*/);

  bool staged_integrate_timestep(FloatT dt);
  void apply_plasticity(FloatT dt) const;

  std::string label;

  int size = 0;

  std::vector<RateNeuronGroup*> neuron_groups;

  int timesteps = 0;

  const EigenVector& rate() const;
  int rate_buffer_interval = 0;
  EigenBuffer rate_history;

  std::vector</*std::pair<*/RateSynapses*/*, RatePlasticity*>*/ > dendrites;

private:
  std::shared_ptr<::Backend::RateNeurons> _backend;
};

/*
class DummyRateNeurons : public virtual RateNeurons {
public:
  /*
  DummyRateNeurons(Context* ctx, int size_, std::string label_/*,
                   FloatT t_on_, FloatT t_off_,
                   EigenVector const& x_on_, EigenVector const& x_off_* /);
  * /
  DummyRateNeurons(Context* ctx, int size_, std::string label_);
  ~DummyRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(DummyRateNeurons, RateNeurons);

  std::vector<std::pair<FloatT, EigenVector> > rate_schedule;
  void add_rate(FloatT duration, EigenVector const& rates);
  
private:
  std::shared_ptr<::Backend::DummyRateNeurons> _backend;
};

class InputDummyRateNeurons : public virtual DummyRateNeurons {
public:
  InputDummyRateNeurons(Context* ctx, int size_, std::string label_,
                        FloatT sigma_IN_, FloatT lambda_, // gamma_,
                        FloatT revolutions_per_second_);
  ~InputDummyRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(InputDummyRateNeurons, DummyRateNeurons);

  FloatT sigma_IN;
  FloatT /*gamma,* / lambda;
  FloatT revolutions_per_second;
  FloatT t_stop_after = infinity<FloatT>();

  EigenVector theta_pref;

private:
  std::shared_ptr<::Backend::InputDummyRateNeurons> _backend;
};
*/

struct RateSynapseGroup {
  RateNeuronGroup* neurons_pre = nullptr;
  RateNeuronGroup* neurons_post = nullptr;

  std::string label;

  unsigned int delay() const;
  void delay(unsigned int d);

  const EigenVector activation() const;
  const EigenMatrix weights() const; // just single, instantaneous dense weights for now
  void weights(const EigenMatrix& w);

  FloatT scaling = 1;

  RateSynapses* parent = nullptr;
};
  

class RateSynapses : public virtual SpikeBase {
public:
  RateSynapses(Context* ctx,
               RateNeurons* neurons/*_pre_, RateNeurons* neurons_post_,
               FloatT scaling_=1, std::string label_=""*/);
  ~RateSynapses() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateSynapses, SpikeBase);

  void reset_state() override;

  // void update_activation(FloatT dt);

  RateNeurons* neurons = nullptr;
  // RateNeurons* neurons_pre = nullptr;
  // RateNeurons* neurons_post = nullptr;

  // EigenMatrix initial_weights;
  // bool initialized = false;

  virtual int add_group(RateSynapseGroup* group);

  unsigned int delay() const;
  void delay(unsigned int d);

  const EigenVector& activation() const;
  const EigenMatrix& weights() const; // just single, instantaneous dense weights for now
  void weights(const EigenMatrix& w);
  void weights_block(const EigenMatrix& w);

  std::vector<RateSynapseGroup*> synapse_groups;

  int timesteps = 0;

  int activation_buffer_interval = 0;
  EigenBuffer activation_history;

private:
  EigenMatrix _temp_weights;
  std::shared_ptr<::Backend::RateSynapses> _backend;
};

// TODO: Some nice Synapse factories for random initializations

/*
class RatePlasticity : public virtual SpikeBase {
public:
  RatePlasticity(Context* ctx, RateSynapses* syns, FloatT eps);
  ~RatePlasticity() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RatePlasticity, SpikeBase);

  void reset_state() override;

  void multipliers(EigenMatrix const& m);

  void apply_plasticity(FloatT dt);

  RateSynapses* synapses = nullptr;
  FloatT epsilon = 0;

  int timesteps = 0;

  int weights_buffer_interval = 0;
  EigenBuffer weights_history;

private:
  std::shared_ptr<::Backend::RatePlasticity> _backend;
};
*/


// TODO: Various RatePlasiticity specialisations for different learning rules

/*
class HebbPlasticity : public virtual RatePlasticity {
public:
  RatePlasticity(Context* ctx, RateSynapses* syns);
  ~RatePlasticity() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RatePlasticity, SpikeBase);

private:
  std::shared_ptr<::Backend::HebbPlasticity> _backend;
};
*/


class RateElectrodes /* : public virtual SpikeBase */ {
  friend class RateModel;

public:
  RateElectrodes(/*Context* ctx,*/ std::string prefix, RateNeurons* neurons_);
  ~RateElectrodes() /*override*/;

  /*
  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateElectrodes, SpikeBase);
  */

  void reset_state() /*override*/;

  std::string output_prefix;
  std::string output_dir;

  void write_output_info() const;

  RateNeurons* neurons;
  std::vector<std::unique_ptr<BufferWriter> > writers;

  void start() const;
  void stop() const;

protected:
  void block_until_empty() const;

/*
private:
  std::shared_ptr<::Backend::RateElectrodes> _backend;
*/
};

class RateModel /* : public virtual SpikeBase */ {
public:
  RateModel(Context* ctx=nullptr);
  ~RateModel() /*override*/;

  Context* context = nullptr;

  // void init_backend() /*override*/;
  /*
  SPIKE_ADD_BACKEND_GETSET(RateModel, SpikeBase);
  */

  void reset_state() /*override*/;

  FloatT t = 0;
  FloatT dt = 0;
  FloatT t_stop = infinity<FloatT>();

  int timesteps = 0;

  std::vector<RateNeurons*> neuron_groups;
  std::vector<RateElectrodes*> electrodes;

  void add(RateNeurons* neurons);
  void add(RateElectrodes* elecs);

  int rate_buffer_interval = 0;
  int activation_buffer_interval = 0;
  int weights_buffer_interval = 0;

  void set_rate_buffer_interval(int n_timesteps);
  void set_activation_buffer_interval(int n_timesteps);
  void set_weights_buffer_interval(int n_timesteps);
  void set_buffer_intervals(int rate_timesteps, int activation_timesteps,
                            int weights_timesteps);
  void set_buffer_intervals(int n_timesteps);
  void set_buffer_intervals(FloatT intval_s);

  bool* dump_trigger = nullptr; // used for signal handling
  bool* stop_trigger = nullptr; // used for signal handling
  void set_dump_trigger(bool* trigger);
  void set_stop_trigger(bool* trigger);

  std::thread simulation_thread;
  void simulation_loop();
  void update_model_per_dt();

  void set_simulation_time(FloatT t_stop_, FloatT dt_);

  void start(bool block=true);
  void stop();
  void wait_for_simulation();

private:
  int timesteps_per_second = 0;
  bool running = false;

  void stop_electrodes() const;
  void wait_for_electrodes() const;
  /*
private:
  std::shared_ptr<::Backend::RateModel> _backend;
  */
};
