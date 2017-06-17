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
#include <Eigen/Sparse>

typedef float FloatT;
typedef Eigen::Matrix<FloatT, Eigen::Dynamic, 1> EigenVector;
typedef Eigen::Matrix<FloatT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrix;
typedef Eigen::SparseMatrix<FloatT, Eigen::RowMajor> EigenSpMatrix;

inline void normalize_matrix_rows(EigenMatrix& R, FloatT scale=1) {
  #pragma omp parallel for
  for (int i = 0; i < R.rows(); ++i) {
    FloatT row_norm = R.row(i).norm();
    if (row_norm > 0)
      R.row(i) /= scale*row_norm;
  }
}

inline void normalize_matrix_rows(EigenSpMatrix& R, FloatT scale=1) {
  assert(R.rows() == R.outerSize());
  #pragma omp parallel for
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

// Forward definitions:
// |---
class RateNeurons;
class DummyRateNeurons;
class InputDummyRateNeurons;
class RandomDummyRateNeurons;
class AgentVISRateNeurons;
class AgentAHVRateNeurons;
class AgentFVRateNeurons;
class RateSynapses;
class RatePlasticity;
class BCMPlasticity;
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

    virtual const EigenVector& activation() = 0;
    virtual void get_weights(EigenMatrix& output) = 0;
    virtual void weights(EigenMatrix const& w) = 0;

    virtual void make_sparse() = 0;

    virtual void delay(unsigned int) = 0;
    virtual unsigned int delay() = 0;
  };

  class RatePlasticity : public virtual SpikeBackendBase {
  public:
    ~RatePlasticity() override = default;
    SPIKE_ADD_BACKEND_FACTORY(RatePlasticity);
    void prepare() override = 0;
    void reset_state() override = 0;

    virtual void apply_plasticity(FloatT dt) = 0;
  };

  class BCMPlasticity : public virtual RatePlasticity {
  public:
    ~BCMPlasticity() override = default;
    SPIKE_ADD_BACKEND_FACTORY(BCMPlasticity);
    void prepare() override = 0;
    void reset_state() override = 0;

    void apply_plasticity(FloatT dt) override = 0;
  };

  class RateNeurons : public virtual SpikeBackendBase {
  public:
    ~RateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(RateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    virtual void connect_input(RateSynapses* synapses,
                               RatePlasticity* plasticity) = 0;
    virtual bool staged_integrate_timestep(FloatT dt) = 0;
    virtual const EigenVector& rate() = 0;
  };

  class DummyRateNeurons : public virtual RateNeurons {
  public:
    ~DummyRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(DummyRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    void connect_input(RateSynapses* synapses,
                       RatePlasticity* plasticity) override = 0;
    virtual void add_schedule(FloatT duration, EigenVector rates) = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    const EigenVector& rate() override = 0;
  };

  class InputDummyRateNeurons : public virtual DummyRateNeurons {
  public:
    ~InputDummyRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(InputDummyRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    const EigenVector& rate() override = 0;
  };

  class RandomDummyRateNeurons : public virtual DummyRateNeurons {
  public:
    ~RandomDummyRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(RandomDummyRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    const EigenVector& rate() override = 0;
  };

  class AgentVISRateNeurons : public virtual RateNeurons {
  public:
    ~AgentVISRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(AgentVISRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    void connect_input(RateSynapses* synapses,
                       RatePlasticity* plasticity) override = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    const EigenVector& rate() override = 0;
  };

  class AgentAHVRateNeurons : public virtual RateNeurons {
  public:
    ~AgentAHVRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(AgentAHVRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    void connect_input(RateSynapses* synapses,
                       RatePlasticity* plasticity) override = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    const EigenVector& rate() override = 0;
  };

  class AgentFVRateNeurons : public virtual RateNeurons {
  public:
    ~AgentFVRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(AgentFVRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    void connect_input(RateSynapses* synapses,
                       RatePlasticity* plasticity) override = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    const EigenVector& rate() override = 0;
  };
}

static_assert(std::has_virtual_destructor<Backend::RateNeurons>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::RateSynapses>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::RatePlasticity>::value,
              "contract violated");

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
  FloatT time_since_last_flush = 0;
  std::thread othread; // TODO: Perhaps having too many
                       //       output threads will cause too much
                       //       seeking on disk, thus slowing things down?
                       // Perhaps better just to have one global thread?
                       // Or one thread per Electrodes?
private:
  bool running = false;
};

/*
class Agent {
public:
  Agent();
  Agent(FloatT bound_x_, FloatT bound_y_, FloatT velocity_scaling_);

  void connect_actor(RateNeurons* actor_);
  void update_per_dt(FloatT dt);

  Eigen::Matrix<FloatT, 2, 1> position;
  FloatT bound_x = 10, bound_y = 10;
  FloatT velocity_scaling = 1;

private:
  RateNeurons* actor;
  Eigen::Matrix<FloatT, 2, Eigen::Dynamic> actor_tuning;
};
*/

class Agent {
public:
  Agent();
  //Agent(FloatT bound_x_, FloatT bound_y_, FloatT velocity_scaling_);

  void set_boundary(FloatT bound_x, FloatT bound_y);

  void add_proximal_object(FloatT x, FloatT y);
  void add_distal_object(FloatT angle);

  void add_FV(FloatT FV, FloatT duration);
  void add_AHV(FloatT AHV, FloatT duration);

  // void connect_actor(RateNeurons* actor_);
  void update_per_dt(FloatT dt);

  void record_history(std::string output_prefix,
                      int buffer_interval, int buffer_start);
  void save_map(std::string output_prefix);

  void seed(unsigned s);

  // FloatT velocity_scaling = 1;

  FloatT bound_x = 10, bound_y = 10;

  typedef Eigen::Matrix<FloatT, 2, 1> EigenVector2D;
  typedef Eigen::Matrix<FloatT, 2, Eigen::Dynamic> EigenMatrix2D;

  EigenVector2D position;
  FloatT head_direction = 0;

  int num_objects = 0;
  int num_proximal_objects = 0;
  int num_distal_objects = 0;

  EigenVector object_bearings;

  int num_AHV_states = 0;
  int num_FV_states = 0;

  int curr_AHV = 0;
  int curr_FV = 0;

  enum actions { AHV, FV };
  FloatT p_fwd = 0.5;
  int curr_action = FV;
  int choose_next_action_ts = 0;
  EigenVector2D target_position;
  FloatT target_head_direction = 0;

  FloatT t = 0;
  int timesteps = 0;

  int agent_buffer_interval = 0;
  int agent_buffer_start = 0;
  EigenBuffer agent_history;
  std::unique_ptr<BufferWriter> history_writer;

private:
  // RateNeurons* actor;
  // EigenMatrix2D actor_tuning;

  EigenMatrix2D proximal_objects;
  std::vector<FloatT> distal_objects;

  std::vector<std::pair<FloatT, FloatT> > FVs;  //  FV, duration
  std::vector<std::pair<FloatT, FloatT> > AHVs; // AHV, duration

  std::default_random_engine rand_engine;

  std::uniform_real_distribution<FloatT> action_die;
  std::uniform_int_distribution<> AHV_die;
  std::uniform_int_distribution<> FV_die;
};

class RateNeurons : public virtual SpikeBase {
public:
  RateNeurons(Context* ctx, int size_, std::string label_,
              FloatT alpha=0.0, FloatT beta=1.0, FloatT tau=1.0);
  ~RateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateNeurons, SpikeBase);

  void reset_state() override;

  void assert_dendritic_consistency(RateSynapses* synapses,
                                    RatePlasticity* plasticity) const;
  void assert_dendritic_consistency() const;
  void connect_input(RateSynapses* synapses,
                     RatePlasticity* plasticity=nullptr);

  bool staged_integrate_timestep(FloatT dt);
  void apply_plasticity(FloatT dt) const;

  int size = 0;

  std::string label;

  FloatT alpha = 0;
  FloatT beta = 1.0;
  FloatT tau = 1.0;

  int timesteps = 0;

  const EigenVector& rate() const;
  int rate_buffer_interval = 0;
  int rate_buffer_start = 0;
  EigenBuffer rate_history;

  std::vector<std::pair<RateSynapses*, RatePlasticity*> > dendrites;

private:
  std::shared_ptr<::Backend::RateNeurons> _backend;
};

class DummyRateNeurons : public virtual RateNeurons {
public:
  DummyRateNeurons(Context* ctx, int size_, std::string label_);
  ~DummyRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(DummyRateNeurons, RateNeurons);

  std::vector<std::pair<FloatT, EigenVector> > rate_schedule;
  void add_schedule(FloatT duration, EigenVector const& rates);
  
private:
  std::shared_ptr<::Backend::DummyRateNeurons> _backend;
};

class InputDummyRateNeurons : public virtual DummyRateNeurons {
public:
  InputDummyRateNeurons(Context* ctx, int size_, std::string label_,
                        FloatT sigma_IN_, FloatT lambda_);
  ~InputDummyRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(InputDummyRateNeurons, DummyRateNeurons);

  FloatT sigma_IN;
  FloatT lambda;
  FloatT t_stop_after = infinity<FloatT>();

  EigenVector theta_pref;

  std::vector<std::pair<FloatT, FloatT> > revs_schedule;
  void add_schedule(FloatT duration, FloatT revs_per_sec);

private:
  std::shared_ptr<::Backend::InputDummyRateNeurons> _backend;
};

// TODO: Generalize / parameterize RandomDummyRateNeurons
class RandomDummyRateNeurons : public virtual DummyRateNeurons {
public:
  RandomDummyRateNeurons(Context* ctx, int size_, std::string label_);
  ~RandomDummyRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RandomDummyRateNeurons, DummyRateNeurons);

  FloatT t_stop_after = infinity<FloatT>();

  // std::vector<std::pair<FloatT, FloatT> > revs_schedule;
  // void add_schedule(FloatT duration, FloatT revs_per_sec);

private:
  std::shared_ptr<::Backend::RandomDummyRateNeurons> _backend;
};

/*
class AgentSenseRateNeurons : public virtual RateNeurons {
public:
  AgentSenseRateNeurons(Context* ctx, Agent* agent_, std::string label_);
  ~AgentSenseRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(AgentSenseRateNeurons, RateNeurons);

  Agent* agent;

private:
  std::shared_ptr<::Backend::AgentSenseRateNeurons> _backend;
};
*/

class AgentVISRateNeurons : public virtual RateNeurons {
public:
  AgentVISRateNeurons(Context* ctx, Agent* agent_,
                      int neurons_per_object_,
                      FloatT sigma_IN_, FloatT lambda_,
                      std::string label_);
  ~AgentVISRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(AgentVISRateNeurons, RateNeurons);

  Agent* agent;
  int neurons_per_object;
  FloatT sigma_IN, lambda;

  EigenVector theta_pref;

  FloatT t_stop_after = infinity<FloatT>();

private:
  std::shared_ptr<::Backend::AgentVISRateNeurons> _backend;
};

class AgentAHVRateNeurons : public virtual RateNeurons {
public:
  AgentAHVRateNeurons(Context* ctx, Agent* agent_,
                      int neurons_per_state_, std::string label_);
  ~AgentAHVRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(AgentAHVRateNeurons, RateNeurons);

  Agent* agent;
  int neurons_per_state;

private:
  std::shared_ptr<::Backend::AgentAHVRateNeurons> _backend;
};

class AgentFVRateNeurons : public virtual RateNeurons {
public:
  AgentFVRateNeurons(Context* ctx, Agent* agent_,
                     int neurons_per_state_, std::string label_);
  ~AgentFVRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(AgentFVRateNeurons, RateNeurons);

  Agent* agent;
  int neurons_per_state;

private:
  std::shared_ptr<::Backend::AgentFVRateNeurons> _backend;
};

class RateSynapses : public virtual SpikeBase {
public:
  RateSynapses(Context* ctx,
               RateNeurons* neurons_pre_, RateNeurons* neurons_post_,
               FloatT scaling_=1, std::string label_="");
  ~RateSynapses() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateSynapses, SpikeBase);

  void reset_state() override;

  RateNeurons* neurons_pre = nullptr;
  RateNeurons* neurons_post = nullptr;

  std::string label;

  unsigned int delay() const;
  void delay(unsigned int d);

  const EigenVector& activation() const;
  void get_weights(EigenMatrix& output) const; // just single, instantaneous dense weights for now
  void weights(const EigenMatrix& w);

  void make_sparse();

  FloatT scaling = 1;

  int timesteps = 0;

  int activation_buffer_interval = 0;
  int activation_buffer_start = 0;
  EigenBuffer activation_history;

private:
  std::shared_ptr<::Backend::RateSynapses> _backend;
};

class RatePlasticity : public virtual SpikeBase {
public:
  RatePlasticity(Context* ctx, RateSynapses* syns);
  ~RatePlasticity() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RatePlasticity, SpikeBase);

  void reset_state() override;

  std::vector<std::pair<FloatT, FloatT> > plasticity_schedule;
  void add_schedule(FloatT duration, FloatT eps);

  virtual void apply_plasticity(FloatT dt);

  RateSynapses* synapses = nullptr;

  int timesteps = 0;

  int weights_buffer_interval = 0;
  int weights_buffer_start = 0;
  EigenBuffer weights_history;

private:
  std::shared_ptr<::Backend::RatePlasticity> _backend;
};

class BCMPlasticity : public virtual RatePlasticity {
public:
  BCMPlasticity(Context* ctx, RateSynapses* syns);
  ~BCMPlasticity() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(BCMPlasticity, RatePlasticity);

  // void apply_plasticity(FloatT dt) override;

private:
  std::shared_ptr<::Backend::BCMPlasticity> _backend;
};

class RateElectrodes {
  friend class RateModel;

public:
  RateElectrodes(std::string prefix, RateNeurons* neurons_);
  ~RateElectrodes();

  std::string output_prefix;
  std::string output_dir;

  void write_output_info() const;

  RateNeurons* neurons;
  std::vector<std::unique_ptr<BufferWriter> > writers;

  void start() const;
  void stop() const;

protected:
  void block_until_empty() const;
};

class RateModel {
public:
  RateModel(Context* ctx=nullptr);
  ~RateModel();

  Context* context = nullptr;

  void reset_state();

  FloatT t = 0;
  FloatT dt = 0;
  FloatT t_stop = infinity<FloatT>();

  int timesteps = 0;

  Agent* agent = nullptr;

  std::vector<RateNeurons*> neuron_groups;
  std::vector<RateElectrodes*> electrodes;

  void add(RateNeurons* neurons);
  void add(RateElectrodes* elecs);
  void add(Agent* w);

  int rate_buffer_interval = 0;
  int activation_buffer_interval = 0;
  int weights_buffer_interval = 0;

  int rate_buffer_start = 0;
  int activation_buffer_start = 0;
  int weights_buffer_start = 0;

  void set_rate_buffer_interval(int n_timesteps);
  void set_activation_buffer_interval(int n_timesteps);
  void set_weights_buffer_interval(int n_timesteps);
  void set_buffer_intervals(int rate_timesteps, int activation_timesteps,
                            int weights_timesteps);
  void set_buffer_intervals(int n_timesteps);
  void set_buffer_intervals(FloatT intval_s);

  void set_rate_buffer_start(int n_timesteps);
  void set_activation_buffer_start(int n_timesteps);
  void set_weights_buffer_start(int n_timesteps);
  void set_buffer_start(FloatT start_t);

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
};
