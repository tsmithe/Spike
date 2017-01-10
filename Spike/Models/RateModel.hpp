#pragma once

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

template<typename T>
inline T infinity() { return std::numeric_limits<T>::infinity(); }

#include <Eigen/Dense>

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

} // namespace Eigen

class RateNeurons;    // Forward definition
class RateSynapses;   // Forward definition
class RatePlasticity; // Forward definition
class RateElectrodes; // Forward definition
// class RateModel;      // Forward definition

namespace Backend {
  class RateNeurons : public virtual SpikeBackendBase {
  public:
    ~RateNeurons() override = default;
    SPIKE_ADD_FRONTEND_GETTER(RateNeurons);
    virtual void update_rate(float dt) = 0;
  };

  class RateSynapses : public virtual SpikeBackendBase {
  public:
    ~RateSynapses() override = default;
    SPIKE_ADD_FRONTEND_GETTER(RateSynapses);
    virtual void update_activation(float dt) = 0;
  };

  class RatePlasticity : public virtual SpikeBackendBase {
  public:
    ~RatePlasticity() override = default;
    SPIKE_ADD_FRONTEND_GETTER(RatePlasticity);
    virtual void apply_plasticity(float dt) = 0;
  };

  class RateElectrodes : public virtual SpikeBackendBase {
  public:
    ~RateElectrodes() override = default;
    SPIKE_ADD_FRONTEND_GETTER(RateElectrodes);
  };

  /*
  class RateModel : public virtual SpikeBackendBase {
  public:
    ~RateModel() override = default;
    SPIKE_ADD_FRONTEND_GETTER(RateModel);
  };
  */
}

static_assert(std::has_virtual_destructor<Backend::RateNeurons>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::RateSynapses>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::RatePlasticity>::value,
              "contract violated");
/*
static_assert(std::has_virtual_destructor<Backend::RateModel>::value,
              "contract violated");
*/

#include "Spike/Backend/Dummy/Models/RateModel.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Models/RateModel.hpp"
#endif
#ifdef SPIKE_WITH_VIENNACL
#include "Spike/Backend/Vienna/Models/RateModel.hpp"
#endif

struct EigenBuffer {
  std::vector<std::pair<int, Eigen::MatrixXf> > buf;
  std::mutex lock;
};

class BufferWriter {
public:
  BufferWriter(const std::string& filename_, EigenBuffer& buf_);
  ~BufferWriter();

  void write_buffer();
  void write_loop();

  void start();
  void stop();

  EigenBuffer& buffer;
  std::string filename;
  std::ofstream file;
  std::thread othread;
  bool running = false;
};

class RateNeurons : public virtual SpikeBase {
public:
  RateNeurons(Context* ctx, int size_);
  ~RateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateNeurons, SpikeBase);

  void reset_state() override;

  void assert_dendritic_consistency() const;
  void connect_input(RateSynapses* synapses,
                     RatePlasticity* plasticity=nullptr);
  void update(float dt);

  std::string label;

  int size = 0;
  int timesteps = 0;
  Eigen::VectorXf rates;

  int rates_buffer_interval = 0;
  EigenBuffer rates_history;

  std::vector<std::pair<RateSynapses*, RatePlasticity*> > dendrites;

protected:
  void update_rate(float dt);
  void update_dendritic_activation(float dt);
  void apply_plasticity(float dt);

private:
  std::shared_ptr<::Backend::RateNeurons> _backend;
};

class RateSynapses : public virtual SpikeBase {
public:
  RateSynapses(Context* ctx,
               RateNeurons* neurons_pre_, RateNeurons* neurons_post_);
  ~RateSynapses() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateSynapses, SpikeBase);

  void reset_state() override;

  void update_activation(float dt);

  RateNeurons* neurons_pre = nullptr;
  RateNeurons* neurons_post = nullptr;

  Eigen::VectorXf activation;
  Eigen::MatrixXf weights; // just single, instantaneous dense weights for now

  int timesteps = 0;
  int activation_buffer_interval = 0;
  int weights_buffer_interval = 0;

  EigenBuffer activation_history;
  EigenBuffer weights_history;

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

  void apply_plasticity(float dt);

  RateSynapses* synapses = nullptr;

private:
  std::shared_ptr<::Backend::RatePlasticity> _backend;
};

class RateElectrodes : public virtual SpikeBase {
public:
  RateElectrodes(Context* ctx, std::string prefix, RateNeurons* neurons_);
  ~RateElectrodes() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateElectrodes, SpikeBase);

  void reset_state() override;

  std::string output_prefix;

  RateNeurons* neurons;
  std::vector<std::unique_ptr<BufferWriter> > writers;

  void start();
  void stop();

private:
  std::shared_ptr<::Backend::RateElectrodes> _backend;
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

  float t = 0;
  float dt = 0;
  float t_stop = infinity<float>();

  std::vector<RateNeurons*> neuron_groups;
  std::vector<RateElectrodes*> electrodes;

  bool* dump_trigger = nullptr; // used for signal handling
  bool* stop_trigger = nullptr; // used for signal handling

  std::thread simulation_thread;
  void simulation_loop();
  void update_model_per_dt();

  void start();
  void stop();

  /*
private:
  std::shared_ptr<::Backend::RateModel> _backend;
  */
};
