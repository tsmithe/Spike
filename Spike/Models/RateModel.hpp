#pragma once

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include <utility>
#include <vector>

#include <Eigen/Dense>

class RateNeurons;    // Forward definition
class RateSynapses;   // Forward definition
class RatePlasticity; // Forward definition
class RateModel;      // Forward definition

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

  class RateModel : public virtual SpikeBackendBase {
  public:
    ~RateModel() override = default;
    SPIKE_ADD_FRONTEND_GETTER(RateModel);
  };
}

static_assert(std::has_virtual_destructor<Backend::RateNeurons>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::RateSynapses>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::RatePlasticity>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::RateModel>::value,
              "contract violated");

#include "Spike/Backend/Dummy/Models/RateModel.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Models/RateModel.hpp"
#endif
#ifdef SPIKE_WITH_VIENNACL
#include "Spike/Backend/Vienna/Models/RateModel.hpp"
#endif

class RateNeurons : public virtual SpikeBase {
public:
  RateNeurons();
  ~RateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateNeurons, SpikeBase);

  void reset_state() override;

  void assert_dendritic_consistency() const;
  void update(float dt);
  Eigen::VectorXf rates() const;

  int size = 0;
  int timesteps = 0;
  std::vector<std::pair<RateSynapses*, RatePlasticity*> > dendrites;

protected:
  void update_rate(float dt);
  void update_dendritic_activation(float dt);
  void apply_plasticity(float dt);

private:
  std::shared_ptr<::Backend::RateNeurons> _backend;
  Eigen::MatrixXf _rates;
};

class RateSynapses : public virtual SpikeBase {
public:
  RateSynapses();
  ~RateSynapses() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateSynapses, SpikeBase);

  void reset_state() override;

  void update_activation(float dt);

  RateNeurons* neurons_pre = nullptr;
  RateNeurons* neurons_post = nullptr;

private:
  std::shared_ptr<::Backend::RateSynapses> _backend;
  Eigen::VectorXf _activation;
  Eigen::MatrixXf _weights; // just single, instantaneous dense weights for now
};

class RatePlasticity : public virtual SpikeBase {
public:
  RatePlasticity();
  ~RatePlasticity() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RatePlasticity, SpikeBase);

  void reset_state() override;

  void apply_plasticity(float dt);

  RateSynapses* synapses = nullptr;

private:
  std::shared_ptr<::Backend::RatePlasticity> _backend;
};

class RateModel : public virtual SpikeBase {
public:
  RateModel();
  ~RateModel() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateModel, SpikeBase);

  void reset_state() override;

  std::vector<RateNeurons*> neuron_groups;

private:
  std::shared_ptr<::Backend::RateModel> _backend;
};
