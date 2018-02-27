#pragma once

#include "Spike/Models/RateBase.hpp"

#include <list>
#include <memory>
#include <queue>
#include <utility>

// Forward definitions:
// |---
class AgentBase;

class WorldBase;
class OpenWorld;
class MazeWorld;

template<typename WorldT>
class WorldAgentBase;

template<typename TrainPolicyT, typename TestPolicyT, typename WorldT=OpenWorld>
class Agent;

class RateNeurons;
class DummyRateNeurons;
class InputDummyRateNeurons;
class RandomDummyRateNeurons;
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
}

static_assert(std::has_virtual_destructor<Backend::RateNeurons>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::RateSynapses>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::RatePlasticity>::value,
              "contract violated");

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

/*
  void start() const;
  void stop() const;

protected:
  void block_until_empty() const;
*/
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

  AgentBase* agent = nullptr;

  std::vector<RateNeurons*> neuron_groups;
  std::vector<RateElectrodes*> electrodes;

  void add(RateNeurons* neurons);
  void add(RateElectrodes* elecs);
  void add(AgentBase* w);

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

  /*
  void stop_electrodes() const;
  void wait_for_electrodes() const;
  */
};
