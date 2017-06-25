#include "RateModel.hpp"

// Forward definitions:
class EventModel;
class EventNeurons;
/*
class EventSynapses;
class EventPlasticity;
*/

namespace Backend {
  class EventModel : public virtual SpikeBackendBase {
  public:
    ~EventModel() override = default;
    SPIKE_ADD_BACKEND_FACTORY(EventModel);
    void prepare() override = 0;
    void reset_state() override = 0;
  };

  class EventNeurons : public virtual SpikeBackendBase {
  public:
    ~EventNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(EventNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    // virtual void project(::Backend::EventSynapses* synapses) = 0;
  };

  /*
  class EventSynapses : public virtual SpikeBackendBase {
  public:
    ~EventSynapses() override = default;
    SPIKE_ADD_BACKEND_FACTORY(EventSynapses);
    void prepare() override = 0;
    void reset_state() override = 0;

    virtual void add_plasticity(::Backend::EventPlasticity* plasticity) = 0;
  };

  class EventPlasticity : public virtual SpikeBackendBase {
  public:
    ~EventPlasticity() override = default;
    SPIKE_ADD_BACKEND_FACTORY(EventPlasticity);
    void prepare() override = 0;
    void reset_state() override = 0;
  };
  */
}


static_assert(std::has_virtual_destructor<Backend::EventModel>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::EventNeurons>::value,
              "contract violated");
/*
static_assert(std::has_virtual_destructor<Backend::EventSynapses>::value,
              "contract violated");
static_assert(std::has_virtual_destructor<Backend::EventPlasticity>::value,
              "contract violated");
*/


class EventModel : public virtual SpikeBase {
public:
  EventModel(Context* ctx);
  ~EventModel() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(EventModel, SpikeBase);

  void reset_state() override;

private:
  std::shared_ptr<::Backend::EventModel> _backend;
};


class EventNeurons : public virtual SpikeBase {
public:
  EventNeurons(Context* ctx, int size_, std::string label_);
  ~EventNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(EventNeurons, SpikeBase);

  void reset_state() override;

  /*
  void assert_dendritic_consistency(RateSynapses* synapses,
                                    RatePlasticity* plasticity) const;
  void assert_dendritic_consistency() const;
  void connect_input(RateSynapses* synapses,
                     RatePlasticity* plasticity=nullptr);

  bool staged_integrate_timestep(FloatT dt);
  void apply_plasticity(FloatT dt) const;
  */

  int size = 0;

  std::string label;

  /*
  FloatT alpha = 0;
  FloatT beta = 1.0;
  FloatT tau = 1.0;

  int timesteps = 0;

  const EigenVector& rate() const;
  int rate_buffer_interval = 0;
  int rate_buffer_start = 0;
  EigenBuffer rate_history;

  std::vector<std::pair<RateSynapses*, RatePlasticity*> > dendrites;
  */

private:
  std::shared_ptr<::Backend::EventNeurons> _backend;
};


/*
class EventSynapses : public virtual SpikeBase {
public:
  EventSynapses(Context* ctx);
  ~EventSynapses() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(EventSynapses, SpikeBase);

  void reset_state() override;

private:
  std::shared_ptr<::Backend::EventSynapses> _backend;
};


class EventPlasticity : public virtual SpikeBase {
public:
  EventPlasticity(Context* ctx);
  ~EventPlasticity() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(EventPlasticity, SpikeBase);

  void reset_state() override;

private:
  std::shared_ptr<::Backend::EventPlasticity> _backend;
};
*/
