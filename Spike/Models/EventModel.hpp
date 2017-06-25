#include "RateModel.hpp"

// Forward definitions:
class EventNeurons;

namespace Backend {
  class EventNeurons : public virtual SpikeBackendBase {
  public:
    ~EventNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(EventNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    /*
    virtual void connect_input(RateSynapses* synapses,
                               RatePlasticity* plasticity) = 0;
    virtual bool staged_integrate_timestep(FloatT dt) = 0;
    virtual const EigenVector& rate() = 0;
    */
  };
}

static_assert(std::has_virtual_destructor<Backend::EventNeurons>::value,
              "contract violated");

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

