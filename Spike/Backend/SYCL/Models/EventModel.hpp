#pragma once

#include <CL/sycl.hpp>

#include "Spike/Models/EventModel.hpp"

namespace Backend {
  namespace SYCL {
    using namespace cl::sycl;

    // Forward definitions:
    class EventModel;
    class EventNeurons;
    /*
    class EventSynapses;
    class EventPlasticity;
    */

    class EventModel : public virtual ::Backend::EventModel {
    public:
      EventModel() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(EventModel);
      ~EventModel() override = default;

      void prepare() override;
      void reset_state() override;

      queue event_queue;
    };

    class EventNeurons : public virtual ::Backend::EventNeurons {
    public:
      EventNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(EventNeurons);
      ~EventNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      buffer<float, 1> V;
      buffer<float, 1> last_input_t;
      buffer<float, 1> last_input_V;

      buffer<float, 1> weights;
      buffer<float, 1> delays;
      buffer<float, 1> synapse_pre_idx;
      buffer<float, 1> synapse_post_delim;

      // void project(::Backend::EventSynapses* synapses) override;

    private:
      // ::Backend::SYCL::EventSynapses* _synapses = nullptr;
    };

    /*
    class EventSynapses : public virtual ::Backend::EventSynapses {
    public:
      EventSynapses() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(EventSynapses);
      ~EventSynapses() override = default;

      void prepare() override;
      void reset_state() override;

      void add_plasticity(::Backend::EventPlasticity* plasticity) override;

    private:
      ::Backend::SYCL::EventPlasticity* _plasticity = nullptr;
    };

    class EventPlasticity : public virtual ::Backend::EventPlasticity {
    public:
      EventPlasticity() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(EventPlasticity);
      ~EventPlasticity() override = default;

      void prepare() override;
      void reset_state() override;
    };
    */
  } // namespace SYCL
} // namespace Backend

