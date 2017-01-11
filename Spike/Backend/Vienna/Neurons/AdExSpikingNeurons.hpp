#pragma once

#include "Spike/Neurons/AdExSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

namespace Backend {
  namespace Vienna {
    class AdExSpikingNeurons : public virtual ::Backend::Vienna::SpikingNeurons,
                               public virtual ::Backend::AdExSpikingNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(AdExSpikingNeurons);

      void prepare() override {
        SpikingNeurons::prepare();
      }

      void reset_state() override {
        SpikingNeurons::reset_state();
      }

      void push_data_front() override {
        SpikingNeurons::push_data_front();
      }

      void pull_data_back() override {
        SpikingNeurons::pull_data_back();
      }

      // May want to override these when writing a new backend, or may not:
      using ::Backend::Vienna::SpikingNeurons::check_for_neuron_spikes;
      using ::Backend::Vienna::SpikingNeurons::update_membrane_potentials;
    };
  }
}