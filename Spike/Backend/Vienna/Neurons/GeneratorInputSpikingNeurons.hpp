#pragma once

#include "Spike/Neurons/GeneratorInputSpikingNeurons.hpp"
#include "InputSpikingNeurons.hpp"

namespace Backend {
  namespace Vienna {
    class GeneratorInputSpikingNeurons : public virtual ::Backend::Vienna::InputSpikingNeurons,
                                         public virtual ::Backend::GeneratorInputSpikingNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(GeneratorInputSpikingNeurons);

      void prepare() override {
        InputSpikingNeurons::prepare();
      }

      void reset_state() override {
        InputSpikingNeurons::reset_state();
      }

      void push_data_front() override {
        InputSpikingNeurons::push_data_front();
      }

      void pull_data_back() override {
        InputSpikingNeurons::pull_data_back();
      }

      // May want to override these when writing a new backend, or may not:
      using ::Backend::Vienna::SpikingNeurons::check_for_neuron_spikes;
      using ::Backend::Vienna::SpikingNeurons::update_membrane_potentials;
    };
  }
}