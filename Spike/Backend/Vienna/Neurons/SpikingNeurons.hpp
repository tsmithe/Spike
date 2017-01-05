#pragma once

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Neurons.hpp"

namespace Backend {
  namespace Vienna {
    class SpikingNeurons : public virtual ::Backend::Vienna::Neurons,
                           public virtual ::Backend::SpikingNeurons {
    public:
      void prepare() override {
        Neurons::prepare();
      }

      void reset_state() override {
        Neurons::reset_state();
      }

      void push_data_front() override {
        Neurons::push_data_front();
      }

      void pull_data_back() override {
        Neurons::pull_data_back();
      }

      void check_for_neuron_spikes(float current_time_in_seconds, float timestep) override {
      }

      void update_membrane_potentials(float timestep, float current_time_in_seconds) override {
      }
    };
  } // namespace Vienna
} // namespace Backend
