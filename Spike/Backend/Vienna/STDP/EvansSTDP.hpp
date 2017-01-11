#pragma once

#include "Spike/STDP/EvansSTDP.hpp"
#include "STDP.hpp"

namespace Backend {
  namespace Vienna {
    class EvansSTDP : public virtual ::Backend::Vienna::STDP,
                      public virtual ::Backend::EvansSTDP {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(EvansSTDP);

      void prepare() override {
        STDP::prepare();
      }

      void reset_state() override {
        STDP::reset_state();
      }

      void push_data_front() override {
        STDP::push_data_front();
      }

      void pull_data_back() override {
        STDP::pull_data_back();
      }

      void update_synaptic_efficacies_or_weights(float current_time_in_seconds) override {
      }

      void update_presynaptic_activities(float timestep, float current_time_in_seconds) override {
      }

      void update_postsynaptic_activities(float timestep, float current_time_in_seconds) override {
      }
    };
  }
}