#pragma once

#include "Spike/Neurons/InputSpikingNeurons.hpp"
#include "SpikingNeurons.hpp"

namespace Backend {
  namespace Vienna {
    class InputSpikingNeurons : public virtual ::Backend::Vienna::SpikingNeurons,
                                public virtual ::Backend::InputSpikingNeurons {
    public:
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
    };
  }
}
