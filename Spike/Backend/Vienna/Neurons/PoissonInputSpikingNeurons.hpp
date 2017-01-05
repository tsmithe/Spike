#pragma once

#include "Spike/Neurons/PoissonInputSpikingNeurons.hpp"
#include "InputSpikingNeurons.hpp"

namespace Backend {
  namespace Vienna {
    class PoissonInputSpikingNeurons : public virtual ::Backend::Vienna::InputSpikingNeurons,
                                       public virtual ::Backend::PoissonInputSpikingNeurons {
    public:
      PoissonInputSpikingNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(PoissonInputSpikingNeurons);

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
    };
  }
}
