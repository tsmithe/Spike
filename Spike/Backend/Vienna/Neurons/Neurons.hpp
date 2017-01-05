#pragma once

#include "Spike/Neurons/Neurons.hpp"

namespace Backend {
  namespace Vienna {
    class Neurons : public virtual ::Backend::Neurons {
    public:
      ~Neurons() override = default;

      void prepare() override {
      }

      void reset_state() override {
        reset_current_injections();
      }

      void reset_current_injections() override {
      }

      void push_data_front() override {
      }

      void pull_data_back() override {
      }
    };
  } // namespace Vienna
} // namespace Backend

