#pragma once

#include "Spike/Models/RateModel.hpp"

namespace Backend {
  namespace Dummy {
    class RateModel : public virtual ::Backend::RateModel {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateModel);
      ~RateModel() override = default;

      void prepare() override {
      }

      void reset_state() override {
      }

      void push_data_front() override {
      }

      void pull_data_back() override {
      }
    };
  } // namespace Dummy
} // namespace Backend

