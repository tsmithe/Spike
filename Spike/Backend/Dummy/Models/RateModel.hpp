#pragma once

#include "Spike/Models/RateModel.hpp"

namespace Backend {
  namespace Dummy {
    class RateNeurons : public virtual ::Backend::RateNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateNeurons);
      ~RateNeurons() override = default;

      void prepare() override {
      }

      void reset_state() override {
      }

      void push_data_front() override {
      }

      void pull_data_back() override {
      }
    };

    class RateSynapses : public virtual ::Backend::RateSynapses {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateSynapses);
      ~RateSynapses() override = default;

      void prepare() override {
      }

      void reset_state() override {
      }

      void push_data_front() override {
      }

      void pull_data_back() override {
      }
    };

    class RatePlasticity : public virtual ::Backend::RatePlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RatePlasticity);
      ~RatePlasticity() override = default;

      void prepare() override {
      }

      void reset_state() override {
      }

      void push_data_front() override {
      }

      void pull_data_back() override {
      }
    };

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

