#pragma once

#include "Spike/Models/RateModel.hpp"

namespace Backend {
  namespace CUDA {
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

      void update_activation(float dt) override {
      }

      inline const Eigen::VectorXf& activation() override {
        return _activation;
      }

      inline const Eigen::MatrixXf& weights() override {
        return _weights;
      }

    private:
      Eigen::VectorXf _activation;
      Eigen::MatrixXf _weights;
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

      void apply_plasticity(float dt) override {
      }
    };

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

      void connect_input(::Backend::RateSynapses* synapses,
                         ::Backend::RatePlasticity* plasticity) override {
      }

      void update_rate(float dt) override {
      }

      inline const Eigen::VectorXf& rate() override {
        return _rate;
      }

    private:
      Eigen::VectorXf _rate;
    };

    /*
    class RateElectrodes : public virtual ::Backend::RateElectrodes {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateElectrodes);
      ~RateElectrodes() override = default;

      void prepare() override {
      }

      void reset_state() override {
      }

      void push_data_front() override {
      }

      void pull_data_back() override {
      }
    };
    */

    /*
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
    */
  } // namespace CUDA
} // namespace Backend

