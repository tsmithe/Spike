#pragma once

#include "Spike/Models/RateModel.hpp"

namespace Backend {
  namespace Dummy {
    class RateSynapses : public virtual ::Backend::RateSynapses {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateSynapses);
      ~RateSynapses() override = default;

      void prepare() override {
        _weights = EigenMatrix::Zero(frontend()->neurons->size, frontend()->neurons->size);
      }

      void reset_state() override {
      }

      // void update_activation(float dt) override {
      // }

      /*
      inline const Eigen::VectorXf& activation() override {
        return _activation;
      }

      inline const Eigen::MatrixXf& weights() override {
        return _weights;
      }
      */

      const EigenVector& activation() override {
        return _activation;
      }

      const EigenMatrix& weights() override {
        return _weights;
      }

      void weights(EigenMatrix const& w) override {
      }

      void delay(unsigned int) override {
      }

      unsigned int delay() override {
        return 0;
      }

    private:
      EigenVector _activation;
      EigenMatrix _weights;
    };

    /*
    class RatePlasticity : public virtual ::Backend::RatePlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RatePlasticity);
      ~RatePlasticity() override = default;

      void prepare() override {
      }

      void reset_state() override {
      }

      void apply_plasticity(float dt) override {
      }
    };
    */

    class RateNeurons : public virtual ::Backend::RateNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateNeurons);
      ~RateNeurons() override = default;

      void prepare() override {
      }

      void reset_state() override {
      }

      void connect_input(::Backend::RateSynapses* synapses/*,
                         ::Backend::RatePlasticity* plasticity*/) override {
      }

      bool staged_integrate_timestep(FloatT dt) override {
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
  } // namespace Dummy
} // namespace Backend

