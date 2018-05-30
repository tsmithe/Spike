#pragma once

#include "Spike/Models/RateAgent.hpp"
#include "Spike/Backend/Eigen/Models/RateModel.hpp"

namespace Backend {
  namespace Eigen {
    class AgentVISRateNeurons
      : public virtual ::Backend::AgentVISRateNeurons,
        public virtual ::Backend::Eigen::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
      friend class BCMPlasticity;
    public:
      // AgentVISRateNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(AgentVISRateNeurons);
      ~AgentVISRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses*,
                         ::Backend::RatePlasticity*) override;

      bool staged_integrate_timestep(FloatT dt) override;
      EigenVector const& rate() override;
      EigenVector const& rate(unsigned int n_back=0) override;

      FloatT mean_rate() override {
        return rate(0).norm();
      }

      FloatT mean_rate(unsigned int) override {
        assert("Not allowed" && false);
      }

    protected:
      FloatT t = 0, dt_ = 0;

      EigenVector _rate;

      EigenVector sigma_IN_sqr;

      EigenVector theta_pref;
      EigenVector d;
    };

    class AgentHDRateNeurons
      : public virtual ::Backend::AgentHDRateNeurons,
        public virtual ::Backend::Eigen::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
      friend class BCMPlasticity;
    public:
      // AgentHDRateNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(AgentHDRateNeurons);
      ~AgentHDRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses*,
                         ::Backend::RatePlasticity*) override;

      bool staged_integrate_timestep(FloatT dt) override;
      EigenVector const& rate() override;
      EigenVector const& rate(unsigned int n_back=0) override;

    protected:
      FloatT t = 0, dt_ = 0;

      EigenVector _rate;

      EigenVector sigma_IN_sqr;

      EigenVector theta_pref;
      EigenVector d;
    };

    class AgentAHVRateNeurons
      : public virtual ::Backend::AgentAHVRateNeurons,
        public virtual ::Backend::Eigen::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
      friend class BCMPlasticity;
    public:
      // AgentAHVRateNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(AgentAHVRateNeurons);
      ~AgentAHVRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses*,
                         ::Backend::RatePlasticity*) override;

      bool staged_integrate_timestep(FloatT dt) override;
      EigenVector const& rate() override;
      EigenVector const& rate(unsigned int n_back=0) override;

    protected:
      EigenVector _rate;
      int curr_AHV = -2;
      FloatT AHV = 0;

      //::Eigen::Matrix<FloatT, Eigen::Dynamic, 2> _tuning;
    };

    class AgentFVRateNeurons
      : public virtual ::Backend::AgentFVRateNeurons,
        public virtual ::Backend::Eigen::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
      friend class BCMPlasticity;
    public:
      // AgentFVRateNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(AgentFVRateNeurons);
      ~AgentFVRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses*,
                         ::Backend::RatePlasticity*) override;

      bool staged_integrate_timestep(FloatT dt) override;
      EigenVector const& rate() override;
      EigenVector const& rate(unsigned int n_back=0) override;

    protected:
      EigenVector _rate;
      int curr_FV = -2;
      //::Eigen::Matrix<FloatT, Eigen::Dynamic, 2> _tuning;
    };
  } // Eigen
} // Backend
