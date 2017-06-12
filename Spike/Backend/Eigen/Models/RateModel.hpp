#pragma once

#include "Spike/Models/RateModel.hpp"

namespace Backend {
  namespace Eigen {
    class RateNeurons;    // forward

    class RateSynapses : public virtual ::Backend::RateSynapses {
      friend class RateNeurons;
      friend class RatePlasticity;
      friend class BCMPlasticity;
    public:
      RateSynapses() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateSynapses);
      ~RateSynapses() override = default;

      void prepare() override;
      void reset_state() override;

      // void update_activation(FloatT dt) override;
      EigenVector const& activation() override;
      void get_weights(EigenMatrix& output) override;
      void weights(EigenMatrix const& w) override;

      void make_sparse() override;

      unsigned int delay() override;
      void delay(unsigned int d) override;

    private:
      unsigned int _delay = 0;

      EigenVector _activation; // TODO: Need an explicit temporary?
      EigenMatrix _weights;    // TODO: Generalize synapse types
      EigenSpMatrix _sp_weights;
      EigenSpMatrix _sparsity;

      bool is_sparse = false;

      ::Backend::Eigen::RateNeurons* neurons_pre = nullptr;
      ::Backend::Eigen::RateNeurons* neurons_post = nullptr;
     };

    class RatePlasticity : public virtual ::Backend::RatePlasticity {
    public:
      RatePlasticity() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RatePlasticity);
      ~RatePlasticity() override = default;

      void prepare() override;
      void reset_state() override;

      void apply_plasticity(FloatT dt) override;

      //private:
      ::Backend::Eigen::RateSynapses* synapses = nullptr;
      FloatT epsilon = 0;

      int _schedule_idx = 0;
      FloatT _curr_rate_t = 0;

      EigenMatrix _multipliers;
      bool _using_multipliers = false;
    };

    class BCMPlasticity : public virtual ::Backend::BCMPlasticity,
                          public virtual ::Backend::Eigen::RatePlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(BCMPlasticity);
      ~BCMPlasticity() override = default;

      void prepare() override;
      void reset_state() override;

      void apply_plasticity(FloatT dt) override;

    private:
      EigenVector _thresh;
    };

    class RateNeurons : public virtual ::Backend::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
      friend class BCMPlasticity;
    public:
      RateNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateNeurons);
      ~RateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses* synapses,
                         ::Backend::RatePlasticity* plasticity) override;

      bool staged_integrate_timestep(FloatT dt) override;

      template<typename T>
      inline T transfer(T const& total_activation);

      virtual EigenVector const& rate(unsigned int n_back);
      EigenVector const& rate() override;

    private:
      bool done_timestep = false;

      EigenVector _total_activation;

      EigenMatrix _rate_history;
      int _rate_hist_idx = 0;

      EigenVector _new_rate;
      EigenVector _rate;
      int _rate_cpu_timestep = 0;
      std::vector<
        std::pair<::Backend::Eigen::RateSynapses*,
                  ::Backend::Eigen::RatePlasticity*> > _eigen_dendrites;
    };

    class DummyRateNeurons : public virtual ::Backend::DummyRateNeurons,
                             public virtual ::Backend::Eigen::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
      friend class BCMPlasticity;
    public:
      DummyRateNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(DummyRateNeurons);
      ~DummyRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses*,
                         ::Backend::RatePlasticity*) override;

      bool staged_integrate_timestep(FloatT dt) override;
      EigenVector const& rate() override;
      EigenVector const& rate(unsigned int n_back=0) override;

      void add_schedule(FloatT duration, EigenVector rates) override;

    protected:
      FloatT t, dt_;

      int _schedule_idx = 0;
      FloatT _curr_rate_t;
    };

    class InputDummyRateNeurons
      : public virtual ::Backend::InputDummyRateNeurons,
        protected virtual ::Backend::Eigen::DummyRateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
      friend class BCMPlasticity;
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(InputDummyRateNeurons);
      ~InputDummyRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      bool staged_integrate_timestep(FloatT dt) override;
      EigenVector const& rate() override;
      EigenVector const& rate(unsigned int n_back=0) override;

    private:
      EigenVector _rate;

      FloatT theta;

      EigenVector sigma_IN_sqr;

      EigenVector theta_pref;
      EigenVector d;
    };

    class RandomDummyRateNeurons
      : public virtual ::Backend::RandomDummyRateNeurons,
        protected virtual ::Backend::Eigen::DummyRateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
      friend class BCMPlasticity;
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RandomDummyRateNeurons);
      ~RandomDummyRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      bool staged_integrate_timestep(FloatT dt) override;
      EigenVector const& rate() override;
      EigenVector const& rate(unsigned int n_back=0) override;

    private:
      EigenVector _rate;
    };

    class AgentSenseRateNeurons
      : public virtual ::Backend::AgentSenseRateNeurons,
        public virtual ::Backend::Eigen::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
      friend class BCMPlasticity;
    public:
      // AgentSenseRateNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(AgentSenseRateNeurons);
      ~AgentSenseRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses*,
                         ::Backend::RatePlasticity*) override;

      bool staged_integrate_timestep(FloatT dt) override;
      EigenVector const& rate() override;
      EigenVector const& rate(unsigned int n_back=0) override;

    protected:
      EigenVector _rate;
      //::Eigen::Matrix<FloatT, Eigen::Dynamic, 2> _tuning;
    };
  } // namespace Eigen
} // namespace Backend

