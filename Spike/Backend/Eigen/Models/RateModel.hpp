#pragma once

#include "Spike/Models/RateModel.hpp"

namespace Backend {
  namespace Eigen {
    class RateNeurons;    // forward

    class RateSynapses : public virtual ::Backend::RateSynapses {
      friend class RateNeurons;
      friend class RatePlasticity;
    public:
      RateSynapses() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateSynapses);
      ~RateSynapses() override = default;

      void prepare() override;
      void reset_state() override;

      // void update_activation(FloatT dt) override;
      const EigenVector& activation() override;
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
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RatePlasticity);
      ~RatePlasticity() override = default;

      void prepare() override;
      void reset_state() override;

      void apply_plasticity(FloatT dt) override;

    private:
      ::Backend::Eigen::RateSynapses/*Base*/* synapses = nullptr;
      FloatT epsilon = 0;

      EigenMatrix _multipliers;
      bool _using_multipliers = false;
    };

    class RateNeurons : public virtual ::Backend::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
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

      virtual const EigenVector& rate(unsigned int n_back);
      const EigenVector& rate() override;

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
    public:
      DummyRateNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(DummyRateNeurons);
      ~DummyRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses*,
                         ::Backend::RatePlasticity*) override;

      bool staged_integrate_timestep(FloatT dt) override;
      const EigenVector& rate() override;
      EigenVector const& rate(unsigned int n_back=0) override;

      void add_rate(FloatT duration, EigenVector rates) override;

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
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(InputDummyRateNeurons);
      ~InputDummyRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      bool staged_integrate_timestep(FloatT dt) override;
      const EigenVector& rate() override;
      EigenVector const& rate(unsigned int n_back=0) override;

    private:
      EigenVector _rate;

      FloatT theta;

      EigenVector sigma_IN_sqr;

      EigenVector theta_pref;
      EigenVector d;
    };
  } // namespace Eigen
} // namespace Backend

