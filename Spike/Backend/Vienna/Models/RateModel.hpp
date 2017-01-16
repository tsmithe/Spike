#pragma once

#include "Spike/Models/RateModel.hpp"

#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/prod.hpp>

namespace Backend {
  namespace Vienna {
    class RateNeurons;    // forward

    class RateSynapses : public virtual ::Backend::RateSynapses {
      friend class RateNeurons;
      friend class RatePlasticity;
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateSynapses);
      ~RateSynapses() override = default;

      void prepare() override;
      void reset_state() override;

      void update_activation(FloatT dt) override;
      const EigenVector& activation() override;
      const EigenMatrix& weights() override;

    private:
      viennacl::vector<FloatT> _activation; // TODO: Need an explicit temporary?
      EigenVector _activation_cpu;
      int _activation_cpu_timestep = 0;

      viennacl::matrix<FloatT> _weights;    // TODO: Generalize synapse types
      EigenMatrix _weights_cpu;
      int _weights_cpu_timestep = 0;

      ::Backend::Vienna::RateNeurons* neurons_pre = nullptr;
     };

    class RatePlasticity : public virtual ::Backend::RatePlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RatePlasticity);
      ~RatePlasticity() override = default;

      void prepare() override;
      void reset_state() override;

      void apply_plasticity(FloatT dt) override;

    private:
      ::Backend::Vienna::RateSynapses* synapses = nullptr;
    };

    class RateNeurons : public virtual ::Backend::RateNeurons {
      friend class RateSynapses;
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateNeurons);
      ~RateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses* synapses,
                         ::Backend::RatePlasticity* plasticity) override;
      void update_rate(FloatT dt) override;

      const EigenVector& rate() override;

    private:
      viennacl::vector<FloatT> _half;
      viennacl::vector<FloatT> _rate;
      EigenVector _rate_cpu;
      int _rate_cpu_timestep = 0;
      std::vector<
        std::pair<::Backend::Vienna::RateSynapses*,
                   ::Backend::Vienna::RatePlasticity*> > _vienna_dendrites;
    };

    /*
    class RateElectrodes : public virtual ::Backend::RateElectrodes {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateElectrodes);
      ~RateElectrodes() override = default;

      void prepare() override;
      void reset_state() override;
    };

    class RateModel : public virtual ::Backend::RateModel {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateModel);
      ~RateModel() override = default;

      void prepare() override;
      void reset_state() override;
    };
    */
  } // namespace Vienna
} // namespace Backend

