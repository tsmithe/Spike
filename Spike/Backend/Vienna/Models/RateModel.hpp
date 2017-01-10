#pragma once

#include "Spike/Models/RateModel.hpp"

#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/prod.hpp>

namespace Backend {
  namespace Vienna {
    class RateNeurons;

    class RateSynapses : public virtual ::Backend::RateSynapses {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateSynapses);
      ~RateSynapses() override = default;

      void prepare() override;
      void reset_state() override;
      void push_data_front() override;
      void pull_data_back() override;

      void update_activation(float dt) override;

      viennacl::vector<float> activation; // TODO: Need an explicit temporary?
      viennacl::matrix<float> weights; // TODO: Generalize synapse types

    private:
      ::Backend::Vienna::RateNeurons* neurons_pre = nullptr;
     };

    class RatePlasticity : public virtual ::Backend::RatePlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RatePlasticity);
      ~RatePlasticity() override = default;

      void prepare() override;
      void reset_state() override;
      void push_data_front() override;
      void pull_data_back() override;

      void apply_plasticity(float dt) override;

    private:
      ::Backend::Vienna::RateSynapses* synapses = nullptr;
    };

    class RateNeurons : public virtual ::Backend::RateNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateNeurons);
      ~RateNeurons() override = default;

      void prepare() override;
      void reset_state() override;
      void push_data_front() override;
      void pull_data_back() override;

      void connect_input(::Backend::RateSynapses* synapses,
                         ::Backend::RatePlasticity* plasticity) override;
      void update_rate(float dt) override;

      viennacl::vector<float> rates;

    private:
      std::vector<std::pair<::Backend::Vienna::RateSynapses*,
                            ::Backend::Vienna::RatePlasticity*> > _dendrites;
    };

    class RateElectrodes : public virtual ::Backend::RateElectrodes {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateElectrodes);
      ~RateElectrodes() override = default;

      void prepare() override;
      void reset_state() override;
      void push_data_front() override;
      void pull_data_back() override;
    };

    /*
    class RateModel : public virtual ::Backend::RateModel {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateModel);
      ~RateModel() override = default;

      void prepare() override;
      void reset_state() override;
      void push_data_front() override;
      void pull_data_back() override;
    };
    */
  } // namespace Vienna
} // namespace Backend

