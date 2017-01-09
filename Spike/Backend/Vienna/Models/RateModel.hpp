#pragma once

#include "Spike/Models/RateModel.hpp"

#include <viennacl/matrix.hpp>
#include <viennacl/vector.hpp>

namespace Backend {
  namespace Vienna {
    class RateSyapses;    // forward def
    class RatePlasticity; // forward def

    class RateNeurons : public virtual ::Backend::RateNeurons {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateNeurons);
      ~RateNeurons() override = default;

      void prepare() override;
      void reset_state() override;
      void push_data_front() override;
      void pull_data_back() override;

      void update_rate(float dt) override;

    private:
      std::vector<std::pair<RateSynapses*, RatePlasticity*> > _dendrites;
      viennacl::vector<float> _rates;
    };

    class RateSynapses : public virtual ::Backend::RateSynapses {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateSynapses);
      ~RateSynapses() override = default;

      void prepare() override;
      void reset_state() override;
      void push_data_front() override;
      void pull_data_back() override;

      void update_activation(float dt) override;

    private:
      viennacl::vector<float> _activation; // TODO: Need an explicit temporary?
      viennacl::matrix<float> _weights;
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
    };

    class RateModel : public virtual ::Backend::RateModel {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateModel);
      ~RateModel() override = default;

      void prepare() override;
      void reset_state() override;
      void push_data_front() override;
      void pull_data_back() override;
    };
  } // namespace Vienna
} // namespace Backend

