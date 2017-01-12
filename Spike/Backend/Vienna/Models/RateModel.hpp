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
      void push_data_front() override;
      void pull_data_back() override;

      void update_activation(float dt) override;
      const Eigen::VectorXf& activation() override;
      const Eigen::MatrixXf& weights() override;

    private:
      viennacl::vector<float> _activation; // TODO: Need an explicit temporary?
      Eigen::VectorXf _activation_cpu;
      int _activation_cpu_timestep = 0;

      viennacl::matrix<float> _weights;    // TODO: Generalize synapse types
      Eigen::MatrixXf _weights_cpu;
      int _weights_cpu_timestep = 0;

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
      friend class RateSynapses;
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

      const Eigen::VectorXf& rate() override;

    private:
      viennacl::vector<float> _rate;
      Eigen::VectorXf _rate_cpu;
      int _rate_cpu_timestep = 0;
      std::vector<::Backend::Vienna::RateSynapses*> _synapses;
      std::vector<::Backend::Vienna::RatePlasticity*> _plasticity;
    };

    /*
    class RateElectrodes : public virtual ::Backend::RateElectrodes {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateElectrodes);
      ~RateElectrodes() override = default;

      void prepare() override;
      void reset_state() override;
      void push_data_front() override;
      void pull_data_back() override;
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
    */
  } // namespace Vienna
} // namespace Backend

