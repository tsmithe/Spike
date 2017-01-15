#include "RateModel.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Vienna, RateNeurons)
SPIKE_EXPORT_BACKEND_TYPE(Vienna, RateSynapses)
SPIKE_EXPORT_BACKEND_TYPE(Vienna, RatePlasticity)

namespace Backend {
  namespace Vienna {
    void RateNeurons::prepare() {
      // std::cout << frontend() << "\n";
      // reset_state();
    }

    void RateNeurons::reset_state() {
      int size = frontend()->size;
      _rate = viennacl::zero_vector<float>(size);
      _rate_cpu = Eigen::VectorXf::Zero(size);
      _rate_cpu_timestep = frontend()->timesteps;
    }

    const Eigen::VectorXf& RateNeurons::rate() {
      // Ensure that host copy is up to date:
      int curr_timestep = frontend()->timesteps;
      if (curr_timestep != _rate_cpu_timestep) {
        viennacl::copy(_rate, _rate_cpu);
        _rate_cpu_timestep = curr_timestep;
      }

      return _rate_cpu;
    }

    void RateNeurons::connect_input(::Backend::RateSynapses* synapses,
                                    ::Backend::RatePlasticity* plasticity) {
      ::Backend::Vienna::RateSynapses* _vienna_synapses
        = dynamic_cast<::Backend::Vienna::RateSynapses*>(synapses);
      ::Backend::Vienna::RatePlasticity* _vienna_plasticity
        = dynamic_cast<::Backend::Vienna::RatePlasticity*>(plasticity);
      // std::cout << synapses_ << ", " << plasticity_ << "\n";
      /*
      std::pair<::Backend::Vienna::RateSynapses*,
        ::Backend::Vienna::RatePlasticity*> p(synapses_, plasticity_);
      _dendrites.push_back(p); // {synapses_, plasticity_});
      */
      _synapses.push_back(_vienna_synapses);
      // _plasticity.push_back(_vienna_plasticity);
    }

    void RateNeurons::update_rate(float dt) {
      viennacl::vector<float> total_activation
        = viennacl::zero_vector<float>(frontend()->size);
      /*
      for (const auto& dendrite_pair : _dendrites)
        total_activation += dendrite_pair.first->_activation;
      */
      for (const auto& synapse : _synapses)
        total_activation += synapse->_activation;

      // TODO: Generalize transfer function
      _rate += dt * viennacl::linalg::element_tanh(total_activation);
    }

    void RateSynapses::prepare() {
      neurons_pre = dynamic_cast<::Backend::Vienna::RateNeurons*>
        (frontend()->neurons_pre->backend());
      // reset_state();
    }

    void RateSynapses::reset_state() {
      int size_post = frontend()->neurons_post->size;
      int size_pre = frontend()->neurons_pre->size;
      int timesteps = frontend()->timesteps;

      std::cout << "size: " << size_pre << ", " << size_post << "\n";

      _activation = viennacl::zero_vector<float>(size_post);
      _activation_cpu = Eigen::VectorXf::Zero(size_post);
      _activation_cpu_timestep = timesteps;

      _weights = viennacl::zero_matrix<float>(size_pre, size_post);
      _weights_cpu = Eigen::MatrixXf::Zero(size_pre, size_post);
      _weights_cpu_timestep = timesteps;
    }

    void RateSynapses::update_activation(float dt) {
      // TODO:: Generalise activation function
      _activation = viennacl::linalg::prod(_weights, neurons_pre->_rate);
    }

    const Eigen::VectorXf& RateSynapses::activation() {
      // Ensure that host copy is up to date:
      int curr_timestep = frontend()->timesteps;
      if (curr_timestep != _activation_cpu_timestep) {
        viennacl::copy(_activation, _activation_cpu);
        _activation_cpu_timestep = curr_timestep;
      }

      return _activation_cpu;
    }

    const Eigen::MatrixXf& RateSynapses::weights() {
      // Ensure that host copy is up to date:
      int curr_timestep = frontend()->timesteps;
      if (curr_timestep != _weights_cpu_timestep) {
        viennacl::copy(_weights, _weights_cpu);
        _weights_cpu_timestep = curr_timestep;
      }

      return _weights_cpu;
    }

    void RatePlasticity::prepare() {
      synapses = dynamic_cast<::Backend::Vienna::RateSynapses*>
        (frontend()->synapses->backend());
    }

    void RatePlasticity::reset_state() {
      // TODO
    }

    void RatePlasticity::apply_plasticity(float dt) {
      // TODO
    }

    /*
    void RateElectrodes::prepare() {}
    void RateElectrodes::reset_state() {}
    */
  }
}
