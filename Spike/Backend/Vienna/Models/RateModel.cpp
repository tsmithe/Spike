#include "RateModel.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Vienna, RateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Vienna, RateSynapses);
SPIKE_EXPORT_BACKEND_TYPE(Vienna, RatePlasticity);

namespace Backend {
  namespace Vienna {
    void RateNeurons::prepare() {
      reset_state();
    }

    void RateNeurons::reset_state() {
      int size = frontend()->size;
      int timesteps = frontend()->timesteps;

      _rate = viennacl::zero_vector<FloatT>(size);
      _rate_cpu = EigenVector::Zero(size);
      _rate_cpu_timestep = frontend()->timesteps;
    }

    const EigenVector& RateNeurons::rate() {
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
      _vienna_dendrites.push_back({_vienna_synapses, _vienna_plasticity});
    }

    void RateNeurons::update_rate(FloatT dt) {
      viennacl::vector<FloatT> total_activation
        = viennacl::zero_vector<FloatT>(frontend()->size);

      for (const auto& dendrite_pair : _vienna_dendrites)
        total_activation += dendrite_pair.first->_activation;

      // TODO: Generalize transfer function
      _rate += dt * viennacl::linalg::element_tanh(total_activation);
    }

    void RateSynapses::prepare() {
      neurons_pre = dynamic_cast<::Backend::Vienna::RateNeurons*>
        (frontend()->neurons_pre.backend());
      reset_state();
    }

    void RateSynapses::reset_state() {
      int size_post = frontend()->neurons_post.size;
      int timesteps = frontend()->timesteps;

      _activation = viennacl::zero_vector<FloatT>(size_post);
      _activation_cpu = EigenVector::Zero(size_post);
      _activation_cpu_timestep = timesteps;
    }

    void RateSynapses::update_activation(FloatT dt) {
      // TODO:: Generalise activation function
      _activation = viennacl::linalg::prod(_weights, neurons_pre->_rate);
    }

    const EigenVector& RateSynapses::activation() {
      // Ensure that host copy is up to date:
      int curr_timestep = frontend()->timesteps;
      if (curr_timestep != _activation_cpu_timestep) {
        viennacl::copy(_activation, _activation_cpu);
        _activation_cpu_timestep = curr_timestep;
      }

      return _activation_cpu;
    }

    const EigenMatrix& RateSynapses::weights() {
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
        (frontend()->synapses.backend());
      reset_state();
    }

    void RatePlasticity::reset_state() {
      int size_post = frontend()->synapses.neurons_post.size;
      int size_pre = frontend()->synapses.neurons_pre.size;
      int timesteps = frontend()->timesteps;

      // TODO: Better record of initial weights state:
      synapses->_weights = viennacl::zero_matrix<FloatT>(size_pre, size_post);
      synapses->_weights_cpu = EigenMatrix::Zero(size_pre, size_post);
      synapses->_weights_cpu_timestep = timesteps;
    }

    void RatePlasticity::apply_plasticity(FloatT dt) {
      // TODO
    }

    /*
    void RateElectrodes::prepare() {}
    void RateElectrodes::reset_state() {}
    */
  }
}
