#include "RateModel.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Vienna, RateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Vienna, RateSynapses);
SPIKE_EXPORT_BACKEND_TYPE(Vienna, RatePlasticity);

namespace Backend {
  namespace Vienna {
    void RateNeurons::prepare() {
      reset_state();

      int size = frontend()->size;
      _total_activation = viennacl::zero_vector<FloatT>(size);
      _half = viennacl::scalar_vector<FloatT>(size, 0.5);
      _alpha = viennacl::scalar_vector<FloatT>(size, frontend()->alpha);
      _beta = frontend()->beta;
      _tau = frontend()->tau;
    }

    void RateNeurons::reset_state() {
      int timesteps = frontend()->timesteps;
      int size = frontend()->size;

      // TODO: Remove this ugly random init hack!
      _rate = viennacl::zero_vector<FloatT>(size);
      _rate_cpu = EigenVector::Random(size);
      viennacl::copy(_rate_cpu, _rate);
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

    /* NB: The argument is the total activation beyond the threshold `alpha` */
    template<typename T>
    inline T RateNeurons::transfer(T const& total_activation) {
      if (_beta == 1)
        return viennacl::linalg::element_tanh(total_activation);
      else
        return viennacl::linalg::element_tanh(_beta * total_activation);
    }

    bool RateNeurons::staged_integrate_timestep(FloatT dt) {
      if (done_timestep) {
        _rate = _new_rate;
        done_timestep = false;
        return true;
      }

      int i = 0;
      for (const auto& dendrite_pair : _vienna_dendrites) {
        auto& synapses = dendrite_pair.first;

        if (i == 0) {
          if (frontend()->alpha == 0)
            _total_activation = viennacl::linalg::prod
              (synapses->_weights, synapses->neurons_pre->_rate);
          else
            _total_activation = viennacl::linalg::prod
              (synapses->_weights, synapses->neurons_pre->_rate) - _alpha;
        } else {
          _total_activation += viennacl::linalg::prod
            (synapses->_weights, synapses->neurons_pre->_rate);
        }

        i++;
      }

      _new_rate = _rate + (dt/_tau)*(-_rate + transfer(_total_activation));

      done_timestep = true;
      return false;
    }

    void RateSynapses::prepare() {
      neurons_pre = dynamic_cast<::Backend::Vienna::RateNeurons*>
        (frontend()->neurons_pre->backend());
      neurons_post = dynamic_cast<::Backend::Vienna::RateNeurons*>
        (frontend()->neurons_post->backend());

      int size_post = frontend()->neurons_post->size;
      int size_pre = frontend()->neurons_pre->size;
      _weights = viennacl::zero_matrix<FloatT>(size_post, size_pre);

      // reset_state();
    }

    void RateSynapses::reset_state() {
      int size_post = frontend()->neurons_post->size;
      int size_pre = frontend()->neurons_pre->size;
      int timesteps = frontend()->timesteps;

      // _activation = viennacl::zero_vector<FloatT>(size_post);
      // _activation_cpu = EigenVector::Zero(size_post);
      // _activation_cpu_timestep = timesteps;

      // TODO: Better record of initial weights state:
      // viennacl::copy(frontend()->initial_weights, _weights);
      // _weights_cpu = frontend()->initial_weights;
      // _weights_cpu_timestep = timesteps;
    }

    /*
    void RateSynapses::update_activation(FloatT dt) {
      // TODO:: Generalise activation function
      // _activation = viennacl::linalg::prod(_weights, neurons_pre->_rate);
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
    */

    const EigenMatrix& RateSynapses::weights() {
      // Ensure that host copy is up to date:
      int curr_timestep = frontend()->timesteps;
      if (curr_timestep != _weights_cpu_timestep) {
        viennacl::copy(_weights, _weights_cpu);
        _weights_cpu_timestep = curr_timestep;
      }

      return _weights_cpu;
    }

    void RateSynapses::weights(EigenMatrix const& w) {
      _weights_cpu = w;
      _weights_cpu_timestep = frontend()->timesteps;
      viennacl::copy(w, _weights);
    }

    void RatePlasticity::prepare() {
      synapses = dynamic_cast<::Backend::Vienna::RateSynapses*>
        (frontend()->synapses->backend());
      reset_state();
    }

    void RatePlasticity::reset_state() {
    }

    void RatePlasticity::apply_plasticity(FloatT dt) {
      // TODO: Parameterize and generalize this
      synapses->_weights += dt * 0.01 *
        viennacl::linalg::outer_prod(synapses->neurons_post->_rate,
                                     synapses->neurons_pre->_rate);
      normalize_matrix_rows(synapses->_weights);
    }

    /*
    void RateElectrodes::prepare() {}
    void RateElectrodes::reset_state() {}
    */
  }
}
