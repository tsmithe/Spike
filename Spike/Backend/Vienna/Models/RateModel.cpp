#include "RateModel.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Vienna, RateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Vienna, DummyRateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Vienna, RateSynapses);
SPIKE_EXPORT_BACKEND_TYPE(Vienna, RatePlasticity);

namespace Backend {
  namespace Vienna {
    void RateNeurons::prepare() {
      reset_state();

      int size = frontend()->size;

      _rate_history = viennacl::zero_matrix<FloatT>(size, 1);

      _total_activation = viennacl::zero_vector<FloatT>(size);
      _half = viennacl::scalar_vector<FloatT>(size, 0.5);
      _alpha = viennacl::scalar_vector<FloatT>(size, frontend()->alpha);
      _beta = frontend()->beta;
      _tau = frontend()->tau;
    }

    void RateNeurons::reset_state() {
      int timesteps = frontend()->timesteps;
      int size = frontend()->size;

      _rate_history = viennacl::zero_matrix<FloatT>(_rate_history.size1(),
                                                    _rate_history.size2());
      _rate_cpu = EigenVector::Zero(size);
      _rate_cpu_timestep = frontend()->timesteps;
    }

    const EigenVector& RateNeurons::rate() {
      // Ensure that host copy is up to date:
      int curr_timestep = frontend()->timesteps;
      if (curr_timestep != _rate_cpu_timestep) {
        viennacl::copy(_rate(), _rate_cpu);
        _rate_cpu_timestep = curr_timestep;
      }

      return _rate_cpu;
    }

    viennacl::vector<FloatT> RateNeurons::_rate(unsigned int n_back) {
      // TODO: performance -- just return a 'view'
      int i = (_rate_hist_idx - n_back) % _rate_history.size2();
      viennacl::vector<FloatT> v(viennacl::column(_rate_history, i));
      return viennacl::column(_rate_history, i);
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
        // Update rate history:
        _rate_hist_idx = (_rate_hist_idx + 1) % _rate_history.size2();
        viennacl::matrix_range<viennacl::matrix<FloatT> > _rate_history_col
          (_rate_history,
           viennacl::range(0, _rate_history.size1()),
           viennacl::range(_rate_hist_idx, _rate_hist_idx + 1));

        viennacl::matrix_base<FloatT> _new_rate_as_matrix
          (_new_rate.handle(),
           _new_rate.size(), _new_rate.start(),
           _new_rate.stride(), _new_rate.internal_size(),
           1, 0, 1, 1, // 1 column; start 0, stride 1, internal columns 1
           _rate_history.row_major());
        _rate_history_col = _new_rate_as_matrix;

        done_timestep = false; // false for next time
        return true;
      }

      int i = 0;
      for (const auto& dendrite_pair : _vienna_dendrites) {
        auto& synapses = dendrite_pair.first;
        auto activation_i = synapses->activation();

        if (i == 0) {
          if (frontend()->alpha == 0)
            _total_activation = activation_i;
          else
            _total_activation = activation_i - _alpha;
        } else {
          _total_activation += activation_i;
        }

        if (isanynan(activation_i)) {
          std::cout << "\n !!!!!!!!!!!!!!!!!! "
                    << i << " " << synapses->frontend() << "\n";
          assert(false);
        }

        i++;
      }

      auto trans = transfer(_total_activation);
      if(isanynan(trans)) {
        std::cout << "\n" << trans << "\n";
        if(isanynan(_total_activation))
          std::cout << "\n!!!!! !!!!!";
        assert(false);
      }
      _new_rate = _rate() + (dt/_tau)*(-_rate() + trans);

      done_timestep = true;
      return false;
    }

    void DummyRateNeurons::prepare() {
      _rate_on.resize(frontend()->x_on.size());
      _rate_off.resize(frontend()->x_off.size());
      viennacl::copy(frontend()->x_on, _rate_on);
      viennacl::copy(frontend()->x_off, _rate_off);
    }

    void DummyRateNeurons::reset_state() {
    }

    void DummyRateNeurons::connect_input(::Backend::RateSynapses*,
                                         ::Backend::RatePlasticity*) {
    }

    bool DummyRateNeurons::staged_integrate_timestep(FloatT dt) {
      t += dt;
      dt_ = dt;
      return true;
    }

    EigenVector const& DummyRateNeurons::rate() {
      if (t > frontend()->t_on && t < frontend()->t_off)
        return frontend()->x_on;
      else
        return frontend()->x_off;
    }

    viennacl::vector<FloatT> DummyRateNeurons::_rate(unsigned int n_back) {
      FloatT t_ = t - dt_*n_back;
      if (t_ > frontend()->t_on && t_ < frontend()->t_off)
        return _rate_on;
      else
        return _rate_off;
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

    viennacl::vector<FloatT> RateSynapses::activation() {
      // TODO: caching; perf enhancements
      auto activ = viennacl::linalg::prod(_weights, neurons_pre->_rate(_delay));
      if (frontend()->scaling != 1)
        return frontend()->scaling * activ;
      else
        return activ;
    }

    void RateSynapses::delay(unsigned int d) {
      _delay = d;
      if (neurons_pre->_rate_history.size2() < (d+1))
        neurons_pre->_rate_history.resize
          (neurons_pre->_rate_history.size1(), d+1);
    }

    unsigned int RateSynapses::delay() {
      return _delay;
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

    void RateSynapses::weights(EigenMatrix const& w) {
      _weights_cpu = w;
      _weights_cpu_timestep = frontend()->timesteps;
      viennacl::copy(w, _weights);
    }

    void RatePlasticity::prepare() {
      synapses = dynamic_cast<::Backend::Vienna::RateSynapses*>
        (frontend()->synapses->backend());
      epsilon = frontend()->epsilon;
      reset_state();
    }

    void RatePlasticity::reset_state() {
    }

    void RatePlasticity::apply_plasticity(FloatT dt) {
      if (epsilon == 0)
        return;

      /*
      if (_using_multipliers) {
        synapses->_weights += dt * viennacl::linalg::element_prod
          (_multipliers, viennacl::linalg::outer_prod
           (synapses->neurons_post->_rate(), synapses->neurons_pre->_rate()));
      } else*/ {
        synapses->_weights += dt * epsilon * viennacl::linalg::outer_prod
          (synapses->neurons_post->_rate(), synapses->neurons_pre->_rate());
      }

      normalize_matrix_rows(synapses->_weights);
    }

    /* Can be used for maintaining a crude kind of sparseness: */
    void RatePlasticity::multipliers(EigenMatrix const& m) {
      _using_multipliers = true;
      viennacl::copy(m, _multipliers);
      _multipliers *= epsilon;
    }

    /*
    void RateElectrodes::prepare() {}
    void RateElectrodes::reset_state() {}
    */
  }
}
