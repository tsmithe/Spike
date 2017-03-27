#include "RateModel.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Vienna, RateNeurons);
//SPIKE_EXPORT_BACKEND_TYPE(Vienna, DummyRateNeurons);
//SPIKE_EXPORT_BACKEND_TYPE(Vienna, InputDummyRateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Vienna, RateSynapses);
//SPIKE_EXPORT_BACKEND_TYPE(Vienna, RatePlasticity);

namespace Backend {
  namespace Vienna {
    void RateNeurons::prepare() {
      reset_state();

      int size = frontend()->size;

      _rate_history = viennacl::zero_matrix<FloatT>(size, 1);

      _total_activation = viennacl::zero_vector<FloatT>(size);

      _ones = viennacl::scalar_vector<FloatT>(size, 1);
      _half = viennacl::scalar_vector<FloatT>(size, 0.5);

      /*
      _alpha = viennacl::scalar_vector<FloatT>(size, frontend()->alpha);
      _beta = frontend()->beta;
      _tau = frontend()->tau;
      */
    }

    void RateNeurons::reset_state() {
      int timesteps = frontend()->timesteps;
      int size = frontend()->size;

      _rate_history = viennacl::zero_matrix<FloatT>(_rate_history.size1(),
                                                    _rate_history.size2());
      _rate_cpu = EigenVector::Zero(size);
      _rate_cpu_timestep = frontend()->timesteps;
    }

    void RateNeurons::add_group(RateNeuronGroup* group) {
      std::cout << "TODO " << group << "\n";
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
      int i = _rate_hist_idx - n_back;
      if (i < 0) i += _rate_history.size2();
      // if (n_back != 0) {
      //   std::cout << this << "- " << _rate_hist_idx << " - " << n_back << " - " << _rate_history.size2() << " - " << i << "\n";
      // }
      /*
      viennacl::vector<FloatT> v(viennacl::column(_rate_history, i));
      if (isanynan(viennacl::vector<FloatT>(1.0*v))) {
        std::cout << "\n" << v << "\n";
        std::cout << "\n????? " << frontend()
                  << " " << _rate_hist_idx << " " << n_back
                  << " " << _rate_history.size2()
                  << "\n";
        assert(false);
      }
      */
      return viennacl::column(_rate_history, i);
    }

    void RateNeurons::connect_input(::Backend::RateSynapses* synapses/*,
                                    ::Backend::RatePlasticity* plasticity*/) {
      ::Backend::Vienna::RateSynapses* _vienna_synapses
        = dynamic_cast<::Backend::Vienna::RateSynapses*>(synapses);
      /*
      ::Backend::Vienna::RatePlasticity* _vienna_plasticity
        = dynamic_cast<::Backend::Vienna::RatePlasticity*>(plasticity);
      */
      _vienna_dendrites.push_back(/*{*/_vienna_synapses/*, _vienna_plasticity}*/);
    }

    /* NB: The argument is the total activation beyond the threshold `alpha` */
    template<typename T>
    inline T RateNeurons::transfer(T const& total_activation) {
      // 2*logistic(x) = 1 + tanh(x/2)
      viennacl::vector<FloatT> transfer_tmp;
      if (_beta == 1)
        transfer_tmp = viennacl::linalg::element_tanh(total_activation) + _ones;
      else
        transfer_tmp = viennacl::linalg::element_tanh(_beta*total_activation) + _ones;
      return viennacl::linalg::element_prod(_half, transfer_tmp);
    }

    bool RateNeurons::staged_integrate_timestep(FloatT dt) {
      if (done_timestep) {
// TODO: THE FOLLOWING GUARD IS VERY HACKY -- WHY DOESN'T IT WORK??
#ifdef VIENNACL_WITH_OPENCL
        // Update rate history:
        _rate_hist_idx = (_rate_hist_idx + 1) % _rate_history.size2();
        // std::cout << this << ": " << _rate_hist_idx << "\n";
        viennacl::range _rate_hist_r1(0, _rate_history.size1());
        viennacl::range _rate_hist_r2(_rate_hist_idx, _rate_hist_idx + 1);
        viennacl::matrix_range<viennacl::matrix<FloatT> > _rate_history_col
          (_rate_history, _rate_hist_r1, _rate_hist_r2);

        viennacl::matrix_base<FloatT> _new_rate_as_matrix
          (_new_rate.handle(),
           _new_rate.size(), _new_rate.start(),
           _new_rate.stride(), _new_rate.internal_size(),
           1, 0, 1, 1, // 1 column; start 0, stride 1, internal columns 1
           _rate_history.row_major());
        _rate_history_col = _new_rate_as_matrix;
#endif

        done_timestep = false; // false for next time
        return true;
      }

      int i = 0;
      for (const auto& dendrite/*_pair*/ : _vienna_dendrites) {
        auto& synapses = dendrite/*_pair.first*/;
        auto activation_i = synapses->_activation();

        if (i == 0) {
          /*if (frontend()->alpha == 0)
            _total_activation = activation_i;
            else*/
            _total_activation = activation_i - _alpha;
        } else {
          _total_activation += activation_i;
        }

        /*
        if (isanynan(activation_i)) {
          std::cout << "\n !!!!!!!!!!!!!!!!!! "
                    << i << " " << synapses->frontend() << "\n";
          assert(false);
        }
        */

        i++;
      }

      auto trans = transfer(_total_activation);
      /*
      if(isanynan(trans)) {
        std::cout << "\n" << trans << "\n";
        if(isanynan(_total_activation))
          std::cout << "\n!!!!! !!!!!";
        assert(false);
      }
      */
      _new_rate = _rate() + (dt/_tau)*(-_rate() + trans);

      done_timestep = true;
      return false;
    }

    /*
    void DummyRateNeurons::prepare() {
      /*
      _rate_on.resize(frontend()->x_on.size());
      _rate_off.resize(frontend()->x_off.size());
      viennacl::copy(frontend()->x_on, _rate_on);
      viennacl::copy(frontend()->x_off, _rate_off);
      * /
      _schedule_idx = 0;
      _curr_rate_t = 0;
    }

    void DummyRateNeurons::reset_state() {
    }

    void DummyRateNeurons::connect_input(::Backend::RateSynapses*,
                                         ::Backend::RatePlasticity*) {
    }

    void DummyRateNeurons::add_rate(FloatT duration, EigenVector rates) {
      viennacl::vector<FloatT> _vcl_rates(frontend()->size);
      viennacl::copy(rates, _vcl_rates);
      _rate_schedule.push_back({duration, _vcl_rates});
    }

    bool DummyRateNeurons::staged_integrate_timestep(FloatT dt) {
      t += dt;
      dt_ = dt;

      _curr_rate_t += dt;
      if (_curr_rate_t > _rate_schedule[_schedule_idx].first) {
        _curr_rate_t = _curr_rate_t - _rate_schedule[_schedule_idx].first;
        _schedule_idx++;
        if (_schedule_idx >= _rate_schedule.size())
          _schedule_idx = 0;
      }

      return true;
    }

    EigenVector const& DummyRateNeurons::rate() {
      return frontend()->rate_schedule[_schedule_idx].second;
    }

    viennacl::vector<FloatT> DummyRateNeurons::_rate(unsigned int n_back) {
      return _rate_schedule[_schedule_idx].second;
    }

    void InputDummyRateNeurons::prepare() {
      theta_pref.resize(frontend()->size);
      d.resize(frontend()->size);
      _rate_cpu.resize(frontend()->size);

      viennacl::copy(frontend()->theta_pref, theta_pref);

      sigma_IN_sqr = viennacl::scalar_vector<FloatT>(frontend()->size,
                                                     frontend()->sigma_IN);
      sigma_IN_sqr = viennacl::linalg::element_prod(sigma_IN_sqr,sigma_IN_sqr);

      reset_state();
    }

    void InputDummyRateNeurons::reset_state() {
      theta = 0;
    }

    bool InputDummyRateNeurons::staged_integrate_timestep(FloatT dt) {
      t += dt;
      dt_ = dt;

      if (t > frontend()->t_stop_after)
        return true;

      theta += dt * 2 * M_PI * frontend()->revolutions_per_second;
      if (theta > 2*M_PI)
        theta -= 2*M_PI;

      // TODO: Inefficient allocation?
      d = theta_pref
        - viennacl::vector<FloatT>
        (viennacl::scalar_vector<FloatT>(frontend()->size, theta));
      d = viennacl::linalg::element_fabs(d);
      // TODO: Very inefficient -- needs custom kernel!
      EigenVector d_tmp(d.size());
      viennacl::copy(d, d_tmp);
      d_tmp = d_tmp.array().min(2*M_PI - d_tmp.array());
      viennacl::copy(d_tmp, d);

      return true;
    }

    viennacl::vector<FloatT> InputDummyRateNeurons::_rate(unsigned int n_back) {
      assert(n_back == 0); // TODO: support delays here?

      if (t > frontend()->t_stop_after)
        return viennacl::zero_vector<FloatT>(frontend()->size);

      // TODO: FIX THIS HERE !
      viennacl::vector<FloatT> v(frontend()->size);
      /*
      v = frontend()->lambda * viennacl::linalg::element_exp
        (viennacl::linalg::element_div
         (viennacl::linalg::element_cos(d), sigma_IN_sqr));
      * /
      // TODO: Probably want custom kernel!
      v = frontend()->lambda * viennacl::linalg::element_exp
        (viennacl::linalg::element_div
         (-viennacl::vector<FloatT>(viennacl::linalg::element_prod(d, d)),
          2*sigma_IN_sqr));
      return v;
    }

    EigenVector const& InputDummyRateNeurons::rate() {
      viennacl::copy(_rate(), _rate_cpu);
      return _rate_cpu;
    }
    */

    void RateSynapses::prepare() {
      /*
      neurons_pre = dynamic_cast<::Backend::Vienna::RateNeurons*>
        (frontend()->neurons_pre->backend());
      neurons_post = dynamic_cast<::Backend::Vienna::RateNeurons*>
        (frontend()->neurons_post->backend());

      int size_post = frontend()->neurons_post->size;
      int size_pre = frontend()->neurons_pre->size;
      */

      neurons = dynamic_cast<::Backend::Vienna::RateNeurons*>
        (frontend()->neurons->backend());
      int size = frontend()->neurons->size;

      _weights = viennacl::zero_matrix<FloatT>(size, size);

      // reset_state();
    }

    void RateSynapses::reset_state() {
      int size = frontend()->neurons->size;
      // int size_post = frontend()->neurons_post->size;
      // int size_pre = frontend()->neurons_pre->size;
      int timesteps = frontend()->timesteps;

      // _activation = viennacl::zero_vector<FloatT>(size_post);
      _activation_cpu = EigenVector::Zero(size);
      _activation_cpu_timestep = timesteps;

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
    */

    void RateSynapses::add_group(RateSynapseGroup* group) {
      std::cout << "TODO " << group << "\n";
    }
      
    const EigenVector& RateSynapses::activation() {
      // Ensure that host copy is up to date:
      int curr_timestep = frontend()->timesteps;
      if (curr_timestep != _activation_cpu_timestep) {
        viennacl::copy(_activation(), _activation_cpu);
        _activation_cpu_timestep = curr_timestep;
      }

      return _activation_cpu;
    }

    viennacl::vector<FloatT> RateSynapses::_activation() {
      // TODO: caching; perf enhancements

      assert(_delay == 0);
      viennacl::vector<FloatT> activ = viennacl::linalg::prod(_weights, neurons->_rate(_delay));

      /*
      if (isanynan(viennacl::vector<FloatT>(1.0*activ))) {
        if (isanynan(_weights)) {
          std::cout << _weights << "\n";
        }
        std::cout << "\n !!!!!!!!!!!!!!!!!! "
                  << frontend() << "\n";
        assert(false);
      }
      */

      /*
      if (frontend()->scaling != 1)
        return frontend()->scaling * activ;
      else */
        return activ;
    }

    void RateSynapses::delay(unsigned int d) {
      _delay = d;
      if (neurons->_rate_history.size2() < (d+1))
        neurons->_rate_history.resize
          (neurons->_rate_history.size1(), d+1);
      // std::cout << neurons_pre << "= " << neurons_pre->_rate_history.size2() << "\n";
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

    /*
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

    void RateSynapses::weights(EigenMatrix const& w) {
      _weights_cpu = w;
      _weights_cpu_timestep = frontend()->timesteps;
      viennacl::copy(w, _weights);
    }

    /*
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
      } else* / {
        synapses->_weights += dt * epsilon * viennacl::linalg::outer_prod
          (synapses->neurons_post->_rate(), synapses->neurons_pre->_rate());
      }

      normalize_matrix_rows(synapses->_weights);
    }

    // Can be used for maintaining a crude kind of sparseness:
    void RatePlasticity::multipliers(EigenMatrix const& m) {
      _using_multipliers = true;
      viennacl::copy(m, _multipliers);
      _multipliers *= epsilon;
    }
    */

    /*
    void RateElectrodes::prepare() {}
    void RateElectrodes::reset_state() {}
    */
  }
}
