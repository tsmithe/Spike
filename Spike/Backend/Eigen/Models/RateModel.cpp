#include "RateModel.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Eigen, RateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Eigen, DummyRateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Eigen, InputDummyRateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Eigen, RandomDummyRateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Eigen, AgentSenseRateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Eigen, RateSynapses);
SPIKE_EXPORT_BACKEND_TYPE(Eigen, RatePlasticity);

namespace Backend {
  namespace Eigen {
    void RateNeurons::prepare() {
      reset_state();

      int size = frontend()->size;

      _rate_history = EigenMatrix::Zero(size, 1);

      _total_activation = EigenVector::Zero(size);
    }

    void RateNeurons::reset_state() {
      int timesteps = frontend()->timesteps;
      int size = frontend()->size;

      _rate_history = EigenMatrix::Zero(_rate_history.rows(),
                                        _rate_history.cols());
      _rate = EigenVector::Zero(size);
    }

    const EigenVector& RateNeurons::rate() {
      return rate(0);
    }

    const EigenVector& RateNeurons::rate(unsigned int n_back) {
      // TODO: performance -- just return a 'view'
      int i = _rate_hist_idx - n_back;
      if (i < 0) i += _rate_history.cols();
      _rate = _rate_history.col(i);
      return _rate;
    }

    void RateNeurons::connect_input(::Backend::RateSynapses* synapses,
                                    ::Backend::RatePlasticity* plasticity) {
      ::Backend::Eigen::RateSynapses* _eigen_synapses
        = dynamic_cast<::Backend::Eigen::RateSynapses*>(synapses);
      ::Backend::Eigen::RatePlasticity* _eigen_plasticity
        = dynamic_cast<::Backend::Eigen::RatePlasticity*>(plasticity);
      _eigen_dendrites.push_back({_eigen_synapses, _eigen_plasticity});
    }

    /* NB: The argument is the total activation beyond the threshold `alpha` */
    template<typename T>
    inline T RateNeurons::transfer(T const& total_activation) {
      // 2*logistic(x) = 1 + tanh(x/2)
      if (frontend()->beta == 1)
        return 0.5*(total_activation.array().tanh() + 1).matrix();
      else
        return 0.5*((frontend()->beta*total_activation).array().tanh() + 1).matrix();
    }

    bool RateNeurons::staged_integrate_timestep(FloatT dt) {
      if (done_timestep) {
        // Update rate history:
        _rate_hist_idx = (_rate_hist_idx + 1) % _rate_history.cols();
        _rate_history.col(_rate_hist_idx).swap(_new_rate);

        done_timestep = false; // false for next time
        return true;
      }

      int i = 0;
      for (const auto& dendrite_pair : _eigen_dendrites) {
        auto& synapses = dendrite_pair.first;
        auto activation_i = synapses->activation();

        if (i == 0) {
          if (frontend()->alpha == 0)
            _total_activation = activation_i;
          else
            _total_activation = (activation_i.array() - frontend()->alpha).matrix();
        } else {
          _total_activation += activation_i;
        }

        ++i;
      }

      _new_rate = rate() + (dt/frontend()->tau)*(-rate() + transfer(_total_activation));

      done_timestep = true;
      return false;
    }

    void DummyRateNeurons::prepare() {
      _schedule_idx = 0;
      _curr_rate_t = 0;
    }

    void DummyRateNeurons::reset_state() {
    }

    void DummyRateNeurons::connect_input(::Backend::RateSynapses*,
                                         ::Backend::RatePlasticity*) {
      assert("You shouldn't be doing this" && false);
    }

    void DummyRateNeurons::add_schedule(FloatT duration, EigenVector rates) {
      // _rate_schedule.push_back({duration, rates});
    }

    bool DummyRateNeurons::staged_integrate_timestep(FloatT dt) {
      t += dt;
      dt_ = dt;

      _curr_rate_t += dt;
      if (_curr_rate_t > frontend()->rate_schedule[_schedule_idx].first) {
        _curr_rate_t = _curr_rate_t - frontend()->rate_schedule[_schedule_idx].first;
        ++_schedule_idx;
        if (_schedule_idx >= frontend()->rate_schedule.size())
          _schedule_idx = 0;
      }

      return true;
    }

    EigenVector const& DummyRateNeurons::rate() {
      return frontend()->rate_schedule[_schedule_idx].second;
    }

    EigenVector const& DummyRateNeurons::rate(unsigned int n_back) {
      assert(n_back == 0);
      return frontend()->rate_schedule[_schedule_idx].second;
    }

    void InputDummyRateNeurons::prepare() {
      _schedule_idx = 0;
      _curr_rate_t = 0;

      theta_pref.resize(frontend()->size);
      d.resize(frontend()->size);
      _rate.resize(frontend()->size);

      theta_pref = frontend()->theta_pref;

      sigma_IN_sqr = EigenVector::Constant(frontend()->size,
                                           frontend()->sigma_IN);
      sigma_IN_sqr = sigma_IN_sqr.cwiseProduct(sigma_IN_sqr);

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

      _curr_rate_t += dt;
      if (_curr_rate_t > frontend()->revs_schedule[_schedule_idx].first) {
        _curr_rate_t = _curr_rate_t - frontend()->revs_schedule[_schedule_idx].first;
        ++_schedule_idx;
        if (_schedule_idx >= frontend()->revs_schedule.size())
          _schedule_idx = 0;
      }

      FloatT revs = frontend()->revs_schedule[_schedule_idx].second;

      theta += dt * 2 * M_PI * revs;
      if (theta > 2*M_PI)
        theta -= 2*M_PI;

      d = (theta_pref.array() - theta).abs().matrix();
      d = d.array().min(2*M_PI - d.array());

      return true;
    }

    EigenVector const& InputDummyRateNeurons::rate(unsigned int n_back) {
      assert(n_back == 0); // TODO: support delays here?

      if (t > frontend()->t_stop_after) {
        _rate = EigenVector::Zero(frontend()->size); 
        return _rate;
      }

      _rate = frontend()->lambda *
        (-d.cwiseProduct(d).cwiseQuotient(2*sigma_IN_sqr)).array().exp().matrix();
      return _rate;
    }

    EigenVector const& InputDummyRateNeurons::rate() {
      return rate(0);
    }

    void RandomDummyRateNeurons::prepare() {
      _rate.resize(frontend()->size);
    }

    void RandomDummyRateNeurons::reset_state() {
    }

    bool RandomDummyRateNeurons::staged_integrate_timestep(FloatT dt) {
      t += dt;
      dt_ = dt;

      if (t > frontend()->t_stop_after) {
        _rate = EigenVector::Zero(frontend()->size);
        return true;
      }

      EigenVector rand_activ
        = ::Eigen::make_random_matrix(frontend()->size, 1, 4,
                                      false, 0, 0, true);
      // _rate += (dt/frontend()->tau)*(-_rate + transfer(rand_activ));
      _rate.swap(rand_activ);
      // _rate = EigenVector::Random(frontend()->size);

      return true;
    }

    EigenVector const& RandomDummyRateNeurons::rate(unsigned int n_back) {
      assert(n_back == 0); // TODO: support delays here?
      return _rate;
    }

    EigenVector const& RandomDummyRateNeurons::rate() {
      return rate(0);
    }

    void AgentSenseRateNeurons::prepare() {
      _rate.resize(frontend()->size);
    }

    void AgentSenseRateNeurons::reset_state() {
    }

    void AgentSenseRateNeurons::connect_input(::Backend::RateSynapses*,
                                              ::Backend::RatePlasticity*) {
      assert("You shouldn't be doing this" && false);
    }

    bool AgentSenseRateNeurons::staged_integrate_timestep(FloatT dt) {
      int n = 0;
      Agent* agent = frontend()->agent;
      ::Eigen::Matrix<FloatT, 2, 1> here;
      FloatT max_dist = sqrt(agent->bound_x * agent->bound_x
                             + agent->bound_y * agent->bound_y);
      for (int i = 0; i < agent->bound_y; ++i) {
        for (int j = 0; j < agent->bound_x; ++j) {
          here = {j, i};
          FloatT distance = (agent->position - here).norm();
          _rate(n) = (max_dist - distance) / max_dist;
          ++n;
        }
      }
      return true;
    }

    EigenVector const& AgentSenseRateNeurons::rate() {
      return rate(0);
    }

    EigenVector const& AgentSenseRateNeurons::rate(unsigned int n_back) {
      if (n_back)
        assert("Delays not yet supported here" && false);

      return _rate;
    }

    void RateSynapses::prepare() {
      neurons_pre = dynamic_cast<::Backend::Eigen::RateNeurons*>
        (frontend()->neurons_pre->backend());
      neurons_post = dynamic_cast<::Backend::Eigen::RateNeurons*>
        (frontend()->neurons_post->backend());

      int size_post = frontend()->neurons_post->size;
      int size_pre = frontend()->neurons_pre->size;

      _weights = EigenMatrix::Zero(size_post, size_pre);

      reset_state();
    }

    void RateSynapses::reset_state() {
      int size_post = frontend()->neurons_post->size;
      int size_pre = frontend()->neurons_pre->size;
      int timesteps = frontend()->timesteps;

      _activation = EigenVector::Zero(size_post);
    }

    void RateSynapses::make_sparse() {
      if (is_sparse)
        return;

      int size_post = frontend()->neurons_post->size;
      int size_pre = frontend()->neurons_pre->size;
      assert(_weights.rows() == size_post);
      assert(_weights.cols() == size_pre);

      _sp_weights.resize(size_post, size_pre);
      _sparsity.resize(size_post, size_pre);
      std::vector<::Eigen::Triplet<FloatT> > coefficients;
      std::vector<::Eigen::Triplet<FloatT> > sparsity_coeffs;
      for (int i = 0; i < size_post; ++i) {
        for (int j = 0; j < size_pre; ++j) {
          FloatT val = _weights(i, j);
          if (val != 0) {
            coefficients.push_back({i, j, val});
            sparsity_coeffs.push_back({i, j, 1});
          }
        }
      }
      _sp_weights.setFromTriplets(coefficients.begin(), coefficients.end());
      _sparsity.setFromTriplets(sparsity_coeffs.begin(),sparsity_coeffs.end());

      _sp_weights.makeCompressed();
      _sparsity.makeCompressed();

      is_sparse = true;
    }

    const EigenVector& RateSynapses::activation() {
      auto __compute_activation = [&](auto& __weights) {
        if (frontend()->scaling != 1)
          _activation = frontend()->scaling * _weights * neurons_pre->rate(_delay);
        else
          _activation = _weights * neurons_pre->rate(_delay);
      };

      if (is_sparse) {
        __compute_activation(_sp_weights);
      } else {
        __compute_activation(_weights);
      }

      return _activation;
    }

    void RateSynapses::delay(unsigned int d) {
      _delay = d;
      if (neurons_pre->_rate_history.cols() < (d+1))
        neurons_pre->_rate_history.resize
          (neurons_pre->_rate_history.rows(), d+1);
    }

    unsigned int RateSynapses::delay() {
      return _delay;
    }

    void RateSynapses::get_weights(EigenMatrix& output) {
      if (is_sparse) {
        output = _sp_weights;
      } else {
        output = _weights;
      }
    }

    void RateSynapses::weights(EigenMatrix const& w) {
      assert(!is_sparse);
      _weights = w;
    }

    void RatePlasticity::prepare() {
      synapses = dynamic_cast<::Backend::Eigen::RateSynapses*>
        (frontend()->synapses->backend());
      reset_state();
    }

    void RatePlasticity::reset_state() {
    }

    void RatePlasticity::apply_plasticity(FloatT dt) {
      if (!(frontend()->plasticity_schedule.size()))
        return;

      _curr_rate_t += dt;
      if (_curr_rate_t > frontend()->plasticity_schedule[_schedule_idx].first) {
        _curr_rate_t = _curr_rate_t - frontend()->plasticity_schedule[_schedule_idx].first;
        ++_schedule_idx;
        if (_schedule_idx >= frontend()->plasticity_schedule.size())
          _schedule_idx = 0;
      }

      epsilon = frontend()->plasticity_schedule[_schedule_idx].second;

      if (epsilon == 0)
        return;

      unsigned int delay = synapses->delay();

      auto dW
        = (dt * epsilon) * (synapses->neurons_post->rate()
                            * synapses->neurons_pre->rate(delay).transpose());

      if (synapses->is_sparse) {
        synapses->_sp_weights += synapses->_sparsity.cwiseProduct(dW);
        normalize_matrix_rows(synapses->_sp_weights);
      } else {
        synapses->_weights.noalias() += dW;
        normalize_matrix_rows(synapses->_weights);
      }
    }
  }
}
