#include "RateAgent.hpp"

SPIKE_EXPORT_BACKEND_TYPE(Eigen, AgentVISRateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Eigen, AgentHDRateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Eigen, AgentAHVRateNeurons);
SPIKE_EXPORT_BACKEND_TYPE(Eigen, AgentFVRateNeurons);

namespace Backend {
  namespace Eigen {
    void AgentVISRateNeurons::prepare() {
      t = 0;
      theta_pref.resize(frontend()->size);
      d.resize(frontend()->size);
      _rate.resize(frontend()->size);

      theta_pref = frontend()->theta_pref;

      sigma_IN_sqr = EigenVector::Constant(frontend()->size,
                                           frontend()->sigma_IN);
      sigma_IN_sqr = sigma_IN_sqr.cwiseProduct(sigma_IN_sqr);
    }

    void AgentVISRateNeurons::reset_state() {
    }

    void AgentVISRateNeurons::connect_input(::Backend::RateSynapses*,
                                            ::Backend::RatePlasticity*) {
      assert("You shouldn't be doing this" && false);
    }

    bool AgentVISRateNeurons::staged_integrate_timestep(FloatT dt) {
      AgentBase* agent = frontend()->agent;

      t += dt;
      dt_ = dt;

      if (t > frontend()->t_stop_after) {
        if (t < frontend()->t_stop_after + 10*dt) { // _rate.norm() > 0) {
          _rate = EigenVector::Zero(frontend()->size);
        }
        return true;
      }

      for (int i = 0; i < dynamic_cast<WorldBase*>(agent)->num_objects; ++i) {
        FloatT theta = agent->object_bearings(i);

        // EigenVector d_i_tmp = (theta_pref.array() - theta).abs().matrix();
        // d_i_tmp = d_i_tmp.array().min(2*M_PI - d_i_tmp.array());
        // d.segment(frontend()->neurons_per_object * i,
        //           frontend()->neurons_per_object) = d_i_tmp;

        auto d_i = d.segment(frontend()->neurons_per_object * i,
                             frontend()->neurons_per_object);
        d_i = (theta_pref.array() - theta).abs().matrix();
        d_i = d_i.array().min(2*M_PI - d_i.array());
      }

      _rate = frontend()->lambda *
        (-d.cwiseProduct(d).cwiseQuotient(2*sigma_IN_sqr)).array().exp().matrix();

      for (int i = 0; i < dynamic_cast<WorldBase*>(agent)->num_objects; ++i) {
        _rate.segment(frontend()->neurons_per_object * i,
                      frontend()->neurons_per_object) *= agent->object_visibility(i);
      }

      return true;
    }

    EigenVector const& AgentVISRateNeurons::rate() {
      return rate(0);
    }

    EigenVector const& AgentVISRateNeurons::rate(unsigned int n_back) {
      if (n_back)
        assert("Delays not yet supported here" && false);

      return _rate;
    }


    void AgentHDRateNeurons::prepare() {
      theta_pref.resize(frontend()->size);
      d.resize(frontend()->size);
      _rate.resize(frontend()->size);

      theta_pref = frontend()->theta_pref;

      sigma_IN_sqr = EigenVector::Constant(frontend()->size,
                                           frontend()->sigma_IN);
      sigma_IN_sqr = sigma_IN_sqr.cwiseProduct(sigma_IN_sqr);
    }

    void AgentHDRateNeurons::reset_state() {
    }

    void AgentHDRateNeurons::connect_input(::Backend::RateSynapses*,
                                            ::Backend::RatePlasticity*) {
      assert("You shouldn't be doing this" && false);
    }

    bool AgentHDRateNeurons::staged_integrate_timestep(FloatT dt) {
      AgentBase* agent = frontend()->agent;

      t += dt;
      dt_ = dt;

      if (t > frontend()->t_stop_after)
        return true;

      FloatT theta = agent->head_direction;
      theta -= 2*M_PI; // probably not necessary ...

      d = (theta_pref.array() - theta).abs().matrix();
      d = d.array().min(2*M_PI - d.array());

      return true;
    }

    EigenVector const& AgentHDRateNeurons::rate() {
      return rate(0);
    }

    EigenVector const& AgentHDRateNeurons::rate(unsigned int n_back) {
      if (n_back)
        assert("Delays not yet supported here" && false);

      if (t > frontend()->t_stop_after) {
        _rate = EigenVector::Zero(frontend()->size); 
        return _rate;
      }

      _rate = frontend()->lambda *
        (-d.cwiseProduct(d).cwiseQuotient(2*sigma_IN_sqr)).array().exp().matrix();
      return _rate;
    }


    void AgentAHVRateNeurons::prepare() {
      _rate.resize(frontend()->size);
    }

    void AgentAHVRateNeurons::reset_state() {
    }

    void AgentAHVRateNeurons::connect_input(::Backend::RateSynapses*,
                                            ::Backend::RatePlasticity*) {
      assert("You shouldn't be doing this" && false);
    }

    bool AgentAHVRateNeurons::staged_integrate_timestep(FloatT dt) {
      AgentBase* agent = frontend()->agent;

      if (AHV != agent->AHV) {
        AHV = agent->AHV;
        if (agent->smooth_AHV) {
          // std::cout << " .?. " << AHV << " -> " << agent->AHV; // << "    " << _rate.size();

          auto rate_sym = _rate.segment(0, frontend()->neurons_per_state);
          rate_sym = (frontend()->smooth_sym_base_rate.array()
                      + frontend()->smooth_sym_scale.array() * (frontend()->smooth_sym_slope * fabs(AHV)).array().tanh()).matrix();

          auto rate_asym_neg = _rate.segment(frontend()->neurons_per_state, frontend()->neurons_per_state);
          rate_asym_neg = frontend()->smooth_asym_neg_base_rate;
          if (AHV < 0) {
            rate_asym_neg += (frontend()->smooth_asym_neg_scale.array() * (frontend()->smooth_asym_neg_slope * (-AHV)).array().tanh()).matrix();
          }

          auto rate_asym_pos = _rate.segment(frontend()->neurons_per_state * 2, frontend()->neurons_per_state);
          rate_asym_pos = frontend()->smooth_asym_pos_base_rate;
          if (AHV > 0) {
            rate_asym_pos += (frontend()->smooth_asym_pos_scale.array() * (frontend()->smooth_asym_pos_slope * AHV).array().tanh()).matrix();
          }
        } else {
          _rate = EigenVector::Zero(frontend()->size);
          _rate.segment(frontend()->neurons_per_state * agent->curr_AHV,
                        frontend()->neurons_per_state)
            = EigenVector::Ones(frontend()->neurons_per_state);
          curr_AHV = agent->curr_AHV;
        }
      }

      return true;
    }

    EigenVector const& AgentAHVRateNeurons::rate() {
      return rate(0);
    }

    EigenVector const& AgentAHVRateNeurons::rate(unsigned int n_back) {
      if (n_back)
        assert("Delays not yet supported here" && false);

      return _rate;
    }



    void AgentFVRateNeurons::prepare() {
      _rate.resize(frontend()->size);
    }

    void AgentFVRateNeurons::reset_state() {
    }

    void AgentFVRateNeurons::connect_input(::Backend::RateSynapses*,
                                           ::Backend::RatePlasticity*) {
      assert("You shouldn't be doing this" && false);
    }

    bool AgentFVRateNeurons::staged_integrate_timestep(FloatT dt) {
      AgentBase* agent = frontend()->agent;

      if (curr_FV != agent->curr_FV) {
        _rate = EigenVector::Zero(frontend()->size);
        _rate.segment(frontend()->neurons_per_state * agent->curr_FV,
                      frontend()->neurons_per_state)
          = EigenVector::Ones(frontend()->neurons_per_state);
        curr_FV = agent->curr_FV;
      }

      return true;
    }

    EigenVector const& AgentFVRateNeurons::rate() {
      return rate(0);
    }

    EigenVector const& AgentFVRateNeurons::rate(unsigned int n_back) {
      if (n_back)
        assert("Delays not yet supported here" && false);

      return _rate;
    }
  } // Eigen
} // Backend
