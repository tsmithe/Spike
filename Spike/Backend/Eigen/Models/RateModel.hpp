#pragma once

#include "Spike/Models/RateModel.hpp"

// #include <viennacl/vector.hpp>
// #include <viennacl/matrix.hpp>
// #include <viennacl/compressed_matrix.hpp>
// #include <viennacl/linalg/matrix_operations.hpp>
// #include <viennacl/linalg/norm_2.hpp>
// #include <viennacl/linalg/prod.hpp>
// #include <viennacl/linalg/sum.hpp>

/*
template<class T>
inline bool isanynan(viennacl::vector<T> v) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> v_(v.size());
  viennacl::copy(v, v_);
  return v_.array().isNaN().any();
}

template<class T>
inline bool isanynan(viennacl::matrix<T> v) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> v_(v.size1(), v.size2());
  viennacl::copy(v, v_);
  return v_.array().isNaN().any();
}

inline void normalize_matrix_rows(viennacl::matrix<FloatT>& R) {
  viennacl::vector<FloatT> ones = viennacl::scalar_vector<FloatT>(R.size1(), 1);
  viennacl::vector<FloatT> inv_norms =
    viennacl::linalg::row_sum(viennacl::linalg::element_prod(R,R));
  inv_norms = viennacl::linalg::element_sqrt(inv_norms);
  inv_norms = viennacl::linalg::element_div(ones, inv_norms + (1e-4)*ones);
  /*
  if (isanynan(inv_norms)) {
    std::cout << "\n" << inv_norms << "\n!!!!!!\n";
    assert(false);
  }
  * /
  R = viennacl::linalg::prod(viennacl::diag(inv_norms), R);
}
*/

namespace Backend {
  namespace Eigen {
    class RateNeurons;    // forward

    class RateSynapses : public virtual ::Backend::RateSynapses {
      friend class RateNeurons;
      friend class RatePlasticity;
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateSynapses);
      ~RateSynapses() override = default;

      void prepare() override;
      void reset_state() override;

      // void update_activation(FloatT dt) override;
      const EigenVector& activation() override;
      const EigenMatrix& weights() override;
      void weights(EigenMatrix const& w) override;

      unsigned int delay() override;
      void delay(unsigned int d) override;

    private:
      unsigned int _delay = 0;

      // viennacl::vector<FloatT> _activation; // TODO: Need an explicit temporary?
      EigenVector _activation();
      EigenVector _activation_cpu;
      int _activation_cpu_timestep = 0;

      EigenMatrix _weights;    // TODO: Generalize synapse types
      EigenMatrix _weights_cpu;
      int _weights_cpu_timestep = 0;

      ::Backend::Eigen::RateNeurons* neurons_pre = nullptr;
      ::Backend::Eigen::RateNeurons* neurons_post = nullptr;
     };

    class RatePlasticity : public virtual ::Backend::RatePlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RatePlasticity);
      ~RatePlasticity() override = default;

      void prepare() override;
      void reset_state() override;

      void apply_plasticity(FloatT dt) override;

      void multipliers(EigenMatrix const& m) override;

    private:
      ::Backend::Eigen::RateSynapses* synapses = nullptr;
      FloatT epsilon = 0;

      EigenMatrix _multipliers;
      bool _using_multipliers = false;
    };

    class RateNeurons : public virtual ::Backend::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
    public:
      RateNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateNeurons);
      ~RateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses* synapses,
                         ::Backend::RatePlasticity* plasticity) override;

      bool staged_integrate_timestep(FloatT dt) override;

      template<typename T>
      inline T transfer(T const& total_activation);

      const EigenVector& rate() override;

    private:
      bool done_timestep = false;

      virtual EigenVector _rate(unsigned int n_back=0);

      FloatT _beta;
      FloatT _tau;

      EigenVector _total_activation;

      EigenVector _alpha;
      EigenVector _half;
      EigenVector _ones;

      EigenMatrix _rate_history;
      int _rate_hist_idx = 0;

      EigenVector _new_rate;

      EigenVector _rate_cpu;
      int _rate_cpu_timestep = 0;
      std::vector<
        std::pair<::Backend::Eigen::RateSynapses*,
                  ::Backend::Eigen::RatePlasticity*> > _eigen_dendrites;
    };

    class DummyRateNeurons : public virtual ::Backend::DummyRateNeurons,
                             public virtual ::Backend::Eigen::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
    public:
      DummyRateNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(DummyRateNeurons);
      ~DummyRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      void connect_input(::Backend::RateSynapses*,
                         ::Backend::RatePlasticity*) override;

      bool staged_integrate_timestep(FloatT dt) override;
      const EigenVector& rate() override;

      void add_rate(FloatT duration, EigenVector rates) override;

    protected:
      FloatT t, dt_;

    private:
      EigenVector _rate(unsigned int n_back=0) override;

      std::vector<std::pair<FloatT, EigenVector> > _rate_schedule;
      int _schedule_idx = 0;
      FloatT _curr_rate_t;
    };

    class InputDummyRateNeurons
      : public virtual ::Backend::InputDummyRateNeurons,
        protected virtual ::Backend::Eigen::DummyRateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(InputDummyRateNeurons);
      ~InputDummyRateNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      bool staged_integrate_timestep(FloatT dt) override;
      const EigenVector& rate() override;

    private:
      EigenVector _rate(unsigned int n_back=0) override;
      EigenVector _rate_cpu;

      FloatT theta;

      EigenVector sigma_IN_sqr;

      EigenVector theta_pref;
      EigenVector d;
    };

    /*
    class RateElectrodes : public virtual ::Backend::RateElectrodes {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateElectrodes);
      ~RateElectrodes() override = default;

      void prepare() override;
      void reset_state() override;
    };

    class RateModel : public virtual ::Backend::RateModel {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RateModel);
      ~RateModel() override = default;

      void prepare() override;
      void reset_state() override;
    };
    */
  } // namespace Eigen
} // namespace Backend

