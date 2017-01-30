#pragma once

#include "Spike/Models/RateModel.hpp"

#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/linalg/matrix_operations.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/linalg/sum.hpp>

inline void normalize_matrix_rows(viennacl::matrix<FloatT>& R) {
  viennacl::vector<FloatT> inv_norms = viennacl::linalg::element_pow
    (viennacl::vector<FloatT>(viennacl::linalg::row_sum
                              (viennacl::linalg::element_prod(R,R))),
     viennacl::vector<FloatT>(viennacl::scalar_vector<FloatT>(R.size1(),-0.5)));
  R = viennacl::linalg::prod(viennacl::diag(inv_norms), R);
}

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

      // void update_activation(FloatT dt) override;
      // const EigenVector& activation() override;
      const EigenMatrix& weights() override;
      void weights(EigenMatrix const& w) override;

    private:
      // viennacl::vector<FloatT> _activation; // TODO: Need an explicit temporary?
      viennacl::vector<FloatT> activation();
      // EigenVector _activation_cpu;
      // int _activation_cpu_timestep = 0;

      viennacl::matrix<FloatT> _weights;    // TODO: Generalize synapse types
      EigenMatrix _weights_cpu;
      int _weights_cpu_timestep = 0;

      ::Backend::Vienna::RateNeurons* neurons_pre = nullptr;
      ::Backend::Vienna::RateNeurons* neurons_post = nullptr;
     };

    class RatePlasticity : public virtual ::Backend::RatePlasticity {
    public:
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(RatePlasticity);
      ~RatePlasticity() override = default;

      void prepare() override;
      void reset_state() override;

      void apply_plasticity(FloatT dt) override;

    private:
      ::Backend::Vienna::RateSynapses* synapses = nullptr;
    };

    class RateNeurons : public virtual ::Backend::RateNeurons {
      friend class RateSynapses;
      friend class RatePlasticity;
    public:
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

      FloatT _beta;
      FloatT _tau;

      viennacl::vector<FloatT> _total_activation;

      viennacl::vector<FloatT> _alpha;
      viennacl::vector<FloatT> _half;

      viennacl::vector<FloatT> _rate(unsigned int n_back=0);

      viennacl::matrix<FloatT> _rate_history;
      int _rate_hist_idx = 0;

      viennacl::vector<FloatT> _new_rate;

      EigenVector _rate_cpu;
      int _rate_cpu_timestep = 0;
      std::vector<
        std::pair<::Backend::Vienna::RateSynapses*,
                   ::Backend::Vienna::RatePlasticity*> > _vienna_dendrites;
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
  } // namespace Vienna
} // namespace Backend

