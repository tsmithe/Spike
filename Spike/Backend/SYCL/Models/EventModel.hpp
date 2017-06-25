#pragma once

#include <CL/sycl.hpp>

#include "Spike/Models/EventModel.hpp"

namespace Backend {
  namespace SYCL {
    using namespace cl::sycl;

    class EventModel : public virtual ::Backend::EventModel {
    public:
      EventModel() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(EventModel);
      ~EventModel() override = default;

      void prepare() override;
      void reset_state() override;
    };

    class EventNeurons : public virtual ::Backend::EventNeurons {
    public:
      EventNeurons() = default;
      SPIKE_MAKE_BACKEND_CONSTRUCTOR(EventNeurons);
      ~EventNeurons() override = default;

      void prepare() override;
      void reset_state() override;

      /*
      void connect_input(::Backend::RateSynapses* synapses,
                         ::Backend::RatePlasticity* plasticity) override;

      bool staged_integrate_timestep(FloatT dt) override;

      template<typename T>
      inline T transfer(T const& total_activation);

      virtual EigenVector const& rate(unsigned int n_back);
      EigenVector const& rate() override;
      */

    private:
      /*
      std::vector<
        std::pair<::Backend::Eigen::RateSynapses*,
                  ::Backend::Eigen::RatePlasticity*> > _eigen_dendrites;
      */
    };
  } // namespace SYCL
} // namespace Backend

