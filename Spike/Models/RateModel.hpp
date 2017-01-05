#pragma once

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

class RateModel; // Forward definition

namespace Backend {
  class RateModel : public virtual SpikeBackendBase {
  public:
    ~RateModel() override = default;
  };
}

static_assert(std::has_virtual_destructor<Backend::RateModel>::value,
              "contract violated");

#include "Spike/Backend/Dummy/Models/RateModel.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Models/RateModel.hpp"
#endif
#ifdef SPIKE_WITH_VIENNACL
#include "Spike/Backend/Vienna/Models/RateModel.hpp"
#endif

class RateModel : public virtual SpikeBase {
public:
  RateModel();
  ~RateModel() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(RateModel, SpikeBase);

  void reset_state() override {}

private:
  std::shared_ptr<::Backend::RateModel> _backend;
};
