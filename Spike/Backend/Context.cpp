#include "Context.hpp"

Context* _global_ctx = nullptr;

namespace Backend {
  /* Set up default context */
  void init_global_context(Device default_device) {
    if (!_global_ctx)
      _global_ctx = new Context;
    if (!default_device) {
#ifdef SPIKE_WITH_CUDA
      _global_ctx->device = SPIKE_DEVICE_CUDA;
#else
      _global_ctx->device = SPIKE_DEVICE_DUMMY;
#endif
    } else {
      _global_ctx->device = default_device;
    }
  }

  Context* get_current_context() {
    return _global_ctx;
  }
}
