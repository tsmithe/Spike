#include "EventModel.hpp"

SPIKE_EXPORT_BACKEND_TYPE(SYCL, EventModel);
SPIKE_EXPORT_BACKEND_TYPE(SYCL, EventNeurons);

namespace Backend {
  namespace SYCL {
    void EventModel::prepare() {
    };

    void EventModel::reset_state() {
    };

    void EventNeurons::prepare() {
    };

    void EventNeurons::reset_state() {
    };
  }
}
