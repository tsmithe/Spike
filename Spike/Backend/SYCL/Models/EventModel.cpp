#include "EventModel.hpp"

SPIKE_EXPORT_BACKEND_TYPE(SYCL, EventModel);
SPIKE_EXPORT_BACKEND_TYPE(SYCL, EventNeurons);
/*
SPIKE_EXPORT_BACKEND_TYPE(SYCL, EventSynapses);
SPIKE_EXPORT_BACKEND_TYPE(SYCL, EventPlasticity);
*/

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

    /*
    void EventSynapses::prepare() {
    };

    void EventSynapses::reset_state() {
    };

    void EventPlasticity::prepare() {
    };

    void EventPlasticity::reset_state() {
    };
    */
  }
}
