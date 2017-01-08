#include "RateModel.hpp"

RateNeurons::RateNeurons() {
}

RateNeurons::~RateNeurons() {
}

void RateNeurons::assert_dendritic_consistency() const {
  /*
    for d in dendrites:
      assert that d[0]->neurons_post = this
      assert that d[1]->synapses = d[0]
   */
}

RateSynapses::RateSynapses() {
}

RateSynapses::~RateSynapses() {
}

RatePlasticity::RatePlasticity() {
}

RatePlasticity::~RatePlasticity() {
}

RateModel::RateModel() {
}

RateModel::~RateModel() {
}

SPIKE_MAKE_INIT_BACKEND(RateNeurons);
SPIKE_MAKE_INIT_BACKEND(RateSynapses);
SPIKE_MAKE_INIT_BACKEND(RatePlasticity);
SPIKE_MAKE_INIT_BACKEND(RateModel);
