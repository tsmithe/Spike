#include "RateModel.hpp"

RateNeurons::RateNeurons() {
}

RateNeurons::~RateNeurons() {
}

void RateNeurons::reset_state() {
}

void RateNeurons::assert_dendritic_consistency() const {
  for (auto& dendrite_pair : dendrites) {
    // Ensure that this set of neurons is post-synaptic:
    assert(dendrite_pair.first->neurons_post == this);
    // Ensure that plasticity is paired correctly with synapses:
    assert(dendrite_pair.second->synapses == dendrite_pair.first);
  }
}

void RateNeurons::update(float dt) {
  update_dendritic_activation(dt);
  update_rate(dt);
  apply_plasticity(dt);
}

void RateNeurons::update_rate(float dt) {
  backend()->update_rate(dt);
}

void RateNeurons::update_dendritic_activation(float dt) {
  for (auto& dendrite_pair : dendrites)
    dendrite_pair.first->update_activation(dt);
}

void RateNeurons::apply_plasticity(float dt) {
  for (auto& dendrite_pair : dendrites)
    dendrite_pair.second->apply_plasticity(dt);
}

RateSynapses::RateSynapses() {
}

RateSynapses::~RateSynapses() {
}

void RateSynapses::reset_state() {
}

void RateSynapses::update_activation(float dt) {
  backend()->update_activation(dt);
}

RatePlasticity::RatePlasticity() {
}

RatePlasticity::~RatePlasticity() {
}

void RatePlasticity::reset_state() {
}

void RatePlasticity::apply_plasticity(float dt) {
  backend()->apply_plasticity(dt);
}

RateModel::RateModel() {
}

RateModel::~RateModel() {
}

void RateModel::reset_state() {
}

SPIKE_MAKE_INIT_BACKEND(RateNeurons);
SPIKE_MAKE_INIT_BACKEND(RateSynapses);
SPIKE_MAKE_INIT_BACKEND(RatePlasticity);
SPIKE_MAKE_INIT_BACKEND(RateModel);
