#include "RateModel.hpp"

BufferWriter::BufferWriter(const std::string& filename_, EigenBuffer* buf_)
  : buffer(buf_), filename(filename_) {
  file.open(filename);
}

BufferWriter::~BufferWriter() {
  file.flush();
  file.close();
}

void BufferWriter::write_buffer() {
}

void BufferWriter::write_loop() {
}

void BufferWriter::start() {
}

void BufferWriter::stop() {
}

RateNeurons::RateNeurons(Context* ctx) {
  init_backend(ctx);
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

RateSynapses::RateSynapses(Context* ctx) {
  init_backend(ctx);
}

RateSynapses::~RateSynapses() {
}

void RateSynapses::reset_state() {
}

void RateSynapses::update_activation(float dt) {
  backend()->update_activation(dt);
}

RatePlasticity::RatePlasticity(Context* ctx) {
  init_backend(ctx);
}

RatePlasticity::~RatePlasticity() {
}

void RatePlasticity::reset_state() {
}

void RatePlasticity::apply_plasticity(float dt) {
  backend()->apply_plasticity(dt);
}

RateElectrodes::RateElectrodes(Context* ctx) {
  init_backend(ctx);
}

RateElectrodes::~RateElectrodes() {
}

void RateElectrodes::reset_state() {
}

void RateElectrodes::start() {
}

void RateElectrodes::stop() {
}

RateModel::RateModel(Context* ctx) {
  Eigen::initParallel();
  if (ctx == nullptr) {
    Backend::init_global_context();
    context = Backend::get_current_context();
  }
}

RateModel::~RateModel() {
}

void RateModel::reset_state() {
}

void RateModel::simulation_loop() {
}

void RateModel::update_model_per_dt() {
}

void RateModel::start() {
}

void RateModel::stop() {
}

SPIKE_MAKE_INIT_BACKEND(RateNeurons);
SPIKE_MAKE_INIT_BACKEND(RateSynapses);
SPIKE_MAKE_INIT_BACKEND(RatePlasticity);
SPIKE_MAKE_INIT_BACKEND(RateElectrodes);
// SPIKE_MAKE_INIT_BACKEND(RateModel);
