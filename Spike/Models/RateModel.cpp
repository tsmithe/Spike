#include "RateModel.hpp"

BufferWriter::BufferWriter(const std::string& filename_, EigenBuffer& buf_)
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

RateNeurons::RateNeurons(Context* ctx, int size_) : size(size_) {
  init_backend(ctx);
  reset_state();
}

RateNeurons::~RateNeurons() {
}

void RateNeurons::reset_state() {
  rates = Eigen::VectorXf::Zero(size);
  backend()->reset_state();
}

void RateNeurons::assert_dendritic_consistency() const {
  for (auto& dendrite_pair : dendrites) {
    // Ensure that this set of neurons is post-synaptic:
    assert(dendrite_pair.first->neurons_post == this);
    // Ensure that plasticity is paired correctly with synapses:
    assert(dendrite_pair.second->synapses == dendrite_pair.first);
  }
}

void RateNeurons::connect_input(RateSynapses* synapses,
                                RatePlasticity* plasticity) {
  // TODO: think about best form for this ...
  dendrites.push_back(std::make_pair(synapses, plasticity));
  // TODO: what about backend? currently only prepared in prepare... !
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

RateSynapses::RateSynapses(Context* ctx,
                           RateNeurons* neurons_pre_,
                           RateNeurons* neurons_post_)
  : neurons_pre(neurons_pre_), neurons_post(neurons_post_) {
  init_backend(ctx);
  reset_state();
}

RateSynapses::~RateSynapses() {
}

void RateSynapses::reset_state() {
  activation = Eigen::VectorXf::Zero(neurons_post->size);
  weights = Eigen::MatrixXf::Zero(neurons_pre->size,
                                  neurons_post->size);
}

void RateSynapses::update_activation(float dt) {
  backend()->update_activation(dt);
}

RatePlasticity::RatePlasticity(Context* ctx, RateSynapses* syns)
  : synapses(syns) {
  init_backend(ctx);
  reset_state();
}

RatePlasticity::~RatePlasticity() {
}

void RatePlasticity::reset_state() {
}

void RatePlasticity::apply_plasticity(float dt) {
  backend()->apply_plasticity(dt);
}

RateElectrodes::RateElectrodes(Context* ctx, std::string prefix,
                               RateNeurons* neurons_)
  : output_prefix(prefix), neurons(neurons_) {
  init_backend(ctx);
  std::string tmp_fname = "TODO"; // TODO!
  writers.push_back
    (std::make_unique<BufferWriter>(tmp_fname, neurons->rates_history));
  for (auto& d : neurons->dendrites) {
    auto synapses = d.first;
    writers.push_back
      (std::make_unique<BufferWriter>
       (tmp_fname, synapses->activation_history));
    writers.push_back
      (std::make_unique<BufferWriter>(tmp_fname, synapses->weights_history));
  }
  reset_state();
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
