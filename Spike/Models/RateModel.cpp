#include "RateModel.hpp"

BufferWriter::BufferWriter(const std::string& filename_, EigenBuffer& buf_)
  : buffer(buf_), filename(filename_) {
  file.open(filename, std::ofstream::out | std::ofstream::binary);
}

BufferWriter::~BufferWriter() {
  file.flush();
  file.close();
}

void BufferWriter::write_output() {
  buffer.lock.lock();
  int buffer_size = buffer.buf.size();
  buffer.lock.unlock();

  while (buffer_size > 0) {
    auto& front = buffer.buf.front();
    // int timestep = front.first; // TODO: perhaps write this out, too?

    auto data = front.second.data();
    int n_bytes = front.second.size() * sizeof(decltype(front.second)::Scalar);

    file.write((char*) data, n_bytes);

    buffer.lock.lock();
    buffer.buf.pop_front();
    buffer_size = buffer.buf.size();
    buffer.lock.unlock();
  }
}

void BufferWriter::write_loop() {
  while (running) {
    // TODO: why 200ms?
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    write_output();
  }
}

void BufferWriter::start() {
  if (running)
    return;

  running = true;
  othread = std::thread(&BufferWriter::write_loop, this);
}

void BufferWriter::stop() {
  if (!running)
    return;

  running = false;
  othread.join();
  file.flush();
}

RateNeurons::RateNeurons(Context* ctx, int size_,
                         std::string label_)
  : size(size_), label(label_) {
  init_backend(ctx);
  reset_state();
}

RateNeurons::~RateNeurons() {
}

void RateNeurons::reset_state() {
  rates = Eigen::VectorXf::Zero(size);
  backend()->reset_state();
}

void RateNeurons::assert_dendritic_consistency
(RateSynapses* synapses, RatePlasticity* plasticity) const {
  // Ensure that this set of neurons is post-synaptic:
  assert(synapses->neurons_post == this);
  // Ensure that plasticity is paired correctly with synapses:
  assert(plasticity->synapses == synapses);
}

void RateNeurons::assert_dendritic_consistency() const {
  for (auto& dendrite_pair : dendrites)
    assert_dendritic_consistency(dendrite_pair.first, dendrite_pair.second);
}

void RateNeurons::connect_input(RateSynapses* synapses,
                                RatePlasticity* plasticity) {
  assert_dendritic_consistency(synapses, plasticity);
  // Connect the synapses to the dendrites:
  dendrites.push_back(std::make_pair(synapses, plasticity));
  backend()->connect_input(synapses->backend(), plasticity->backend());
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
                           RateNeurons* neurons_post_,
                           std::string label_)
  : neurons_pre(neurons_pre_), neurons_post(neurons_post_), label(label_) {
  init_backend(ctx);
  reset_state();
  if(!(label.length()))
    label = neurons_pre->label;
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

  std::string dirname = output_prefix + "/" + neurons->label;
  const int err = mkdir(dirname.c_str(),
                        S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  if (-1 == err && EEXIST != errno)
    std::cout << "\nTrouble making output directory "
                << dirname << "\n";

  std::ofstream output_info_file(dirname + "/output.info");
  output_info_file << "size = " << neurons->size << "\n"
                   << "rates_buffer_interval = "
                   << neurons->rates_buffer_interval << "\n";

  std::string rates_fname = dirname + "/rates.bin";
  writers.push_back
    (std::make_unique<BufferWriter>(rates_fname, neurons->rates_history));

  for (auto& d : neurons->dendrites) {
    auto synapses = d.first;
    output_info_file << "[" << synapses->label << "]\n"
                     << "neurons_pre->size = "
                     << synapses->neurons_pre->size << "\n"
                     << "activation_buffer_interval = "
                     << synapses->activation_buffer_interval << "\n"
                     << "weights_buffer_interval = "
                     << synapses->weights_buffer_interval << "\n";
    std::string activation_fname
      = dirname + "/activation_" + synapses->label + ".bin";
    writers.push_back
      (std::make_unique<BufferWriter>
       (activation_fname, synapses->activation_history));
    std::string weights_fname
      = dirname + "/weights_" + synapses->label + ".bin";
    writers.push_back
      (std::make_unique<BufferWriter>
       (weights_fname, synapses->weights_history));
  }
  output_info_file.close();

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
