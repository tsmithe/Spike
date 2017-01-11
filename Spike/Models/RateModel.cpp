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
  while (buffer.size() > 0) {
    auto& front = buffer.buf.front();
    // int timestep = front.first; // TODO: perhaps write this out, too?

    auto data = front.second.data();
    int n_bytes = front.second.size() * sizeof(decltype(front.second)::Scalar);

    file.write((char*) data, n_bytes);

    buffer.lock.lock();
    buffer.buf.pop_front();
    buffer.lock.unlock();
  }
}

void BufferWriter::block_until_empty() {
  while (buffer.size() > 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
  timesteps = 0;
  rates_history.clear();
  backend()->reset_state();
  for (auto& d : dendrites) {
    d.first->reset_state(); // TODO: should the weights be zeroed here?!
    d.second->reset_state();
  }
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
  // TODO: reset_state should revert the network to the state at t=0
  assert(false && "TODO: think about weights and resetting..."); // TODO!
  weights = Eigen::MatrixXf::Zero(neurons_pre->size,
                                  neurons_post->size);
  timesteps = 0;
  activation_history.clear();
  weights_history.clear();
  backend()->reset_state();
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
  backend()->reset_state();
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
  for (auto& writer : writers) {
    writer->start();
  }
}

void RateElectrodes::stop() {
  for (auto& writer : writers) {
    writer->stop();
  }
}

void RateElectrodes::block_until_empty() {
  for (auto& writer : writers)
    writer->block_until_empty();
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

void RateModel::set_dump_trigger(bool* trigger) {
  dump_trigger = trigger;
}

void RateModel::set_stop_trigger(bool* trigger) {
  stop_trigger = trigger;
}

void RateModel::reset_state() {
  for (auto& n : neuron_groups)
    n->reset_state();
  for (auto& e : electrodes)
    e->reset_state();
  t = 0;
}

void RateModel::simulation_loop() {
  while (running && t < t_stop) {
    update_model_per_dt();

    if (stop_trigger) {
      if (*stop_trigger)
        stop();
    }

    if (dump_trigger) {
      if (*dump_trigger)
        wait_for_electrodes();
    }
  }

  // Stop electrodes before declaring simulation done
  // (so as to block the program from exiting prematurely):
  stop_electrodes();
  running = false;
}

void RateModel::wait_for_electrodes() {
  for (auto& e : electrodes)
    e->block_until_empty();
}

void RateModel::update_model_per_dt() {
  // TODO: Currently, neuron groups are updated in (some) sequence.
  //       But should they be updated all simultaneously?
  //
  //       The problem with using a sequence is that
  //          neuron_groups[2](t)
  //       sees
  //          neuron_groups[0](t+dt)
  //       when it should see
  //          neuron_groups[0](t)
  //       !!
  //
  //       However, if dt is small, does this matter? ...
  for (auto& n : neuron_groups)
    n->update(dt);
}

void RateModel::start() {
  if (running)
    return;

  // Start `recording' before simulation starts:
  for (auto& e : electrodes)
    e->start();

  running = true;
  simulation_thread = std::thread(&RateModel::simulation_loop, this);
}

void RateModel::stop() {
  if (!running)
    return;

  running = false;
  simulation_thread.join();

  // Stop recording only once simulation is stopped:
  stop_electrodes();
}

void RateModel::stop_electrodes() {
  for (auto& e : electrodes)
    e->stop();
}

SPIKE_MAKE_INIT_BACKEND(RateNeurons);
SPIKE_MAKE_INIT_BACKEND(RateSynapses);
SPIKE_MAKE_INIT_BACKEND(RatePlasticity);
SPIKE_MAKE_INIT_BACKEND(RateElectrodes);
// SPIKE_MAKE_INIT_BACKEND(RateModel);
