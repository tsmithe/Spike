#include "RateModel.hpp"

SpikeException::SpikeException(std::string msg) : _msg(msg) {}
const char* SpikeException::what() const noexcept {
  return _msg.c_str();
}

BufferWriter::BufferWriter(const std::string& filename_, EigenBuffer& buf_)
  : buffer(buf_), filename(filename_) {
  file.open(filename, std::ofstream::out | std::ofstream::binary);
}

BufferWriter::~BufferWriter() {
  if (running)
    stop();
  if (othread.joinable())
    othread.join();
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

void BufferWriter::block_until_empty() const {
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
  if (othread.joinable())
    othread.join();
  file.flush();
}

RateNeurons::RateNeurons(Context* ctx, int size_,
                         std::string label_)
  : size(size_), label(label_) {
  init_backend(ctx);
  // reset_state();

  if (ctx->verbose) {
    std::cout << "Spike: Created RateNeurons with size "
              << size << " and label '" << label << "'.\n";
  }
}

RateNeurons::~RateNeurons() {
}

void RateNeurons::reset_state() {
  timesteps = 0;
  rate_history.clear();
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
  dendrites.push_back({synapses, plasticity});
  backend()->connect_input(synapses->backend(), plasticity->backend());
}

void RateNeurons::update(FloatT dt) {
  // update_dendritic_activation(dt);
  update_rate(dt);
  apply_plasticity(dt);
}

void RateNeurons::update_rate(FloatT dt) {
  backend()->update_rate(dt);
  timesteps += 1;
  if (rate_buffer_interval && !(timesteps % rate_buffer_interval))
    rate_history.push_back(timesteps, rate());
}

const EigenVector& RateNeurons::rate() const {
  return backend()->rate();
}

void RateNeurons::update_dendritic_activation(FloatT dt) const {
  for (auto& dendrite_pair : dendrites)
    dendrite_pair.first->update_activation(dt);
}

void RateNeurons::apply_plasticity(FloatT dt) const {
  for (auto& dendrite_pair : dendrites)
    dendrite_pair.second->apply_plasticity(dt);
}

RateSynapses::RateSynapses(Context* ctx,
                           RateNeurons* neurons_pre_,
                           RateNeurons* neurons_post_,
                           std::string label_)
  : neurons_pre(neurons_pre_), neurons_post(neurons_post_), label(label_) {
  init_backend(ctx);
  // reset_state();
  if(!(label.length()))
    label = neurons_pre->label;

  if (ctx->verbose) {
    std::cout << "Spike: Created synapses '" << label << "' from "
              << neurons_pre->label << " to " << neurons_post->label << ".\n";
  }
}

RateSynapses::~RateSynapses() {
}

void RateSynapses::reset_state() {
  // TODO: reset_state should revert the network to the state at t=0
  // assert(false && "TODO: think about weights and resetting..."); // TODO!
  timesteps = 0;
  activation_history.clear();
  backend()->reset_state();
}

const EigenVector& RateSynapses::activation() const {
  return backend()->activation();
}

const EigenMatrix& RateSynapses::weights() const {
  return backend()->weights();
}

void RateSynapses::update_activation(FloatT dt) {
  backend()->update_activation(dt);
  timesteps += 1;
  if (activation_buffer_interval && !(timesteps % activation_buffer_interval))
    activation_history.push_back(timesteps, activation());
}

RatePlasticity::RatePlasticity(Context* ctx, RateSynapses* syns)
  : synapses(syns) {
  init_backend(ctx);
  // reset_state();

  if (ctx->verbose) {
    std::cout << "Spike: Created plasticity for " << syns->label << ".\n";
  }
}

RatePlasticity::~RatePlasticity() {
}

void RatePlasticity::reset_state() {
  timesteps = 0;
  weights_history.clear();
  backend()->reset_state();
}

void RatePlasticity::apply_plasticity(FloatT dt) {
  backend()->apply_plasticity(dt);
  timesteps += 1;
  if (weights_buffer_interval && !(timesteps % weights_buffer_interval))
    weights_history.push_back(timesteps, synapses->weights());
}

RateElectrodes::RateElectrodes(/*Context* ctx,*/ std::string prefix,
                               RateNeurons* neurons_)
  : output_prefix(prefix), neurons(neurons_) {

  // init_backend(ctx);

  {
    const int err = mkdir(output_prefix.c_str(),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == err && EEXIST != errno)
      std::cout << "\nTrouble making output directory "
                << output_prefix << "\n";
  }  

  output_dir = output_prefix + "/" + neurons->label;

  {
    const int err = mkdir(output_dir.c_str(),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == err && EEXIST != errno)
      std::cout << "\nTrouble making output directory "
                << output_dir << "\n";
  }

  std::string rate_fname = output_dir + "/rate.bin";
  writers.push_back
    (std::make_unique<BufferWriter>(rate_fname, neurons->rate_history));

  for (auto& d : neurons->dendrites) {
    auto& synapses = d.first;
    auto& plasticity = d.second;
    std::string activation_fname
      = output_dir + "/activation_" + synapses->label + ".bin";
    writers.push_back
      (std::make_unique<BufferWriter>
       (activation_fname, synapses->activation_history));
    std::string weights_fname
      = output_dir + "/weights_" + synapses->label + ".bin";
    writers.push_back
      (std::make_unique<BufferWriter>
       (weights_fname, plasticity->weights_history));
  }

  // reset_state();

  if (neurons->backend()->context->verbose) {
    std::cout << "Spike: Created electrodes for " << neurons->label
              << " writing to " << output_prefix << ".\n";
  }
}

RateElectrodes::~RateElectrodes() {
  stop();
}

void RateElectrodes::write_output_info() const {
  std::ofstream output_info_file(output_dir + "/output.info");
  output_info_file << "size = " << neurons->size << "\n"
                   << "rate_buffer_interval = "
                   << neurons->rate_buffer_interval << "\n";

  for (auto& d : neurons->dendrites) {
    auto& synapses = d.first;
    auto& plasticity = d.second;
    output_info_file << "[" << synapses->label << "]\n"
                     << "neurons_pre->size = "
                     << synapses->neurons_pre->size << "\n"
                     << "activation_buffer_interval = "
                     << synapses->activation_buffer_interval << "\n"
                     << "weights_buffer_interval = "
                     << plasticity->weights_buffer_interval << "\n";
  }

  output_info_file.close();
}

void RateElectrodes::reset_state() {
}

void RateElectrodes::start() const {
  for (auto& writer : writers) {
    writer->start();
  }
}

void RateElectrodes::stop() const {
  for (auto& writer : writers) {
    writer->stop();
  }
}

void RateElectrodes::block_until_empty() const {
  for (auto& writer : writers)
    writer->block_until_empty();
}

RateModel::RateModel(Context* ctx) {
  // Eigen::initParallel();
  if (ctx == nullptr) {
    Backend::init_global_context();
    context = Backend::get_current_context();
  }
}

RateModel::~RateModel() {
  if (running)
    stop();
  if (simulation_thread.joinable())
    simulation_thread.join();
}

void RateModel::add(RateNeurons* neurons) {
  // Ensure buffer intervals match those set here:
  if (rate_buffer_interval != 0)
    neurons->rate_buffer_interval = rate_buffer_interval;
  if (activation_buffer_interval != 0) {
    for (auto& d : neurons->dendrites) {
      auto& synapses = d.first;
      synapses->activation_buffer_interval = activation_buffer_interval;
    }
  }
  if (weights_buffer_interval != 0) {
    for (auto& d : neurons->dendrites) {
      auto& plasticity = d.second;
      plasticity->weights_buffer_interval = weights_buffer_interval;
    }
  }

  // Add neurons to model:
  neuron_groups.push_back(neurons);

  if (context->verbose) {
    std::cout << "Spike: Added neurons " << neurons->label << " to model.\n";
  }
}

void RateModel::add(RateElectrodes* elecs) {
  electrodes.push_back(elecs);

  if (context->verbose) {
    std::cout << "Spike: Added electrodes on "
              << elecs->neurons->label << " to model.\n";
  }
}

void RateModel::set_rate_buffer_interval(int n_timesteps) {
  rate_buffer_interval = n_timesteps;
  for (auto& n : neuron_groups)
    n->rate_buffer_interval = n_timesteps;

  if (context->verbose) {
    std::cout << "Spike: Rate buffer interval is "
              << n_timesteps << " timesteps.\n";
  }
}

void RateModel::set_activation_buffer_interval(int n_timesteps) {
  activation_buffer_interval = n_timesteps;
  for (auto& n : neuron_groups) {
    for (auto& d : n->dendrites) {
      auto& synapses = d.first;
      synapses->activation_buffer_interval = n_timesteps;
    }
  }

  if (context->verbose) {
    std::cout << "Spike: Activation buffer interval is "
              << n_timesteps << " timesteps.\n";
  }
}

void RateModel::set_weights_buffer_interval(int n_timesteps) {
  weights_buffer_interval = n_timesteps;
  for (auto& n : neuron_groups) {
    for (auto& d : n->dendrites) {
      auto& plasticity = d.second;
      plasticity->weights_buffer_interval = n_timesteps;
    }
  }

  if (context->verbose) {
    std::cout << "Spike: Weights buffer interval is "
              << n_timesteps << " timesteps.\n";
  }
}

void RateModel::set_buffer_intervals(int rate_timesteps,
                                     int activation_timesteps,
                                     int weights_timesteps) {
  set_rate_buffer_interval(rate_timesteps);
  set_activation_buffer_interval(activation_timesteps);
  set_weights_buffer_interval(weights_timesteps);
}

void RateModel::set_buffer_intervals(int n_timesteps) {
  set_buffer_intervals(n_timesteps, n_timesteps, n_timesteps);
}

void RateModel::set_buffer_intervals(FloatT intval_s) {
  if (dt == 0)
    throw SpikeException("Must set simulation dt first!");

  int n_timesteps = round(intval_s / dt);
  set_buffer_intervals(n_timesteps, n_timesteps, n_timesteps);
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

    // Print simulation time every 0.05s:
    if (!((timesteps * 20) % timesteps_per_second)) {
      printf("\r%.2f", t);
      std::cout.flush();
    }

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

  printf("\r%.1f\n", t);
  running = false;
}

void RateModel::wait_for_electrodes() const {
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
  t += dt;
  timesteps += 1;
}

void RateModel::set_simulation_time(FloatT t_stop_, FloatT dt_) {
  t_stop = t_stop_;
  dt = dt_;
  timesteps_per_second = round(1 / dt);

  if (context->verbose) {
    std::cout << "Spike: dt = " << dt << " seconds.\n"
              << "Spike: t_stop = " << t_stop << " seconds.\n"
              << "Spike: timesteps_per_second = "
              << timesteps_per_second << "\n";
  }
}

void RateModel::start(bool block) {
  if (running)
    return;

  if (t == 0) {
    if (context->verbose) {
      std::cout << "Spike: Starting simulation...\n";
    }
    reset_state();
    for (auto& e : electrodes)
      e->write_output_info();
  }

  // Start `recording' before simulation starts:
  for (auto& e : electrodes)
    e->start();

  running = true;
  simulation_thread = std::thread(&RateModel::simulation_loop, this);

  if (block)
    wait_for_simulation();
}

void RateModel::wait_for_simulation() const {
  while (running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void RateModel::stop() {
  if (!running)
    return;

  running = false;
  if (simulation_thread.joinable())
    simulation_thread.join();

  // Stop recording only once simulation is stopped:
  stop_electrodes();
}

void RateModel::stop_electrodes() const {
  for (auto& e : electrodes)
    e->stop();
}

SPIKE_MAKE_INIT_BACKEND(RateNeurons);
SPIKE_MAKE_INIT_BACKEND(RateSynapses);
SPIKE_MAKE_INIT_BACKEND(RatePlasticity);
// SPIKE_MAKE_INIT_BACKEND(RateElectrodes);
// SPIKE_MAKE_INIT_BACKEND(RateModel);
