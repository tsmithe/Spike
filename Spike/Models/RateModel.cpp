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

void Agent::update_per_dt(FloatT dt) {
  assert("TODO" && false);
}

RateNeurons::RateNeurons(Context* ctx, int size_,
                         std::string label_,
                         FloatT alpha_, FloatT beta_, FloatT tau_)
  : size(size_), label(label_),
    alpha(alpha_), beta(beta_), tau(tau_) {

  // TODO: This is hacky!
  if (ctx == nullptr)
    return;

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

/* This returns true if the timestep update is complete,
   and false otherwise. If false, call repeatedly until true. */
bool RateNeurons::staged_integrate_timestep(FloatT dt) {
  bool res = backend()->staged_integrate_timestep(dt);

  // TODO: Is this the best place for the buffering?
  if (res) {
    timesteps += 1;
    if (rate_buffer_interval && !(timesteps % rate_buffer_interval))
      rate_history.push_back(timesteps, rate());
    // TODO: Best place for synapse activation buffering?
    for (auto& dendrite_pair : dendrites) {
      auto& syns = dendrite_pair.first;
      if (syns->activation_buffer_interval && !(timesteps % syns->activation_buffer_interval)) {
        syns->activation_history.push_back(timesteps, syns->activation());
      }
    }
  }
  return res;
}

const EigenVector& RateNeurons::rate() const {
  return backend()->rate();
}

void RateNeurons::apply_plasticity(FloatT dt) const {
  for (auto& dendrite_pair : dendrites)
    dendrite_pair.second->apply_plasticity(dt);
}

DummyRateNeurons::DummyRateNeurons(Context* ctx, int size_, std::string label_)
  : RateNeurons(nullptr, size_, label_, 0, 1, 1) {
  if (ctx)
    init_backend(ctx);
}

DummyRateNeurons::~DummyRateNeurons() {
}

void DummyRateNeurons::add_schedule(FloatT duration, EigenVector const& rates) {
  backend()->add_schedule(duration, rates);
  rate_schedule.push_back({duration, rates});
}

InputDummyRateNeurons::InputDummyRateNeurons
(Context* ctx, int size_, std::string label_,
 FloatT sigma_IN_, FloatT lambda_)
  : DummyRateNeurons(nullptr, size_, label_),
    RateNeurons(nullptr, size_, label_, 0, 1, 1),
    sigma_IN(sigma_IN_) {

  lambda = lambda_;

  theta_pref = EigenVector::Zero(size);
  for (int j = 0; j < size; ++j)
    theta_pref(j) = 2 * M_PI * j/size;

  if (ctx)
    init_backend(ctx);
}

InputDummyRateNeurons::~InputDummyRateNeurons() {
}

void InputDummyRateNeurons::add_schedule(FloatT duration, FloatT revs_per_second) {
  revs_schedule.push_back({duration, revs_per_second});
}

RateSynapses::RateSynapses(Context* ctx,
                           RateNeurons* neurons_pre_,
                           RateNeurons* neurons_post_,
                           FloatT scaling_,
                           std::string label_)
  : neurons_pre(neurons_pre_), neurons_post(neurons_post_),
    scaling(scaling_), label(label_) {

  init_backend(ctx);
  // reset_state();
  if(!(label.length()))
    label = neurons_pre->label;

  if (ctx->verbose) {
    std::cout << "Spike: Created synapses '" << label
              << "' (at " << this <<  ") from "
              << neurons_pre->label << " to " << neurons_post->label << ".\n";
  }
}

RateSynapses::~RateSynapses() {
}

void RateSynapses::reset_state() {
  // TODO: reset_state should revert the network to the state at t=0
  // assert(false && "TODO: think about weights and resetting..."); // TODO!
  timesteps = 0;
  // activation_history.clear();
  backend()->reset_state();
}


const EigenVector& RateSynapses::activation() const {
  return backend()->activation();
}

void RateSynapses::get_weights(EigenMatrix& output) const {
  return backend()->get_weights(output);
}

void RateSynapses::weights(EigenMatrix const& w) {
  backend()->weights(w);
}

void RateSynapses::make_sparse() {
  backend()->make_sparse();
}

unsigned int RateSynapses::delay() const {
  return backend()->delay();
}

void RateSynapses::delay(unsigned int d) {
  backend()->delay(d);
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

void RatePlasticity::add_schedule(FloatT duration, FloatT eps) {
  plasticity_schedule.push_back({duration, eps});
}

void RatePlasticity::apply_plasticity(FloatT dt) {
  backend()->apply_plasticity(dt);
  timesteps += 1;
  synapses->timesteps += 1; // TODO: Tidy this up
  // TODO: Is this the best place for the buffering?
  if (weights_buffer_interval && !(timesteps % weights_buffer_interval)) {
    EigenMatrix tmp_buffer;
    synapses->get_weights(tmp_buffer);
    weights_history.push_back(timesteps, tmp_buffer);
  }
}

RateElectrodes::RateElectrodes(std::string prefix, RateNeurons* neurons_)
  : output_prefix(prefix), neurons(neurons_) {

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

  std::string lock_fname = output_dir + "/simulation.lock";
  if (file_exists(lock_fname))
    throw SpikeException("Lock file exists at " + lock_fname
                         + " -- do you already have a simulation running?");
  std::ofstream lock_file(lock_fname);
  if (!lock_file.good())
    throw SpikeException("Couldn't create lock file at " + lock_fname);
  lock_file << "Electrodes active!\n";

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

  if (neurons->backend()->context->verbose) {
    std::cout << "Spike: Created electrodes for " << neurons->label
              << " writing to " << output_prefix << ".\n";
  }
}

RateElectrodes::~RateElectrodes() {
  stop();
  std::string lock_fname = output_dir + "/simulation.lock";
  remove(lock_fname.c_str());
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

void RateModel::add(Agent* a) {
  if (agent && context->verbose)
    std::cout << "Spike: RateModel at " << this
              << " already associated with Agent at " << agent << " !\n";

  agent = a;

  if (context->verbose)
    std::cout << "Spike: RateModel " << this
              << " associated with Agent " << agent << "\n";
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
  if (agent)
    agent->update_per_dt(dt);

  std::vector<RateNeurons*> grps(neuron_groups);

  // Loop through the neuron groups, computing the rate update in stages.
  // Stop when the update has been computed for each group.
  // This allows us to implement an arbitrary-order forwards integration
  // scheme, without the neuron groups becoming unsynchronized.
  while (grps.size() > 0) {
    std::list<int> grps_done;
    for (int i = 0; i < grps.size(); ++i) {
      auto& n = grps[i];
      bool res = n->staged_integrate_timestep(dt);
      if (res)
        grps_done.push_front(i);
    }
    for (auto j : grps_done)
      grps.erase(grps.begin()+j);
  }

  for (auto& n : neuron_groups) {
    n->apply_plasticity(dt);
  }

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

void RateModel::wait_for_simulation() {
  simulation_thread.join();
  // while (running) {
  //   std::this_thread::sleep_for(std::chrono::seconds(1));
  // }
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
SPIKE_MAKE_INIT_BACKEND(DummyRateNeurons);
SPIKE_MAKE_INIT_BACKEND(InputDummyRateNeurons);
SPIKE_MAKE_INIT_BACKEND(RateSynapses);
SPIKE_MAKE_INIT_BACKEND(RatePlasticity);
