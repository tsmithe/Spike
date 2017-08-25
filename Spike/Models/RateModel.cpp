#include "RateModel.hpp"

BufferWriter global_writer;

namespace Eigen {
std::mt19937 global_random_generator;
}

SpikeException::SpikeException(std::string msg) : _msg(msg) {}
const char* SpikeException::what() const noexcept {
  return _msg.c_str();
}

BufferWriter::~BufferWriter() {
  if (running)
    stop();
  if (othread.joinable())
    othread.join();
  // file.close();
}

void BufferWriter::add_buffer(EigenBuffer* buf) {
  buffers.push_back(buf);
}

void BufferWriter::write_output() {
  for (auto buffer : buffers) {
    while (buffer->size() > 0) {
      auto& front = buffer->front();
      // int timestep = front.first; // TODO: perhaps write this out, too?

      auto data = front.second.data();
      int n_bytes = front.second.size() * sizeof(decltype(front.second)::Scalar);

      buffer->file.write((char*) data, n_bytes);

      buffer->pop_front();
    }
  }
}

void BufferWriter::block_until_empty() const {
  for (auto buffer : buffers) {
    while (buffer->size() > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void BufferWriter::write_loop() {
  while (running) {
    // TODO: why 200ms?
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    write_output();
    time_since_last_flush += 0.2;
    if (time_since_last_flush > 10) {
      for (auto buffer : buffers) {
        buffer->file.flush();
      }
      time_since_last_flush = 0;
    }
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

  for (auto buffer : buffers) {
    buffer->file.flush();
  }
}

Agent::Agent() {
  position = EigenVector2D::Zero();

  add_FV(0, 0);
  add_AHV(0, 0);

  action_die = std::uniform_real_distribution<FloatT>(0, 1);
}

void Agent::seed(unsigned s) {
  rand_engine.seed(s);
}

/*
Agent::Agent(FloatT bound_x_, FloatT bound_y_, FloatT velocity_scaling_)
  : bound_x(bound_x_), bound_y(bound_y_), velocity_scaling(velocity_scaling_)
{
  position = Eigen::Matrix<FloatT, 2, 1>::Zero(); // TODO: is this line needed?
}
*/

void Agent::record_history(std::string output_prefix,
                           int buffer_interval, int buffer_start) {

  agent_buffer_interval = buffer_interval;
  agent_buffer_start = buffer_start;

  // TODO: Consolidate this by generalising RateElectrodes

  {
    const int err = mkdir(output_prefix.c_str(),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == err && EEXIST != errno)
      std::cout << "\nTrouble making output directory "
                << output_prefix << "\n";
  }

  std::string output_dir = output_prefix + "/Agent";

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

  std::string history_fname = output_dir + "/history.bin";
  agent_history.open(history_fname);
  global_writer.add_buffer(&agent_history);
}

void Agent::save_map(std::string output_prefix) {
  {
    const int err = mkdir(output_prefix.c_str(),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == err && EEXIST != errno)
      std::cout << "\nTrouble making output directory "
                << output_prefix << "\n";
  }

  std::string output_dir = output_prefix + "/Agent";

  {
    const int err = mkdir(output_dir.c_str(),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == err && EEXIST != errno)
      std::cout << "\nTrouble making output directory "
                << output_dir << "\n";
  }

  std::ofstream map_file(output_dir + "/map.info");
  map_file << bound_x << "," << bound_y << "\n";
  for (int i = 0; i < num_distal_objects; ++i) {
    map_file << distal_objects[i];
    if (i == num_distal_objects-1) {
      map_file << "\n";
    } else {
      map_file << ",";
    }
  }
  for (int i = 0; i < num_proximal_objects; ++i) {
    EigenVector2D obj_pos = proximal_objects.col(i);
    map_file << obj_pos(0) << "," << obj_pos(1) << "\n";
  }
  map_file.flush();
}

void Agent::set_boundary(FloatT bound_x_, FloatT bound_y_) {
  bound_x = bound_x_;
  bound_y = bound_y_;
}

void Agent::add_proximal_object(FloatT x, FloatT y) {
  num_objects += 1;
  num_proximal_objects += 1;
  if (num_proximal_objects == 1) {
    proximal_objects.resize(Eigen::NoChange, 1);
  } else {
    proximal_objects.conservativeResize(Eigen::NoChange, num_proximal_objects);
  }
  proximal_objects(0, num_proximal_objects-1) = x;
  proximal_objects(1, num_proximal_objects-1) = y;

  object_bearings.resize(num_objects);
  object_bearings = EigenVector::Zero(num_objects);
}

void Agent::add_distal_object(FloatT angle) {
  // ensure angle is in range [0, 2pi)
  if (angle < 0) angle += 2*M_PI;
  if (angle > 2*M_PI) angle -= 2*M_PI;

  distal_objects.push_back(angle);
  num_distal_objects += 1;
  num_objects += 1;

  object_bearings.resize(num_objects);
  object_bearings = EigenVector::Zero(num_objects);
}

void Agent::add_FV(FloatT FV, FloatT duration) {
  FVs.push_back({FV, duration});
  num_FV_states += 1;
  FV_die = std::uniform_int_distribution<>(0, num_FV_states-1);
}

void Agent::add_AHV(FloatT AHV, FloatT duration) {
  AHVs.push_back({AHV, duration});
  num_AHV_states += 1;
  AHV_die = std::uniform_int_distribution<>(0, num_AHV_states-1);
}

void Agent::update_per_dt(FloatT dt) {
  // buffer position & head_direction if necessary
  if (agent_buffer_interval
      && (timesteps >= agent_buffer_start)
      && !(timesteps % agent_buffer_interval)) {
    EigenVector agent_buf = EigenVector::Zero(3);
    agent_buf(0) = position(0);
    agent_buf(1) = position(1);
    agent_buf(2) = head_direction;
    agent_history.push_back(timesteps, agent_buf);
  }

  t += dt;
  timesteps += 1;

  if (false) {
    // Do testing ...
    return;
  }

  if (timesteps >= choose_next_action_ts) {
    choose_new_action(dt);
  } else {
    // make sure only one of AHV and FV is active currently:
    assert(!(curr_AHV && curr_FV));
    perform_action(dt);
  }

  update_bearings();
}

void Agent::update_bearings() {
  int i = 0;
  for (; i < num_distal_objects; ++i) {
    object_bearings(i) = distal_objects[i] - head_direction;
  }
  for (int j = 0; j < num_proximal_objects; ++i, ++j) {
    // There is possibly a neater way to do this, using the updates above...
    EigenVector2D obj_vector = proximal_objects.col(j) - position;
    object_bearings(i) = atan2(obj_vector(1), obj_vector(0)) - head_direction;
  }
  for (i = 0; i < num_objects; ++i) {
    FloatT angle = object_bearings(i);
    /* // For 270deg restricted visual field:
    if ((angle > 0.75*M_PI && angle < M_PI)
        || (angle > -M_PI && angle < -0.75*M_PI)) {
      angle = infinity<float>();
    } else*/
    if (angle < 0) {
      angle += 2*M_PI;
    }
    object_bearings(i) = angle;
  }
}

void Agent::perform_action(FloatT dt) {
  if (timesteps == choose_next_action_ts - 1) {
    // if last timestep of action, just set angle / position to target
    if (curr_action == AHV) {
      head_direction = target_head_direction;
      if (head_direction > 2*M_PI) {
        head_direction -= 2*M_PI;
      } else if (head_direction < 0) {
        head_direction += 2*M_PI;
      }
    } else {
      position = target_position;
    }
  } else {
    // otherwise, compute update
    if (curr_action == AHV) {
      FloatT AHV = AHVs[curr_AHV].first;
      head_direction += AHV * dt;
      if (head_direction > 2*M_PI) {
        head_direction -= 2*M_PI;
      } else if (head_direction < 0) {
        head_direction += 2*M_PI;
      }
    } else {
      FloatT FV = FVs[curr_FV].first;
      FloatT r = FV * dt;
      position(0) += r * cos(head_direction);
      position(1) += r * sin(head_direction);
    }
  }
}

void Agent::choose_new_action(FloatT dt) {
    // choose new action, ensuring legality
    // choice random with uniform distribution (for now)

    bool is_legal = false;
    FloatT duration;
    while (!is_legal) {
      if (action_die(rand_engine) > p_fwd) {
        curr_action = AHV;
        curr_FV = 0;
        curr_AHV = AHV_die(rand_engine);

        is_legal = true;

        duration = AHVs[curr_AHV].second;
        FloatT angle_change = AHVs[curr_AHV].first * AHVs[curr_AHV].second;
        target_head_direction = head_direction + angle_change;
      } else {
        curr_action = FV;
        curr_AHV = 0;
        curr_FV = FV_die(rand_engine);

        duration = FVs[curr_FV].second;
        FloatT FV = FVs[curr_FV].first;
        FloatT r = FV * duration;

        target_position = position;
        target_position(0) += r * cos(head_direction);
        target_position(1) += r * sin(head_direction);

        if ((fabs(target_position(0)) > bound_x)
            || (fabs(target_position(1)) > bound_y)) {
          is_legal = false;
        } else {
          is_legal = true;          
        }
      }
    }
    int timesteps_per_action = round(duration / dt);
    choose_next_action_ts = timesteps + timesteps_per_action;
}

/*
void Agent::connect_actor(RateNeurons* actor_) {
  actor = actor_;

  // NB: For now, the tuning of the actor is imposed,
  //     and assumes movement on a 2D plane.
  actor_tuning.resize(2, actor->size);
  FloatT angle = 0;
  for (int i = 0; i < actor->size; ++i) {
    Eigen::Matrix<FloatT, 2, 1> direction(2);
    direction(0) = cos(angle);
    direction(1) = sin(angle);
    actor_tuning.col(i) = direction;
    angle += 2 * M_PI / actor->size;
  }
}
*/

/*
void Agent::update_per_dt(FloatT dt) {
  /*
     + Turn actor into a velocity (assume a continuous state space)
       - this ~is the `output' of the RateModel
     + Set new position according to this velocity, dt, and the maze structure
     + Update the state vector according to the new position
       - NB this should probably be in AgentSenseRateNeurons
       - each element of the state vector measures `inverse distance' ??

     % NB: eventually, want to be able to represent self-embeddedness
   * /
  position.noalias() += (dt * velocity_scaling) * (actor_tuning * actor->rate());
  if (position(0) > bound_x)
    position(0) = bound_x;
  else if (position(0) < 0)
    position(0) = 0;

  if (position(1) > bound_y)
    position(1) = bound_y;
  else if (position(1) < 0)
    position(1) = 0;
}
*/

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
    if (rate_buffer_interval
        && (timesteps >= rate_buffer_start)
        && !(timesteps % rate_buffer_interval)) {
      rate_history.push_back(timesteps, rate());
    }
    // TODO: Best place for synapse activation buffering?
    for (auto& dendrite_pair : dendrites) {
      auto& syns = dendrite_pair.first;
      if (syns->activation_buffer_interval
          && (timesteps >= syns->activation_buffer_start)
          && !(timesteps % syns->activation_buffer_interval)) {
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

RandomDummyRateNeurons::RandomDummyRateNeurons
(Context* ctx, int size_, std::string label_)
  : DummyRateNeurons(nullptr, size_, label_),
    RateNeurons(nullptr, size_, label_, 4, 1, 1) {

  // TODO: Generalize / parameterize RandomDummyRateNeurons

  if (ctx)
    init_backend(ctx);
}

RandomDummyRateNeurons::~RandomDummyRateNeurons() {
}

/*
void RandomDummyRateNeurons::add_schedule(FloatT duration, FloatT revs_per_second) {
  revs_schedule.push_back({duration, revs_per_second});
}
*/

/*
AgentSenseRateNeurons::AgentSenseRateNeurons(Context* ctx,
                                             Agent* agent_, std::string label_)
  : RateNeurons(nullptr, ceil(agent_->bound_x * agent_->bound_y),
                label_, 0, 1, 1),
    agent(agent_) {

  if (ctx)
    init_backend(ctx);
}

AgentSenseRateNeurons::~AgentSenseRateNeurons() {
}
*/

AgentVISRateNeurons::AgentVISRateNeurons(Context* ctx,
                                         Agent* agent_,
                                         int neurons_per_object_,
                                         FloatT sigma_IN_,
                                         FloatT lambda_,
                                         std::string label_)
  : RateNeurons(nullptr, (int)(agent_->num_objects) * neurons_per_object_,
                label_, 0, 1, 1),
    agent(agent_),
    neurons_per_object(neurons_per_object_),
    sigma_IN(sigma_IN_), lambda(lambda_) {

  theta_pref = EigenVector::Zero(neurons_per_object);
  for (int j = 0; j < neurons_per_object; ++j) {
    theta_pref(j) = 2 * M_PI * j/neurons_per_object;
  }

  if (ctx)
    init_backend(ctx);
}

AgentVISRateNeurons::~AgentVISRateNeurons() {
}



AgentAHVRateNeurons::AgentAHVRateNeurons(Context* ctx,
                                         Agent* agent_,
                                         int neurons_per_state_,
                                         std::string label_)
  : RateNeurons(nullptr, (int)(agent_->num_AHV_states) * neurons_per_state_,
                label_, 0, 1, 1),
    agent(agent_),
    neurons_per_state(neurons_per_state_) {

  if (ctx)
    init_backend(ctx);
}

AgentAHVRateNeurons::~AgentAHVRateNeurons() {
}



AgentFVRateNeurons::AgentFVRateNeurons(Context* ctx,
                                       Agent* agent_,
                                       int neurons_per_state_,
                                       std::string label_)
  : RateNeurons(nullptr, (int)(agent_->num_FV_states) * neurons_per_state_,
                label_, 0, 1, 1),
    agent(agent_),
    neurons_per_state(neurons_per_state_) {

  if (ctx)
    init_backend(ctx);
}

AgentFVRateNeurons::~AgentFVRateNeurons() {
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
  if (weights_buffer_interval
      && (timesteps >= weights_buffer_start) && !(timesteps % weights_buffer_interval)) {
    EigenMatrix tmp_buffer;
    synapses->get_weights(tmp_buffer);
    weights_history.push_back(timesteps, tmp_buffer);
  }
}

BCMPlasticity::BCMPlasticity(Context* ctx, RateSynapses* syns)
  : RatePlasticity(ctx, syns) {

  init_backend(ctx);
  // reset_state();

  if (ctx->verbose) {
    std::cout << "Spike: Created BCM plasticity for " << syns->label << ".\n";
  }
}

BCMPlasticity::~BCMPlasticity() {
}

/*
void BCMPlasticity::reset_state() {
  timesteps = 0;
  weights_history.clear();
  backend()->reset_state();
}
*/

/*
void BCMPlasticity::apply_plasticity(FloatT dt) {
  backend()->apply_plasticity(dt);
  timesteps += 1;
  synapses->timesteps += 1; // TODO: Tidy this up
  // TODO: Is this the best place for the buffering?
  if (weights_buffer_interval
      && (timesteps >= weights_buffer_start) && !(timesteps % weights_buffer_interval)) {
    EigenMatrix tmp_buffer;
    synapses->get_weights(tmp_buffer);
    weights_history.push_back(timesteps, tmp_buffer);
  }
}
*/

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
  neurons->rate_history.open(rate_fname);
  global_writer.add_buffer(&(neurons->rate_history));

  for (auto& d : neurons->dendrites) {
    auto& synapses = d.first;
    auto& plasticity = d.second;
    
    std::string activation_fname
      = output_dir + "/activation_" + synapses->label + ".bin";
    synapses->activation_history.open(activation_fname);
    global_writer.add_buffer(&(synapses->activation_history));

    std::string weights_fname
      = output_dir + "/weights_" + synapses->label + ".bin";
    plasticity->weights_history.open(weights_fname);
    global_writer.add_buffer(&(plasticity->weights_history));
  }

  if (neurons->backend()->context->verbose) {
    std::cout << "Spike: Created electrodes for " << neurons->label
              << " writing to " << output_prefix << ".\n";
  }
}

RateElectrodes::~RateElectrodes() {
  // stop();
  std::string lock_fname = output_dir + "/simulation.lock";
  remove(lock_fname.c_str());
}

void RateElectrodes::write_output_info() const {
  std::ofstream output_info_file(output_dir + "/output.info");
  output_info_file << "size = " << neurons->size << "\n"
                   << "rate_buffer_interval = "
                   << neurons->rate_buffer_interval << "\n"
                   << "rate_buffer_start = "
                   << neurons->rate_buffer_start << "\n";

  for (auto& d : neurons->dendrites) {
    auto& synapses = d.first;
    auto& plasticity = d.second;
    output_info_file << "[" << synapses->label << "]\n"
                     << "neurons_pre->size = "
                     << synapses->neurons_pre->size << "\n"
                     << "activation_buffer_interval = "
                     << synapses->activation_buffer_interval << "\n"
                     << "activation_buffer_start = "
                     << synapses->activation_buffer_start << "\n"
                     << "weights_buffer_interval = "
                     << plasticity->weights_buffer_interval << "\n"
                     << "weights_buffer_start = "
                     << plasticity->weights_buffer_start << "\n";
  }

  output_info_file.close();
}

/*
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
*/

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
  neurons->rate_buffer_interval = rate_buffer_interval;
  neurons->rate_buffer_start = rate_buffer_start;
  
  for (auto& d : neurons->dendrites) {
    auto& synapses = d.first;
    synapses->activation_buffer_interval = activation_buffer_interval;
    synapses->activation_buffer_start = activation_buffer_start;
  }

  for (auto& d : neurons->dendrites) {
    auto& plasticity = d.second;
    plasticity->weights_buffer_interval = weights_buffer_interval;
    plasticity->weights_buffer_start = weights_buffer_start;
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

void RateModel::set_rate_buffer_start(int n_timesteps) {
  rate_buffer_start = n_timesteps;
  for (auto& n : neuron_groups)
    n->rate_buffer_start = n_timesteps;

  if (context->verbose) {
    std::cout << "Spike: Rate buffer starts after "
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

void RateModel::set_activation_buffer_start(int n_timesteps) {
  activation_buffer_start = n_timesteps;
  for (auto& n : neuron_groups) {
    for (auto& d : n->dendrites) {
      auto& synapses = d.first;
      synapses->activation_buffer_start = n_timesteps;
    }
  }

  if (context->verbose) {
    std::cout << "Spike: Activation buffer starts after "
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

void RateModel::set_weights_buffer_start(int n_timesteps) {
  weights_buffer_start = n_timesteps;
  for (auto& n : neuron_groups) {
    for (auto& d : n->dendrites) {
      auto& plasticity = d.second;
      plasticity->weights_buffer_start = n_timesteps;
    }
  }

  if (context->verbose) {
    std::cout << "Spike: Weights buffer starts after "
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

void RateModel::set_buffer_start(FloatT start_t) {
  if (dt == 0)
    throw SpikeException("Must set simulation dt first!");

  int n_timesteps = round(start_t / dt);
  set_rate_buffer_start(n_timesteps);
  set_activation_buffer_start(n_timesteps);
  set_weights_buffer_start(n_timesteps);
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
      if (agent)
        printf("\r%.2f: %.2f, %.2f, %.2f", t, agent->position(0), agent->position(1), (180/M_PI)*(agent->head_direction));
      else
        printf("\r%.2f", t);
      std::cout.flush();
    }

    if (stop_trigger) {
      if (*stop_trigger) {
        stop();
      }
    }

    if (dump_trigger) {
      if (*dump_trigger) {
        // wait_for_electrodes();
        global_writer.block_until_empty();
      }
    }
  }

  // Stop electrodes before declaring simulation done
  // (so as to block the program from exiting prematurely):
  // stop_electrodes();
  global_writer.block_until_empty();

  printf("\r%.1f\n", t);
  running = false;
}

/*
void RateModel::wait_for_electrodes() const {
  for (auto& e : electrodes)
    e->block_until_empty();
}
*/

void RateModel::update_model_per_dt() {
  if (agent)
    agent->update_per_dt(dt);

  int num_groups = neuron_groups.size();
  bool all_done = false;
  bool groups_done[num_groups];
  for (int i = 0; i < num_groups; ++i) {
    groups_done[i] = false;
  }

  // Loop through the neuron groups, computing the rate update in stages.
  // Stop when the update has been computed for each group.
  // This allows us to implement an arbitrary-order forwards integration
  // scheme, without the neuron groups becoming unsynchronized.
  while (!all_done) {
    #pragma omp parallel for schedule(nonmonotonic:dynamic)
    for (int i = 0; i < num_groups; ++i) {
      if (groups_done[i]) continue;
      auto& n = neuron_groups[i];
      groups_done[i] = n->staged_integrate_timestep(dt);
    }

    all_done = true;
    for (int i = 0; i < num_groups; ++i) {
      if (!groups_done[i]) {
        all_done = false;
        break;
      }
    }
  }

  #pragma omp parallel for schedule(nonmonotonic:dynamic)
  for (int i = 0; i < num_groups; ++i) {
    auto& n = neuron_groups[i];
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

  /*
  // Start `recording' before simulation starts:
  for (auto& e : electrodes)
    e->start();
  */
  global_writer.start();

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

  /*
  // Stop recording only once simulation is stopped:
  stop_electrodes();
  */
  global_writer.stop();
}

/*
void RateModel::stop_electrodes() const {
  for (auto& e : electrodes)
    e->stop();
}
*/

SPIKE_MAKE_INIT_BACKEND(RateNeurons);
SPIKE_MAKE_INIT_BACKEND(DummyRateNeurons);
SPIKE_MAKE_INIT_BACKEND(InputDummyRateNeurons);
SPIKE_MAKE_INIT_BACKEND(RandomDummyRateNeurons);
SPIKE_MAKE_INIT_BACKEND(AgentVISRateNeurons);
SPIKE_MAKE_INIT_BACKEND(AgentAHVRateNeurons);
SPIKE_MAKE_INIT_BACKEND(AgentFVRateNeurons);
SPIKE_MAKE_INIT_BACKEND(RateSynapses);
SPIKE_MAKE_INIT_BACKEND(RatePlasticity);
SPIKE_MAKE_INIT_BACKEND(BCMPlasticity);
