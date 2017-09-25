#pragma once

#include "Spike/Models/RateBase.hpp"
#include "Spike/Models/RateModel.hpp"

class AgentVISRateNeurons;
class AgentHDRateNeurons;
class AgentAHVRateNeurons;
class AgentFVRateNeurons;

namespace Backend {
  class AgentVISRateNeurons : public virtual RateNeurons {
  public:
    ~AgentVISRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(AgentVISRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    void connect_input(RateSynapses* synapses,
                       RatePlasticity* plasticity) override = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    const EigenVector& rate() override = 0;
  };

  class AgentHDRateNeurons : public virtual RateNeurons {
  public:
    ~AgentHDRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(AgentHDRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    void connect_input(RateSynapses* synapses,
                       RatePlasticity* plasticity) override = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    const EigenVector& rate() override = 0;
  };

  class AgentAHVRateNeurons : public virtual RateNeurons {
  public:
    ~AgentAHVRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(AgentAHVRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    void connect_input(RateSynapses* synapses,
                       RatePlasticity* plasticity) override = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    const EigenVector& rate() override = 0;
  };

  class AgentFVRateNeurons : public virtual RateNeurons {
  public:
    ~AgentFVRateNeurons() override = default;
    SPIKE_ADD_BACKEND_FACTORY(AgentFVRateNeurons);
    void prepare() override = 0;
    void reset_state() override = 0;
    void connect_input(RateSynapses* synapses,
                       RatePlasticity* plasticity) override = 0;
    bool staged_integrate_timestep(FloatT dt) override = 0;
    const EigenVector& rate() override = 0;
  };
}

class AgentVISRateNeurons : public virtual RateNeurons {
public:
  AgentVISRateNeurons(Context* ctx, AgentBase* agent_,
                      int neurons_per_object_,
                      FloatT sigma_IN_, FloatT lambda_,
                      std::string label_);
  ~AgentVISRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(AgentVISRateNeurons, RateNeurons);

  AgentBase* agent;
  int neurons_per_object;
  FloatT sigma_IN, lambda;

  EigenVector theta_pref;

  FloatT t_stop_after = infinity<FloatT>();

private:
  std::shared_ptr<::Backend::AgentVISRateNeurons> _backend;
};

class AgentHDRateNeurons : public virtual RateNeurons {
public:
  AgentHDRateNeurons(Context* ctx, AgentBase* agent_,
                     int size_,
                     FloatT sigma_IN_, FloatT lambda_,
                     std::string label_);
  ~AgentHDRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(AgentHDRateNeurons, RateNeurons);

  AgentBase* agent;

  FloatT sigma_IN, lambda;

  EigenVector theta_pref;

  FloatT t_stop_after = infinity<FloatT>();

private:
  std::shared_ptr<::Backend::AgentHDRateNeurons> _backend;
};

class AgentAHVRateNeurons : public virtual RateNeurons {
public:
  AgentAHVRateNeurons(Context* ctx, AgentBase* agent_,
                      int neurons_per_state_, std::string label_);
  ~AgentAHVRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(AgentAHVRateNeurons, RateNeurons);

  AgentBase* agent;
  int neurons_per_state;

private:
  std::shared_ptr<::Backend::AgentAHVRateNeurons> _backend;
};

class AgentFVRateNeurons : public virtual RateNeurons {
public:
  AgentFVRateNeurons(Context* ctx, AgentBase* agent_,
                     int neurons_per_state_, std::string label_);
  ~AgentFVRateNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(AgentFVRateNeurons, RateNeurons);

  AgentBase* agent;
  int neurons_per_state;

private:
  std::shared_ptr<::Backend::AgentFVRateNeurons> _backend;
};


typedef Eigen::Matrix<FloatT, 2, 1> EigenVector2D;
typedef Eigen::Matrix<FloatT, 2, Eigen::Dynamic> EigenMatrix2D;

struct AgentBase {
  // FloatT velocity_scaling = 1;

  FloatT bound_x = 10, bound_y = 10;

  int num_objects = 0;
  int num_proximal_objects = 0;
  int num_distal_objects = 0;

  int num_AHV_states = 0;
  int num_FV_states = 0;

  EigenVector2D position;
  FloatT head_direction = 0;

  EigenVector object_bearings;

  int curr_AHV = 0;
  int curr_FV = 0;

  enum struct actions_t { AHV, FV, STAY };
  actions_t curr_action = actions_t::FV;
  int choose_next_action_ts = 0;
  EigenVector2D target_position;
  FloatT target_head_direction = 0;

  std::priority_queue<FloatT, std::vector<FloatT>, std::greater<FloatT> > test_times;

  FloatT t = 0;
  int timesteps = 0;

  int agent_buffer_interval = 0;
  int agent_buffer_start = 0;
  EigenBuffer agent_history;
  std::unique_ptr<BufferWriter> history_writer;

  virtual void update_per_dt(FloatT dt) = 0;

  // RateNeurons* actor;
  // EigenMatrix2D actor_tuning;

  EigenMatrix2D proximal_objects;
  std::vector<FloatT> distal_objects;

  std::vector<std::pair<FloatT, FloatT> > FVs;  //  FV, duration
  std::vector<std::pair<FloatT, FloatT> > AHVs; // AHV, duration
};


class NullActionPolicy {
protected:
  inline void choose_new_action(AgentBase&, FloatT) {}
};


class RandomWalkPolicy {
  std::default_random_engine rand_engine;

  std::uniform_real_distribution<FloatT> action_die;
  std::uniform_int_distribution<> AHV_die;
  std::uniform_int_distribution<> FV_die;

  bool prepared = false;

  void prepare(AgentBase& a);

protected:
  void choose_new_action(AgentBase& a, FloatT dt);

public:
  FloatT p_fwd = 0.5; // TODO: should use setter...

  RandomWalkPolicy() {
    action_die = std::uniform_real_distribution<FloatT>(0, 1);
  }
};


class ScanWalkPolicy {
protected:
  FloatT bound_x{0}, bound_y{0};
  FloatT row_separation{0};
  bool prepared{false};
  int walk_direction{1};

  void prepare(AgentBase& a);
  virtual void choose_new_action(AgentBase& a, FloatT dt);
public:
  void set_scan_bounds(FloatT x, FloatT y);
  void set_row_separation(FloatT distance);
};


class ScanWalkTestPolicy : public ScanWalkPolicy {
  EigenVector2D old_position, old_target_pos;
  FloatT old_hd, old_target_hd;
  bool started{false};
  
protected:
  void choose_new_action(AgentBase& a, FloatT dt) override;
};


class PlaceTestPolicy {
  std::vector<EigenVector2D> test_positions;
  FloatT test_approach_radius = 1.0;
  int test_approach_angles = 8;
  // actions_t curr_test = actions_t::AHV;
  int curr_test_position = -1;
  int curr_test_approach_angle = -1;
  FloatT t_equilibration = 1.0;

protected:
  void choose_new_action(AgentBase& a, FloatT dt);

public:
  void add_test_position(FloatT x, FloatT y) {
    EigenVector2D pos;
    pos(0) = x;
    pos(1) = y;
    test_positions.push_back(pos);
  }

  void set_place_test_params(FloatT radius, FloatT num_directions) {
    test_approach_radius = radius;
    test_approach_angles = num_directions;
  }
};


class HDTestPolicy {
  EigenVector2D old_position, old_target_pos;
  FloatT old_hd, old_target_hd;

  std::vector<EigenVector2D> test_positions;
  // actions_t curr_test = actions_t::AHV;
  int curr_test_position = -1;
  FloatT t_equilibration = 1.0;

protected:
  void choose_new_action(AgentBase& a, FloatT dt);

public:
  void add_test_position(FloatT x, FloatT y) {
    EigenVector2D pos;
    pos(0) = x;
    pos(1) = y;
    test_positions.push_back(pos);
  }
};


template<typename TrainPolicyT, typename TestPolicyT>
class Agent : public AgentBase,
              public TrainPolicyT,
              public TestPolicyT {
public:
  Agent() {
    position = EigenVector2D::Zero();

    add_FV(0, 0);
    add_AHV(0, 0);
  }

  void set_boundary(FloatT bound_x_, FloatT bound_y_) {
    bound_x = bound_x_;
    bound_y = bound_y_;
  }

  void add_proximal_object(FloatT x, FloatT y) {
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

  void add_distal_object(FloatT angle) {
    // ensure angle is in range [0, 2pi)
    if (angle < 0) angle += 2*M_PI;
    if (angle > 2*M_PI) angle -= 2*M_PI;

    distal_objects.push_back(angle);
    num_distal_objects += 1;
    num_objects += 1;

    object_bearings.resize(num_objects);
    object_bearings = EigenVector::Zero(num_objects);
  }

  void add_test_time(FloatT t_test) {
    test_times.push(t_test);
  }

  void add_FV(FloatT FV, FloatT duration) {
    FVs.push_back({FV, duration});
    num_FV_states += 1;
  }

  void add_AHV(FloatT AHV, FloatT duration) {
    AHVs.push_back({AHV, duration});
    num_AHV_states += 1;
  }

  void update_bearings() {
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

  void update_per_dt(FloatT dt) override {
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

    if (timesteps >= choose_next_action_ts) {
      if (test_times.empty() || t < test_times.top()) {
        TrainPolicyT::choose_new_action(*this, dt);
      } else {
        TestPolicyT::choose_new_action(*this, dt);
      }
    }

    // ensure only one of AHV and FV is active currently:
    assert(!(curr_AHV && curr_FV));
    perform_action(dt);
    update_bearings();
  }

  void perform_action(FloatT dt) {
    if (curr_action == actions_t::STAY)
      return;

    if (timesteps == choose_next_action_ts - 1) {
      // if last timestep of action, just set angle / position to target
      if (curr_action == actions_t::AHV) {
        head_direction = target_head_direction;
        if (head_direction > 2*M_PI) {
          head_direction -= 2*M_PI;
        } else if (head_direction < 0) {
          head_direction += 2*M_PI;
        }
      } else if (curr_action == actions_t::FV) {
        position = target_position;
      }
    } else {
      // otherwise, compute update
      if (curr_action == actions_t::AHV) {
        FloatT AHV = AHVs[curr_AHV].first;
        head_direction += AHV * dt;
        if (head_direction > 2*M_PI) {
          head_direction -= 2*M_PI;
        } else if (head_direction < 0) {
          head_direction += 2*M_PI;
        }
      } else if (curr_action == actions_t::FV) {
        FloatT FV = FVs[curr_FV].first;
        FloatT r = FV * dt;
        position(0) += r * cos(head_direction);
        position(1) += r * sin(head_direction);
      }
    }
  }

  void record_history(std::string output_prefix,
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

  void save_map(std::string output_prefix) {
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
};
