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

  // FloatT smooth_base_rate = 0.1;
  // FloatT smooth_slope = 1.0 / (1.5*M_PI);
  EigenVector smooth_sym_base_rate;
  EigenVector smooth_sym_scale;
  EigenVector smooth_sym_slope;

  EigenVector smooth_asym_pos_base_rate;
  EigenVector smooth_asym_pos_scale;
  EigenVector smooth_asym_pos_slope;

  EigenVector smooth_asym_neg_base_rate;
  EigenVector smooth_asym_neg_scale;
  EigenVector smooth_asym_neg_slope;

  inline void set_smooth_params(FloatT base, FloatT max, FloatT slope) {
    smooth_sym_base_rate = EigenVector::Constant(neurons_per_state, base);
    smooth_sym_scale = EigenVector::Constant(neurons_per_state, max - base);
    smooth_sym_slope = EigenVector::Constant(neurons_per_state, slope);

    smooth_asym_neg_base_rate = EigenVector::Constant(neurons_per_state, base);
    smooth_asym_neg_scale = EigenVector::Constant(neurons_per_state, max - base);
    smooth_asym_neg_slope = EigenVector::Constant(neurons_per_state, slope);

    smooth_asym_pos_base_rate = EigenVector::Constant(neurons_per_state, base);
    smooth_asym_pos_scale = EigenVector::Constant(neurons_per_state, max - base);
    smooth_asym_pos_slope = EigenVector::Constant(neurons_per_state, slope);
  }

  inline void set_smooth_params(EigenVector base_sym, EigenVector max_sym, EigenVector slope_sym,
                                EigenVector base_asym_neg, EigenVector max_asym_neg, EigenVector slope_asym_neg,
                                EigenVector base_asym_pos, EigenVector max_asym_pos, EigenVector slope_asym_pos) {
    // assert correct sizes
    smooth_sym_base_rate = base_sym;
    smooth_sym_scale = max_sym - base_sym;
    smooth_sym_slope = slope_sym;

    smooth_asym_neg_base_rate = base_asym_neg;
    smooth_asym_neg_scale = max_asym_neg - base_asym_neg;
    smooth_asym_neg_slope = slope_asym_neg;

    smooth_asym_pos_base_rate = base_asym_pos;
    smooth_asym_pos_scale = max_asym_pos - base_asym_pos;
    smooth_asym_pos_slope = slope_asym_pos;
  }

  inline void set_smooth_params(EigenVector base, EigenVector max, EigenVector slope) {
    // assert correct sizes
    smooth_sym_base_rate = base; smooth_asym_pos_base_rate = base; smooth_asym_neg_base_rate = base;
    smooth_sym_scale = max - base; smooth_asym_pos_scale = max - base; smooth_asym_neg_scale = max - base;
    smooth_sym_slope = slope; smooth_asym_pos_slope = slope; smooth_asym_neg_slope = slope;
  }

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

  int num_AHV_states = 0;
  int num_FV_states = 0;

  EigenVector2D position;
  FloatT head_direction = 0;

  EigenVector object_bearings;

  bool smooth_AHV = false;
  FloatT AHV_speed = 0;
  FloatT AHV, FV;
  FloatT AHV_change, FV_change;

  int curr_AHV = 0;
  // FloatT curr_AHV_speed = 0;
  int curr_FV = 0;

  enum struct actions_t { AHV, FV, STAY };
  actions_t curr_action = actions_t::FV;
  int choose_next_action_ts = 0;
  int change_action_ts = 0;
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

  std::vector<std::pair<FloatT, FloatT> > FVs;  //  FV, duration
  std::vector<std::pair<FloatT, FloatT> > AHVs; // AHV, duration
};


class WorldBase {
protected:
  virtual void prepare(AgentBase&) {}
  virtual void update_per_dt(AgentBase&, FloatT) {}

public:
  FloatT bound_x = 10, bound_y = 10;

  int num_objects = 0;
  int num_proximal_objects = 0;
  int num_distal_objects = 0;

  EigenMatrix2D proximal_objects;
  std::vector<FloatT> distal_objects;
};


class OpenWorld : public virtual WorldBase {
public:
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
  }

  void add_distal_object(FloatT angle) {
    // ensure angle is in range [0, 2pi)
    if (angle < 0) angle += 2*M_PI;
    if (angle > 2*M_PI) angle -= 2*M_PI;

    distal_objects.push_back(angle);
    num_distal_objects += 1;
    num_objects += 1;
  }
};


class MazeWorld : public virtual WorldBase {
  /* It is assumed that every line in map is of the same length */

  Eigen::Vector2i world_size; // {rows, cols}

  /* ray_t: {start_pos, stop_pos} */
  using ray_t = std::pair<Eigen::Vector2i, Eigen::Vector2i>;
  std::vector<ray_t> barriers;

protected:
  void prepare(AgentBase&) override {
    load_map(
"xxxxxxxxxxxxxxx\n"
"x    x   x    x\n"
"*    x   x    *\n"
"x    xx*xx    x\n"
"*             *\n"
"x    x*x*x    x\n"
"*    x   x    *\n"
"x    x   x    x\n"
"xxxxxxxxxxxxxxx");

    print_map();
  }

  static Eigen::Vector2i compute_size(std::string const& map) {
    std::string::size_type pos = 0;
    int rows = 0;
    int cols = 0;
    while(pos < map.size()) {
      ++rows;
      std::string::size_type new_pos = map.find('\n', pos);
      if (std::string::npos == new_pos) new_pos = map.size();
      int row_len = new_pos - pos;
      if (0 == row_len) break;
      if (cols > 0) assert(cols == row_len);
      cols = row_len;
      pos = new_pos + 1;
    }
    return {rows, cols};
  }

  static auto get_map_index(Eigen::Vector2i world_size) {
    return [world_size](Eigen::Vector2i coord) -> std::string::size_type {
      const unsigned row_len = world_size(1) + 1; // + 1 for the newlines
      return coord(0) * row_len + coord(1);
    };
  }

  static auto get_map_coord(Eigen::Vector2i world_size) {
    return [world_size](std::string::size_type pos) -> Eigen::Vector2i {
      const unsigned row_len = world_size(1) + 1; // + 1 for the newlines
      int row, col;
      col = pos % row_len;
      row = (pos - col) / row_len;
      return {row, col};
    };
  }

  static std::vector<ray_t> extract_barriers(std::string map, Eigen::Vector2i world_size) {
    const std::vector<Eigen::Vector2i> steps{{1, 0}, {0, 1}, {1, 1}};

    auto map_coord = get_map_coord(world_size);
    auto map_index = get_map_index(world_size);

    std::vector<ray_t> barriers;
    std::string::size_type idx = 0;
    while (true) {
      idx = map.find_first_of("x*");
      if (std::string::npos == idx) break;

      auto barrier_start = map_coord(idx);

      std::vector<std::pair<ray_t, Eigen::Vector2i> > candidates; // {ray, step}
      FloatT best_length = -1;
      std::size_t best_candidate = 0;
      for (auto const& step : steps) {
        auto barrier_stop = barrier_start;
        while (true) {
          barrier_stop += step;
          if (barrier_stop(0) >= world_size(0) || barrier_stop(1) >= world_size(1)) {
            barrier_stop -= step;
            break;
          }
          auto new_idx = map_index(barrier_stop);
          if (map[new_idx] != 'x' && map[new_idx] != '*') { // TODO: barrier symbols elsewhere
            barrier_stop -= step;
            break;
          }
        }
        EigenVector ray = (barrier_stop - barrier_start).cast<FloatT>();
        FloatT barrier_length = ray.norm();
        if (barrier_length > best_length) {
          best_candidate = candidates.size();
          best_length = barrier_length;
          candidates.push_back({{barrier_start, barrier_stop}, step});
        }
      }

      if (best_length >= -1) {
        barriers.push_back(candidates[best_candidate].first);

        // Now erase barrier from map copy:
        auto const& step = candidates[best_candidate].second;
        auto const& barrier_stop = candidates[best_candidate].first.second;
        while (barrier_start != barrier_stop + step) {
          idx = map_index(barrier_start);
          map[idx] = ' ';
          barrier_start += step;
        }
      }
    }

    return barriers;
  }

public:
  // void load_map(std::string filename);

  void load_map(std::string map) {
    trim(map);
    world_size = compute_size(map);
    barriers = extract_barriers(map, world_size);
  }

  void load_rewards(std::string map);

  void print_map() {
    std::string map((world_size(0)+1) * (world_size(1)+1), ' ');
    auto map_index = get_map_index(world_size);

    for (unsigned i = 0; i <= world_size(0); ++i) {
      map[map_index({i, world_size(1)})] = '\n';
    }

    for (auto const& barrier : barriers) {
      Eigen::Vector2i barrier_start = barrier.first;
      Eigen::Vector2i barrier_stop = barrier.second;
      auto ray = barrier_stop - barrier_start;
      Eigen::Vector2i step;
      if (ray(0) == 0) {
        step << 0, 1;
      } else if (ray(1) == 0) {
        step << 1, 0;
      } else {
        step << 1, 1;
      }
      while (barrier_start != barrier_stop + step) {
        auto idx = map_index(barrier_start);
        map[idx] = 'x';
        barrier_start += step;
      }
    }

    trim(map);
    std::cout << "\n" << map << std::endl;;
  }

  /*

    first, strip whitespace from edges

    map: "x" -> invisible barrier
         "*" -> barrier with visible object
         "o" -> visible object (no barrier)
              - else passable

    rewards:
    + first, define character mapping
    + second, load from map string
      - again, strip whitespace
      - ignore unmapped chars

    provide function to compute intersection / visibility of ray with any barrier

   */
};


template<typename WorldT>
class WorldAgentBase : public virtual AgentBase,
                       public virtual WorldT {
public:
  using world_type = WorldT;
};


class NullActionPolicy {
protected:
  inline void prepare(AgentBase&) {}
  inline void choose_new_action(AgentBase&, FloatT) {}
  inline void update_per_dt(AgentBase&, FloatT) {}
};


class RandomWalkPolicy {
  std::default_random_engine rand_engine;

  std::uniform_real_distribution<FloatT> action_die;
  std::uniform_int_distribution<> AHV_die;
  std::uniform_int_distribution<> FV_die;

  bool prepared = false;

protected:
  void prepare(AgentBase& a);
  inline void update_per_dt(AgentBase&, FloatT) {}
  void choose_new_action(AgentBase& a, FloatT dt);

public:
  FloatT p_fwd = 0.5; // TODO: should use setter...

  RandomWalkPolicy() {
    action_die = std::uniform_real_distribution<FloatT>(0, 1);
  }
};


template<typename AgentT>
class QMazePolicy {
  /*
   * Load a maze from a file
     - including reward values
   * During exploration, learn Q
   * Sequence actions according to softmax policy over Q]
     - set of actions is predefined: maps to sequences of primitive actions
       (in principle, this can be replace with a circuit mechanism, too)

   * In future, want to replace action selection with circuits
   * - need a mechanism!
   */

  /* get_state_index:
     - given state coordinates, return state index into Q
  */
  uint32_t get_state_index(/*coords*/);

protected:
  /* prepare:
     - compute action sequences from velocity data
       - at first, ignore smooth_ahv
     - construct initial Q matrix
       - uniform positive over reachable states;
       - uniform negative over unreachable states
       - OR LOAD FROM FILE?
   */
  void prepare(AgentT& a);

  /* update_per_dt:
     - update Q matrix
     - update visible objects (algorithm?) */
  void update_per_dt(AgentT& agent, FloatT dt);
  void update_Q(AgentT& agent, FloatT dt);

  /* update_visible_objects:
     - represent maze as set of line segments
       (we assume a MazeWorld agent instance, accessible using CRTP)
   */
  void update_visible_objects(AgentT& agent, FloatT dt);

  /* choose_new_action:
     - use Q matrix to choose probabilistically (softmax)
     - then sequence action primitives accordingly
       - add_velocity has already added the (hd, speed) pairs,
         so just need to do a rotation then a forward movement
   */
  void choose_new_action(AgentT& a, FloatT dt);

public:
  void add_velocity(FloatT theta, FloatT r);
  void buffer_q_interval(int timesteps);
};


class ScanWalkPolicy {
protected:
  FloatT bound_x{0}, bound_y{0};
  FloatT row_separation{0};
  bool prepared{false};
  int walk_direction{1};

  void prepare(AgentBase& a);
  virtual void choose_new_action(AgentBase& a, FloatT dt);

  inline void update_per_dt(AgentBase&, FloatT) {}
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
  inline void prepare(AgentBase&) {}
  void choose_new_action(AgentBase& a, FloatT dt);
  inline void update_per_dt(AgentBase&, FloatT) {}

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
  inline void prepare(AgentBase&) {}
  void choose_new_action(AgentBase& a, FloatT dt);
  inline void update_per_dt(AgentBase&, FloatT) {}

public:
  void add_test_position(FloatT x, FloatT y) {
    EigenVector2D pos;
    pos(0) = x;
    pos(1) = y;
    test_positions.push_back(pos);
  }
};


template<typename TrainPolicyT, typename TestPolicyT, typename WorldT>
class Agent : public virtual AgentBase,
              public virtual WorldT,
              public virtual WorldAgentBase<WorldT>,
              public virtual TrainPolicyT,
              public virtual TestPolicyT {
  bool prepared = false;

public:
  Agent() {
    this->position = EigenVector2D::Zero();

    add_FV(0, 0);
    add_AHV(0, 0);
  }

  void prepare() {
    if (prepared) return;

    WorldT::prepare(*this);

    object_bearings.resize(this->num_objects);
    object_bearings = EigenVector::Zero(this->num_objects);

    TrainPolicyT::prepare(*this);
    TestPolicyT::prepare(*this);

    prepared = true;
  }

  void set_smooth_AHV(FloatT speed) {
    AHV_speed = speed;
    if (fabs(AHV_speed) > 0) {
      smooth_AHV = true;
    } else {
      smooth_AHV = false;
    }
  }

  void add_test_time(FloatT t_test) {
    test_times.push(t_test);
  }

  void add_FV(FloatT FV, FloatT duration) {
    this->FVs.push_back({FV, duration});
    this->num_FV_states += 1;
  }

  void add_AHV(FloatT AHV, FloatT duration) {
    this->AHVs.push_back({AHV, duration});
    this->num_AHV_states += 1;
  }

  void update_bearings() {
    int i = 0;
    for (; i < this->num_distal_objects; ++i) {
      object_bearings(i) = this->distal_objects[i] - head_direction;
    }
    for (int j = 0; j < this->num_proximal_objects; ++i, ++j) {
      // There is possibly a neater way to do this, using the updates above...
      EigenVector2D obj_vector = this->proximal_objects.col(j) - position;
      object_bearings(i) = atan2(obj_vector(1), obj_vector(0)) - head_direction;
    }
    for (i = 0; i < this->num_objects; ++i) {
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
    if (!prepared) prepare();

    // buffer position & head_direction if necessary
    if (this->agent_buffer_interval
        && (this->timesteps >= this->agent_buffer_start)
        && !(this->timesteps % this->agent_buffer_interval)) {
      EigenVector agent_buf = EigenVector::Zero(6);
      agent_buf(0) = this->position(0);
      agent_buf(1) = this->position(1);
      agent_buf(2) = this->head_direction;
      agent_buf(3) = this->FV;
      agent_buf(4) = this->AHV;
      agent_buf(5) = this->t;
      this->agent_history.push_back(this->timesteps, agent_buf);
    }

    this->t += dt;
    this->timesteps += 1;

    WorldT::update_per_dt(*this, dt);
    TrainPolicyT::update_per_dt(*this, dt);
    TestPolicyT::update_per_dt(*this, dt);

    if (this->timesteps >= this->choose_next_action_ts) {
      if (this->test_times.empty() || this->t < this->test_times.top()) {
        TrainPolicyT::choose_new_action(*this, dt);
      } else {
        TestPolicyT::choose_new_action(*this, dt);
      }
      // In case the policy hasn't updated change_action_ts
      // (which is new, to support continuous changes in action):
      if (this->change_action_ts <= this->timesteps) {
        this->change_action_ts = this->timesteps + 1;
      }
    }

    if (this->timesteps < this->change_action_ts) {
      update_action(dt);
    }

    // ensure only one of AHV and FV is active currently:
    assert(!(this->curr_AHV && this->curr_FV));
    perform_action(dt);
    update_bearings();
  }

  void update_action(FloatT dt) {
    if (timesteps == change_action_ts - 1) {
      // then set action to target action
      if (curr_action == actions_t::AHV) {
        AHV = AHVs[curr_AHV].first;
        FV = 0;
      } else if (curr_action == actions_t::FV) {
        FV = FVs[curr_FV].first;
        AHV = 0;
      } else {
        assert(false);
      }
      change_action_ts = 0;
    } else {
      // then update action towards target action
      if (curr_action == actions_t::AHV) {
        AHV += AHV_change;
        FV = 0;
      } else if (curr_action == actions_t::FV) {
        FV += FV_change;
        AHV = 0;
      } else {
        assert(false);
      }
    }
  }

  void perform_action(FloatT dt) {
    if (curr_action == actions_t::STAY)
      return;

    if (timesteps == choose_next_action_ts - 1) {
      // if last timestep of action, just set angle / position to target
      //
      // TODO: SMOOTH TRANSITIONS BETWEEN ACTIONS
      if (curr_action == actions_t::AHV) {
        FV = 0;
        head_direction = target_head_direction;
        if (head_direction > 2*M_PI) {
          head_direction -= 2*M_PI;
        } else if (head_direction < 0) {
          head_direction += 2*M_PI;
        }
      } else if (curr_action == actions_t::FV) {
        AHV = 0;
        position = target_position;
      }
      // std::cout << "\n@ " << t << "   " << head_direction << "   " << AHV;
    } else {
      // otherwise, compute update
      if (curr_action == actions_t::AHV) {
        FV = 0;
        head_direction += AHV * dt;
        if (head_direction > 2*M_PI) {
          head_direction -= 2*M_PI;
        } else if (head_direction < 0) {
          head_direction += 2*M_PI;
        }
      } else if (curr_action == actions_t::FV) {
        AHV = 0;
        FloatT r = FV * dt;
        position(0) += r * cos(head_direction);
        position(1) += r * sin(head_direction);
      }
      // std::cout << "\n. " << t << "   " << head_direction << "   " << FV << "    " << AHV;
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
    map_file << this->bound_x << "," << this->bound_y << "\n";
    for (int i = 0; i < this->num_distal_objects; ++i) {
      map_file << this->distal_objects[i];
      if (i == this->num_distal_objects-1) {
        map_file << "\n";
      } else {
        map_file << ",";
      }
    }
    for (int i = 0; i < this->num_proximal_objects; ++i) {
      EigenVector2D obj_pos = this->proximal_objects.col(i);
      map_file << obj_pos(0) << "," << obj_pos(1) << "\n";
    }
    map_file.flush();
  }
};
