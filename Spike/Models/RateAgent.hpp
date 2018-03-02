#pragma once

#include <unordered_map>

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
  AgentBase* agent = nullptr;

protected:
  template<typename T>
  void embed_agent(T& a) {
    agent = &a;
  }

  virtual void prepare() {}
  virtual void update_per_dt(FloatT) {}

public:
  FloatT bound_x = 10, bound_y = 10;

  int num_objects = 0;
  int num_proximal_objects = 0;
  int num_distal_objects = 0;

  EigenMatrix2D proximal_objects;
  std::vector<FloatT> distal_objects;

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

  std::default_random_engine rand_engine;

  bool prepared = false;

public:
  /* Coordinates in the following data are in the map frame {row, col}: */
  Eigen::Vector2i world_size;

  /* ray_t: {start_pos, stop_pos} */
  using ray_t = std::pair<Eigen::Vector2i, Eigen::Vector2i>;

  std::vector<ray_t> barriers;
  EigenMatrix2D test_locations;
  EigenMatrix2D start_locations;

  EigenMatrix reward;
  std::unordered_map<char, FloatT> reward_legend;

protected:
  void prepare() override {
    if (prepared) return;
    prepared = true;

    // std::cout << "\n" << map_to_string() << "\n"
    //           << "\n" << reward_to_string() << "\n" << std::endl;
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
    const std::vector<Eigen::Vector2i> steps{{1, 0}, {0, 1}/*, {1, 1}*/};

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
          bool is_vertex = false;
          for (auto const& other_step : steps) {
            if (other_step == step) {
              continue;
            }
            unsigned new_idx = map_index(barrier_start + other_step);
            if (new_idx < map.size()) {
              if ('x' == map[new_idx] || '*' == map[new_idx]) { // TODO: barrier symbols elsewhere
                is_vertex = true;
                break;
              }
            }
            // and the other way:
            new_idx = map_index(barrier_start - other_step);
            if (new_idx < map.size()) {
              if ('x' == map[new_idx] || '*' == map[new_idx]) { // TODO: barrier symbols elsewhere
                is_vertex = true;
                break;
              }
            }
          }
          if (!is_vertex) map[idx] = ' ';
          barrier_start += step;
        }
      }
    }

    return barriers;
  }

  static EigenMatrix2D extract_objects(std::string map, Eigen::Vector2i world_size,
                                       std::string obj_chars) {
    auto map_coord = get_map_coord(world_size);
    auto map_index = get_map_index(world_size);

    unsigned num_objects = 0;
    EigenMatrix2D objects;

    while (true) {
      auto idx = map.find_first_of(obj_chars);
      if (std::string::npos == idx) break;

      ++num_objects;
      if (num_objects == 1) {
        objects.resize(Eigen::NoChange, 1);
      } else {
        objects.conservativeResize(Eigen::NoChange, num_objects);
      }

      auto obj_coord = map_coord(idx).cast<FloatT>();
      objects.col(num_objects-1) = obj_coord.transpose();

      map[idx] = ' ';
    }

    return objects;
  }

public:
  EigenVector2D ij_to_xy(EigenVector2D ij) {
    return {ij(1), static_cast<float>(world_size(0)) - ij(0)};
  }

  EigenVector2D xy_to_ij(EigenVector2D xy) {
    return {static_cast<float>(world_size(0)) - xy(1), xy(0)};
  }

  void load_map(std::string map) {
    trim(map);
    world_size = compute_size(map);

    // In the old code, the origin is at the centre of the map:
    bound_y = world_size.cast<FloatT>()(0) + 1; // nb axes are swapped
    bound_x = world_size.cast<FloatT>()(1) + 1;

    barriers = extract_barriers(map, world_size);
    test_locations = extract_objects(map, world_size, "t");
    start_locations = extract_objects(map, world_size, "s");
    if (0 == start_locations.cols()) {
      start_locations.resize(Eigen::NoChange, 1);
      start_locations(0) = world_size.cast<FloatT>()(0) / 2;
      start_locations(1) = world_size.cast<FloatT>()(1) / 2;
    }

    proximal_objects = extract_objects(map, world_size, "*o");
    // Now, proximal_objects should be in xy basis:
    for (unsigned j = 0; j < proximal_objects.cols(); ++j) {
      proximal_objects.col(j) = ij_to_xy(proximal_objects.col(j));
    }

    num_proximal_objects = proximal_objects.cols();
    num_distal_objects = 0;
    num_objects = num_proximal_objects + num_distal_objects;
  }

  void load_rewards(std::string map, std::unordered_map<char, FloatT> legend) {
    trim(map);
    assert(compute_size(map) == world_size);

    reward = EigenMatrix::Zero(world_size(0), world_size(1));
    reward_legend = legend;

    auto map_coord = get_map_coord(world_size);
    for (auto const& r : legend) {
      while (true) {
        auto idx = map.find(r.first);
        if (std::string::npos == idx) break;
        auto coord = map_coord(idx);
        reward(coord(0), coord(1)) = r.second;
        map[idx] = ' ';
      }
    }
  }

  EigenVector2D propose_start_location_ij() {
    return start_locations.col(std::uniform_int_distribution<unsigned>{0, start_locations.cols()-1}(rand_engine));
  }

  EigenVector2D propose_start_location_xy() {
    return ij_to_xy(propose_start_location_ij());
  }

  std::string barriers_to_string() {
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

    return map;
  }

  std::string map_to_string() {
    std::string map = barriers_to_string();
    auto map_index = get_map_index(world_size);

    for (unsigned i = 0; i < proximal_objects.cols(); ++i) {
      Eigen::Vector2i coord = xy_to_ij(proximal_objects.col(i).array().floor().matrix()).cast<int>();
      auto idx = map_index(coord);
      if ('x' == map[idx]) {
        map[idx] = '*';
      } else {
        map[idx] = 'o';
      }
    }

    for (unsigned i = 0; i < test_locations.cols(); ++i) {
      Eigen::Vector2i coord = test_locations.col(i).cast<int>();
      auto idx = map_index(coord);
      map[idx] = 't';
    }

    for (unsigned i = 0; i < start_locations.cols(); ++i) {
      Eigen::Vector2i coord = start_locations.col(i).cast<int>();
      auto idx = map_index(coord);
      map[idx] = 's';
    }

    trim(map);
    return map;
  }

  std::string reward_to_string() {
    std::string map = barriers_to_string();

    std::unordered_map<FloatT, char> reverse_legend;
    for (auto const& r : reward_legend) {
      reverse_legend[r.second] = r.first;
    }

    auto map_index = get_map_index(world_size);
    for (unsigned i = 0; i < world_size(0); ++i) {
      for (unsigned j = 0; j < world_size(1); ++j) {
        const FloatT r = reward(i, j);
        if (0 == r) continue;
        map[map_index({i, j})] = reverse_legend[r];
      }
    }

    trim(map);
    return map;
  }

  void save_map(std::string output_prefix) {
    prepare();

    WorldBase::save_map(output_prefix);

    std::string output_dir = output_prefix + "/Agent";

    std::ofstream barrier_file(output_dir + "/barriers_xy.info");
    for (auto const& b : barriers) {
      // nb in xy, cols are x and rows y (and rows count from bottom of map)
      auto b_first_xy = ij_to_xy(b.first.cast<float>()).cast<int>();
      auto b_second_xy = ij_to_xy(b.second.cast<float>()).cast<int>();
      barrier_file << b_first_xy(0) << "," << b_first_xy(1) << ","
                   << b_second_xy(0) << "," << b_second_xy(1) << "\n";
    }
    barrier_file.flush();

    std::ofstream map_file(output_dir + "/map.txt");
    map_file << map_to_string() << std::endl;

    Eigen::write_binary((output_dir + "/start_locations.bin").c_str(), start_locations);

    if (test_locations.rows() > 0 && test_locations.cols() > 0) {
      Eigen::write_binary((output_dir + "/test_locations.bin").c_str(), test_locations);
    }

    if (reward.rows() > 0 && reward.cols() > 0) {
      Eigen::write_binary((output_dir + "/reward.bin").c_str(), reward);
    }

    std::ofstream reward_file(output_dir + "/reward.txt");
    reward_file << reward_to_string() << std::endl;
  }

  static bool line_segments_intersect(EigenVector2D P1, EigenVector2D Q1,
                                      EigenVector2D P2, EigenVector2D Q2) {

    FloatT const& x0 = P1(0);
    FloatT const& y0 = P1(1);

    FloatT const& x1 = Q1(0);
    FloatT const& y1 = Q1(1);

    FloatT const& a0 = P2(0);
    FloatT const& b0 = P2(1);

    FloatT const& a1 = Q2(0);
    FloatT const& b1 = Q2(1);

    // below from https://stackoverflow.com/a/14143738
    //
    // four endpoints are x0, y0 & x1,y1 & a0,b0 & a1,b1
    // returned values xy and ab are the fractional distance along xy and ab
    // and are only defined when the result is true

    auto IsBetween = [](const FloatT& x0, const FloatT& x, const FloatT& x1){
      return (x >= x0) && (x <= x1);
    };

    FloatT ab, xy;

    bool partial = false;
    FloatT denom = (b0 - b1) * (x0 - x1) - (y0 - y1) * (a0 - a1);
    if (denom == 0) {
      xy = -1;
      ab = -1;
    } else {
      xy = (a0 * (y1 - b1) + a1 * (b0 - y1) + x1 * (b1 - b0)) / denom;
      partial = IsBetween(0, xy, 1);
      if (partial) {
        // no point calculating this unless xy is between 0 & 1
        ab = (y1 * (x0 - a1) + b1 * (x1 - x0) + y0 * (a1 - x1)) / denom;
      }
    }
    if ( partial && IsBetween(0, ab, 1)) {
      ab = 1-ab;
      xy = 1-xy;
      return true;
    }  else return false;
  }

  bool intersects_barrier_xy(EigenVector2D Pxy, EigenVector2D Qxy) {
    for (auto const& b : barriers) {
      // barriers are represented in {row, col} indices, not {x, y}:

      auto b_first_xy = ij_to_xy(b.first.cast<FloatT>());
      auto b_second_xy = ij_to_xy(b.second.cast<FloatT>());

      if (line_segments_intersect(Pxy, Qxy, b_first_xy, b_second_xy)) {
        return true;
      }
    }
    return false;
  }


  /*

    first, strip whitespace from edges

    map: "x" -> invisible barrier
         "*" -> barrier with visible object
         "o" -> visible object (no barrier)
              - else passable
      +  "s" for start locations
      +  "t" for test locations

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


class QMazePolicy {
  /*
   * In future, want to replace action selection with circuits
   * - need a mechanism!
   */

  std::default_random_engine rand_engine;

  std::vector<std::pair<FloatT, FloatT> > actions; // {r, theta}
  std::uniform_real_distribution<FloatT> action_die;

  EigenMatrix Q;
  unsigned buffer_Q_timesteps = 0;

  FloatT q0 = 10;                 // initial Q values: larger encourages exploration
  FloatT beta = 0.5;              // softmax inverse temperature
  FloatT invalid_act_reward = -5; // 'reward' for taking an invalid action
  FloatT alpha = 0.2;             // Q learning rate
  FloatT gamma = 0.85;            // Q learning discount factor

  FloatT ep_length = infinity<FloatT>();

  bool prepared = false;
  bool started = false;
  unsigned prev_state;
  unsigned curr_action;
  int action_stage = 0;

  FloatT t_episode = 0;
  unsigned _timesteps = 0;

  unsigned state_index(FloatT x, FloatT y) {
    /* given state coordinates, return state index into Q
       + nb: x is horizontal, which means along columns of map
             - also nb this gives rows upside down */
    int i = std::floor(y);
    int j = std::floor(x);
    assert(i >= 0 && j >= 0);
    return i * Q.cols() + j;
  }

protected:
  void reset_start_location(AgentBase& a) {
    auto& w = dynamic_cast<MazeWorld&>(a);
    started = false;
    action_stage = 0;
    a.change_action_ts = 0;
    a.choose_next_action_ts = 0;
    a.position = w.propose_start_location_xy();
  }

  void prepare(AgentBase& a) {
    if (prepared) return;
    assert(!a.smooth_AHV); // for now...
    prepared = true;
    _timesteps = 0;
    t_episode = 0;
    reset_start_location(a);
    auto& w = dynamic_cast<MazeWorld&>(a);
    Q = q0 * EigenMatrix::Ones(w.world_size(0) * w.world_size(1), actions.size());
    // std::cout << "\nQ: " << Q.rows() << "," << Q.cols() << "\n";
  }

  void update_per_dt(AgentBase& agent, FloatT dt) {
    // TODO: update_visible_objects(agent, dt);

    /*
    if (buffer_Q_timesteps > 0 && !(_timesteps % buffer_Q_timesteps)) {
      buffer_Q();
    }
    */

    t_episode += dt;
    ++_timesteps;

    if (t_episode > ep_length) {
      t_episode = 0;
      reset_start_location(agent);
    }
  }

  void update_Q(AgentBase& a, FloatT dt) {
    /* standard q learning, given world reward structure */
    auto& w = dynamic_cast<MazeWorld&>(a);

    // rows are along y axis; columns along x
    auto ij = w.xy_to_ij(a.position.array().floor().matrix()).cast<unsigned>();
    FloatT x = a.position(0);
    FloatT y = a.position(1);

    auto const& r = w.reward(ij(0), ij(1));
    auto q_max = Q.row(state_index(x, y)).maxCoeff();
    auto& q = Q(prev_state, curr_action);

    q = (1 - alpha) * q + alpha * (r + gamma * q_max);
  }

  /* check intersections of rays from self to objects, and barriers */
  void update_visible_objects(AgentBase& agent, FloatT dt); // TODO

  void choose_new_action(AgentBase& a, FloatT dt) {
    if (0 == action_stage) {
      /* We only update Q when we choose a new action,
         thereby simplifying the problem through discretisation.
         Later, we can update Q continuously in update_per_dt. */
      auto curr_state = state_index(a.position(0), a.position(1));
      if (started) update_Q(a, dt);
      else started = true;

      while (true) {
        /* - use Q matrix to choose probabilistically (softmax) */
        EigenVector p_action = (beta * Q.row(curr_state)).array().exp().matrix();
        p_action /= p_action.sum();

        FloatT action_choice = action_die(rand_engine);
        unsigned action_i = 0;
        for (; action_i < p_action.size(); ++action_i) {
          action_choice -= p_action(action_i);
          if (action_choice <= 0) break;
        }

        // Handle the rare case where rounding means we go off the end of the action vector:
        if (action_i >= p_action.size()) {
          action_i = p_action.size() - 1;
        }

        FloatT const& r = actions[action_i].first;
        FloatT const& theta = actions[action_i].second;
        FloatT delta_x = r * std::cos(theta);
        FloatT delta_y = r * std::sin(theta);

        EigenVector2D target_position = a.position + EigenVector2D{delta_x, delta_y};
        auto& w = dynamic_cast<MazeWorld&>(a);

        // check action validity:
        if (!w.intersects_barrier_xy(a.position, target_position)) {
          // valid, so update info
          // std::cout << "\nSELECT " << action_i << "(" << p_action.size() << "," << p_action.sum() << "," << action_choice << "): " << a.position.transpose() << " -> " << target_position.transpose() << "\n";
          prev_state = curr_state;
          curr_action = action_i;
          break; // and stop trying here
        } else {
          // invalid, so update Q matrix accordingly and then try again
          // std::cout << "\nINVALID " << action_i << "; " << r << ", " << theta << ": " << a.position.transpose() << " -> " << target_position.transpose() << "\n";
          Q(curr_state, action_i)
            = (1 - alpha) * Q(curr_state, action_i)
            + alpha * (invalid_act_reward + gamma * Q.row(curr_state).maxCoeff());
          // std::cout << Q.row(curr_state) << "\n";
        }
      }

      // now sequence actions accordingly
      // first of all, do rotation:
      {
        FloatT const& theta = actions[curr_action].second;
        FloatT delta = theta - a.head_direction;
        if (delta < -M_PI) delta += 2 * M_PI;
        if (delta > M_PI) delta -= 2 * M_PI; // just to be safe

        // choose AHV that will lead to closest to the user-intended action duration:
        int AHV_i = -1; unsigned i = 0;
        FloatT delta_delta = infinity<FloatT>();
        for (auto const& AHV : a.AHVs) {
          FloatT intended_duration = AHV.second <= 0 ? 1 : AHV.second;
          FloatT d = delta - AHV.first * intended_duration;
          if (std::abs(d) < std::abs(delta_delta)
              && !(std::abs(delta) > 0 && AHV.first == 0)) { // ignore 0 AHV
            delta_delta = d;
            AHV_i = i;
          }
          ++i;
        }
        assert(AHV_i >= 0);

        // if best AHV is in wrong direction, adjust delta accordingly: ...
        if (std::signbit(a.AHVs[AHV_i].first) != std::signbit(delta)) {
          if (delta < 0) delta += 2 * M_PI;
          else if (delta > 0) delta -= 2 * M_PI;
        }

        // finally, set agent parameters:
        a.curr_action = AgentBase::actions_t::AHV;
        a.curr_AHV = AHV_i;
        a.curr_FV = 0;

        a.target_head_direction = theta;

        FloatT duration = delta / a.AHVs[AHV_i].first;
        a.change_action_ts = 0;
        a.choose_next_action_ts = a.timesteps + round(duration / dt);
      }

      ++action_stage;
    } else {
      // finish currently sequenced action
      // do forward motion:
      FloatT const& r = actions[curr_action].first;

      // choose FV that will lead to closest to the user-intended action duration:
      int FV_i = -1; unsigned i = 0;
      FloatT delta = infinity<FloatT>();
      for (auto const& FV : a.FVs) {
        FloatT intended_duration = FV.second <= 0 ? 1 : FV.second;
        FloatT d = r - FV.first * intended_duration;
        if (std::abs(d) < std::abs(delta)
            && !(std::abs(r) > 0 && FV.first == 0)) { // ignore 0 FV
          delta = d;
          FV_i = i;
        }
        ++i;
      }
      assert(FV_i >= 0);

      // if best FV is in the wrong direction, panic!
      assert(std::signbit(a.FVs[FV_i].first) == std::signbit(r));

      // finally, set agent parameters
      a.curr_action = AgentBase::actions_t::FV;
      a.curr_FV = FV_i;
      a.curr_AHV = 0;

      a.target_position = a.position;
      a.target_position(0) += r * cos(a.head_direction);
      a.target_position(1) += r * sin(a.head_direction);

      FloatT duration = r / a.FVs[FV_i].first;
      a.change_action_ts = 0;
      a.choose_next_action_ts = a.timesteps + round(duration / dt);

      // next time, choose new action
      action_stage = 0;
    }
  }

public:
  QMazePolicy() {
    action_die = std::uniform_real_distribution<FloatT>(0, 1);
  }

  void add_velocity(FloatT r, FloatT theta) {
    assert(r > 0 && theta >= -M_PI && theta < 2 * M_PI);
    actions.push_back({r, theta});
  }

  void buffer_q_interval(int timesteps) {
    buffer_Q_timesteps = timesteps;
  }

  void episode_length(FloatT ep_len) {
    ep_length = ep_len;
  }

  void set_rl_params(FloatT _q0, FloatT _beta, FloatT _invalid_act_reward,
                     FloatT _alpha, FloatT _gamma) {
    q0 = _q0;
    beta = _beta;
    invalid_act_reward = _invalid_act_reward;
    alpha = _alpha;
    gamma = _gamma;
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

    WorldT::embed_agent(*this);
  }

  void prepare() {
    if (prepared) return;

    WorldT::prepare();

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

    WorldT::update_per_dt(dt);
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
};
