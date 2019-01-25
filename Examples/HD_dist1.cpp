#include "Spike/Models/RateModel.hpp"
#include "Spike/Models/RateAgent.hpp"

#include "toml/toml.h"

#include <fenv.h>
#include <omp.h>

// TODO: Add signal handlers

#define PLASTICITY_TYPE RatePlasticity

class model_t {
public:
  // Global data:
  toml::Value config;

  std::string output_path;

  bool read_weights;
  std::string weights_path;

  RateModel model;
  Context* ctx;

  Agent<RandomWalkPolicy, HDTestPolicy, OpenWorld> agent;

  // Basic parameters:
  FloatT timestep = pow(2, -10);
  FloatT buffer_timestep = pow(2, -6);

  FloatT train_time;
  FloatT test_on_time;
  FloatT test_off_time;

  FloatT test_interval = 1000; // Time between tests
  FloatT test_duration = 1000; // Test length -- TODO: Should automate this...

  std::vector<FloatT> test_times; // No tests right now


  // Action parameters:
  FloatT fwd_move_dist = 0.8;
  FloatT rot_angle = M_PI / 8;

  FloatT fwd_move_time = 0.2; // seconds per forward move
  FloatT angle_move_time = 0.4; // seconds per angular move


  // Neuron group parameters:
  //
  // VIS neurons:
  int N_per_obj = 80;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 1.0;

  std::shared_ptr<AgentVISRateNeurons> VIS;
  int N_VIS; // this is set by construct_neurons()

  // VIS feedforward inhibition:
  EigenVector VIS_INH_on;
  EigenVector VIS_INH_off;
  std::shared_ptr<DummyRateNeurons> VIS_INH;

  // HD neurons:
  int N_HD = 180;
  std::shared_ptr<AgentHDRateNeurons> HD;

  // S neurons:
  int N_S = 400;
  FloatT alpha_S = 2.0;
  FloatT beta_S = 0.5;
  FloatT tau_S = 1e-2;
  std::shared_ptr<RateNeurons> S;

  // A neurons:
  int N_A = 400;
  FloatT alpha_A = 2.0;
  FloatT beta_A = 0.5;
  FloatT tau_A = 1e-2;
  std::shared_ptr<RateNeurons> A;

  // SA neurons:
  int N_SA = 800;
  FloatT alpha_SA = 2.0;
  FloatT beta_SA = 0.5;
  FloatT tau_SA = 1e-2;
  std::shared_ptr<RateNeurons> SA;

  // Connectivity parameters:
  //
  FloatT axonal_delay = pow(2, -8); // seconds
  FloatT eps; // learning rate

  // HD -> A connectivity:
  FloatT HD_A_sparsity = 1.0;
  FloatT HD_A_scaling;

  // A -> SA connectivity:
  FloatT A_SA_sparsity = 0.5;
  FloatT A_SA_scaling;
  
  // S -> SA connectivity:
  FloatT S_SA_sparsity = 0.05;
  FloatT S_SA_scaling;

  // SA -> SA connectivity:
  FloatT SA_inhibition;

  // VIS -> S connectivity:
  FloatT VIS_S_sparsity = 0.05;
  FloatT VIS_S_scaling;
  FloatT VIS_S_INH_scaling;
  FloatT eps_VIS_S;

  // SA -> S connectivity:
  FloatT SA_S_sparsity = 1.0;
  FloatT SA_S_scaling;

  // S -> S connectivity:
  FloatT S_inhibition;


  // Synapse data:
  std::shared_ptr<RateSynapses> HD_A, A_SA, S_SA, SA_SA_INH, VIS_S, VIS_S_INH, SA_S, S_S_INH;
  std::shared_ptr<PLASTICITY_TYPE> plast_HD_A, plast_A_SA, plast_S_SA, plast_SA_SA_INH,
    plast_VIS_S, plast_VIS_S_INH, plast_SA_S, plast_S_S_INH;

  // Which weights to save after training?
  std::vector<std::shared_ptr<RateSynapses> > weights_to_store;


  // Electrodes:
  std::shared_ptr<RateElectrodes> VIS_elecs, HD_elecs, A_elecs, SA_elecs, S_elecs;


  void load_weights(RateSynapses& s) const {
    s.load_weights(weights_path + std::string("/") + s.label + std::string(".bin"));
  };

  void save_weights(RateSynapses& s) const {
    s.save_weights(output_path + std::string("/") + s.label + std::string(".bin"));
  };

  template <typename NeuronsPreT, typename NeuronsPostT>
  std::shared_ptr<RateSynapses>
  connect_neurons(NeuronsPreT& pre, NeuronsPostT& post, std::string label,
                  FloatT scaling, FloatT sparsity, FloatT delay,
                  bool try_load = true) const {
    auto syns = std::make_shared<RateSynapses>(ctx, &pre, &post, scaling, label);
    if (delay > 0) { syns->delay(ceil(delay / timestep));}

    if (try_load && read_weights) {
      load_weights(*syns);
    } else {
      // Create a uniform random weight matrix of given sparsity, with 1-normalized rows
      syns->weights
        (Eigen::make_random_matrix(post.size, pre.size, 1.0, true, 1.0-sparsity, 0, false));
    }
    if (sparsity < 1.0f) syns->make_sparse();

    return syns;
  }

  template <typename NeuronsPreT, typename NeuronsPostT>
  std::shared_ptr<RateSynapses>
  connect_neurons(NeuronsPreT& pre, NeuronsPostT& post, std::string label,
                  FloatT scaling, EigenMatrix W,
                  FloatT delay, bool is_sparse = false) const {
    auto syns = std::make_shared<RateSynapses>(ctx, &pre, &post, scaling, label);
    if (delay > 0) { syns->delay(ceil(delay / timestep));}

    syns->weights(W);
    if (is_sparse) syns->make_sparse();

    return syns;
  }


  void config_simulation_schedule() {
    train_time = config.get<double>("train_time");
    test_on_time = config.get<double>("test_on_time");
    test_off_time = config.get<double>("test_off_time");

    agent.add_pause_time(config.get<double>("pause_time"),
                         config.get<double>("pause_duration"));

    for (unsigned i = 0; i < 0; ++i) {
      test_times.push_back(test_times[i] + test_duration + test_interval);
    }
    std::cout << "Test times: ";
    for (auto const& t : test_times) {
      std::cout << t << " ";
      agent.add_test_time(t);
    }
    std::cout << std::endl;
  }


  void add_actions() {
    agent.add_AHV(rot_angle / angle_move_time, angle_move_time);
    agent.add_AHV(-rot_angle / angle_move_time, angle_move_time);

    agent.add_FV(fwd_move_dist / fwd_move_time, fwd_move_time);

    agent.add_distal_object(0);
    agent.add_distal_object(0.5 * M_PI);
    agent.add_distal_object(M_PI);
    agent.add_distal_object(1.5 * M_PI);
  }


  void construct_neurons() {
    VIS = std::make_shared<AgentVISRateNeurons>
      (ctx, &agent, N_per_obj, sigma_VIS, lambda_VIS, "VIS");
    N_VIS = VIS->size;

    VIS_INH_on = EigenVector::Ones(N_VIS);
    VIS_INH_off = EigenVector::Zero(N_VIS);
    VIS_INH = std::make_shared<DummyRateNeurons>(ctx, N_VIS, "VIS_INH");

    HD = std::make_shared<AgentHDRateNeurons>(ctx, &agent, N_HD, sigma_VIS, lambda_VIS, "HD");

    S = std::make_shared<RateNeurons>(ctx, N_S, "S", alpha_S, beta_S, tau_S);
    A = std::make_shared<RateNeurons>(ctx, N_A, "A", alpha_A, beta_A, tau_A);
    SA = std::make_shared<RateNeurons>(ctx, N_SA, "SA", alpha_SA, beta_SA, tau_SA);

    //S.enable_homeostasis(0.02, 0.001, true);
    //A.enable_homeostasis(0.02, 0.001, true);

    printf("%d, %d, %d, %d, %d\n", N_VIS, N_HD,  N_S, N_A, N_SA);
  }

  void config_connectivity_params() {
    // axonal_delay = fixed above

    // HD_A_sparsity = fixed above
    // A_SA_sparsity = fixed above
    // S_SA_sparsity = fixed above
    // VIS_S_sparsity = fixed above
    // SA_S_sparsity = fixed above

    eps = config.get<double>("eps");
    eps_VIS_S = eps;

    HD_A_scaling = config.get<double>("HD_A_scaling");

    A_SA_scaling = config.get<double>("A_SA_scaling");
    S_SA_scaling = config.get<double>("S_SA_scaling");

    SA_inhibition = config.get<double>("SA_inhibition");

    VIS_S_scaling = config.get<double>("VIS_S_scaling");
    VIS_S_INH_scaling = config.get<double>("VIS_S_INH_scaling");

    SA_S_scaling = config.get<double>("SA_S_scaling");

    S_inhibition = config.get<double>("S_inhibition");
  }

  void construct_synapses() {
    // HD -> A connectivity:   (NB: not saved/loaded to/from disk)
    HD_A = connect_neurons(*HD, *A, "HD_A", HD_A_scaling, HD_A_sparsity, 0.0, false);
    plast_HD_A = std::make_shared<PLASTICITY_TYPE>(ctx, HD_A.get());
    A->connect_input(HD_A.get(), plast_HD_A.get());

    // A -> SA connectivity:   (NB: not saved/loaded to/from disk)
    A_SA = connect_neurons(*A, *SA, "A_SA", A_SA_scaling, A_SA_sparsity, 0.0, false);
    plast_A_SA = std::make_shared<PLASTICITY_TYPE>(ctx, A_SA.get());
    SA->connect_input(A_SA.get(), plast_A_SA.get());

    // S -> SA connectivity:
    S_SA = connect_neurons(*S, *SA, "S_SA", S_SA_scaling, S_SA_sparsity, axonal_delay);
    plast_S_SA = std::make_shared<PLASTICITY_TYPE>(ctx, S_SA.get());
    SA->connect_input(S_SA.get(), plast_S_SA.get());
    weights_to_store.push_back(S_SA);

    // SA -> SA connectivity:   (NB: not saved/loaded to/from disk)
    SA_SA_INH = connect_neurons(*SA, *SA, "SA_SA_INH", SA_inhibition,
                                EigenMatrix::Ones(N_SA, N_SA), 0.0);
    plast_SA_SA_INH = std::make_shared<PLASTICITY_TYPE>(ctx, SA_SA_INH.get());
    SA->connect_input(SA_SA_INH.get(), plast_SA_SA_INH.get());

    // VIS -> S connectivity:
    VIS_S = connect_neurons(*VIS, *S, "VIS_S", VIS_S_scaling, VIS_S_sparsity, 0.0);
    plast_VIS_S = std::make_shared<PLASTICITY_TYPE>(ctx, VIS_S.get());
    S->connect_input(VIS_S.get(), plast_VIS_S.get());
    weights_to_store.push_back(VIS_S);

    VIS_S_INH = connect_neurons(*VIS_INH, *S, "VIS_S_INH", VIS_S_INH_scaling,
                                EigenMatrix::Ones(N_S, N_VIS), 0.0);
    plast_VIS_S_INH = std::make_shared<PLASTICITY_TYPE>(ctx, VIS_S_INH.get());
    S->connect_input(VIS_S_INH.get(), plast_VIS_S_INH.get());

    // SA -> S connectivity:
    SA_S = connect_neurons(*SA, *S, "SA_S", SA_S_scaling, SA_S_sparsity, axonal_delay);
    plast_SA_S = std::make_shared<PLASTICITY_TYPE>(ctx, SA_S.get());
    S->connect_input(SA_S.get(), plast_SA_S.get());
    weights_to_store.push_back(SA_S);

    // S -> S connectivity:
    S_S_INH = connect_neurons(*S, *S, "S_S_INH", S_inhibition,
                              EigenMatrix::Ones(N_S, N_S), 0.0);
    plast_S_S_INH = std::make_shared<PLASTICITY_TYPE>(ctx, S_S_INH.get());
    S->connect_input(S_S_INH.get(), plast_S_S_INH.get());
  }

  void init_electrodes() {
    VIS_elecs = std::make_shared<RateElectrodes>(output_path, VIS.get());
    HD_elecs = std::make_shared<RateElectrodes>(output_path, HD.get());
    A_elecs = std::make_shared<RateElectrodes>(output_path, A.get());
    SA_elecs = std::make_shared<RateElectrodes>(output_path, SA.get());
    S_elecs = std::make_shared<RateElectrodes>(output_path, S.get());
  }

  void construct_model() {
    model.add(&agent);

    model.add(VIS.get());
    model.add(VIS_INH.get());

    model.add(HD.get());
    model.add(A.get());
    model.add(SA.get());
    model.add(S.get());

    model.add(VIS_elecs.get());

    model.add(HD_elecs.get());
    model.add(A_elecs.get());
    model.add(SA_elecs.get());
    model.add(S_elecs.get());
  }


  void set_neuron_schedule() {
    VIS->t_stop_after = train_time + test_on_time;
    VIS_INH->add_schedule(VIS->t_stop_after, VIS_INH_on);
    VIS_INH->add_schedule(infinity<FloatT>(), VIS_INH_off);
  }

  void set_plasticity_schedule() {
    // No inhibitory plasticity:
    plast_SA_SA_INH->add_schedule(infinity<FloatT>(), 0);
    plast_S_S_INH->add_schedule(infinity<FloatT>(), 0);

    // No plasticity on action inputs:
    plast_HD_A->add_schedule(infinity<FloatT>(), 0);
    plast_A_SA->add_schedule(infinity<FloatT>(), 0);

    // Rest have plasticity on only during training, with params above:
    auto add_plast_schedule = [&](FloatT dur, FloatT rate) {
      plast_VIS_S->add_schedule(dur, rate);
      plast_SA_S->add_schedule(dur, rate);
      plast_S_SA->add_schedule(dur, rate);
      // plast_A_SA->add_schedule(dur, rate);
    };

    FloatT total_time = 0;
    for (unsigned i = 0; i < test_times.size(); ++i) {
      FloatT this_test_start = test_times[i];
      FloatT prev_test_end = 0 == i ? 0 : (test_times[i-1] + test_duration);
      FloatT train_duration = this_test_start - prev_test_end;
      total_time += train_duration;

      add_plast_schedule(train_duration, eps);
      add_plast_schedule(test_duration, 0);
    }
    if (test_times.size() > 0) total_time += test_duration;
    FloatT remaining_train_time = train_time - total_time;

    if (remaining_train_time > 0) add_plast_schedule(remaining_train_time, eps);
    add_plast_schedule(infinity<FloatT>(), 0);
  }

  void set_storage_schedule() {
    model.set_buffer_intervals((float)buffer_timestep); // TODO: Use proper units
    model.set_buffer_start(0);

    model.set_weights_buffer_interval(0);
    model.set_weights_buffer_start(infinity<float>());

    bool stored_weights = false;
    if (train_time > 0) {
      model.add_hook([&]() {
          if (model.current_time() > train_time && !read_weights && !stored_weights) {
            for (auto& s : weights_to_store) {
              save_weights(*s);
            }
            stored_weights = true;
          }
        });
    }

    agent.save_map(output_path);
    agent.record_history(output_path, round(buffer_timestep/timestep), 0);
  }

  void set_runtime_schedule() {
    set_neuron_schedule();
    set_plasticity_schedule();

    model.set_simulation_time(train_time + test_on_time + test_off_time, timestep);

    set_storage_schedule();
  }


  model_t(std::string toml_path) {
    std::ifstream toml_ifstream(toml_path);
    config = toml::parse(toml_ifstream).value;

    output_path = config.get<std::string>("output_path");

    read_weights = false;
    try {
      weights_path = config.get<std::string>("weights_path");
      read_weights = true;
    } catch (...) {
      std::cout << "Not reading weights from disk\n";
      read_weights = false;
    }

    ctx = model.context;

    // Tell Spike to talk
    ctx->verbose = true;
    ctx->backend = "Eigen";

    // Get simulation schedule from config:
    config_simulation_schedule();

    // Give actions to agent:
    add_actions();

    // Construct neuron groups
    construct_neurons();

    // Use config to set connectivity parameters:
    config_connectivity_params();

    // Construct synapses accordingly:
    construct_synapses();

    // Have to construct electrodes after neurons:
    init_electrodes();

    // Construct simulation model (ie, enable circuits & electrodes):
    construct_model();

    // Set runtime schedule:
    set_runtime_schedule();
  }

  void run() {
    model.start();
    printf("%f, %f, %f; %d\n", model.current_time(), model.dt, model.t_stop, model.timesteps());
  }
};


int main(int argc, char *argv[]) {
  Eigen::initParallel();
  std::cout << Eigen::nbThreads() << std::endl;
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  assert(argc > 1);
  auto model = model_t(std::string(argv[1]));

  model.run();
}
