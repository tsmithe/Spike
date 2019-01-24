#include "Spike/Models/RateModel.hpp"
#include "Spike/Models/RateAgent.hpp"

#include "toml/toml.h"

#include <fenv.h>
#include <omp.h>

// TODO: Add signal handlers

#define PLASTICITY_TYPE RatePlasticity

int main(int argc, char *argv[]) {
  Eigen::initParallel();
  std::cout << Eigen::nbThreads() << std::endl;
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  assert(argc > 1);
  std::string toml_path = argv[1];
  std::ifstream toml_stream(toml_path);
  toml::ParseResult pr = toml::parse(toml_stream);
  auto& toml_value = pr.value;

  std::string output_path = toml_value.get<std::string>("output_path");

  FloatT timestep = pow(2, -10); // seconds (TODO units)
  FloatT buffer_timestep = pow(2, -6);
  FloatT train_time = toml_value.get<double>("train_time"); // 6000; // 300

  bool read_weights = false;
  std::string weights_path;
  try {
    weights_path = toml_value.get<std::string>("weights_path");
    read_weights = true;
  } catch (...) {
    std::cout << "Not reading weights from disk\n";
    read_weights = false;
  }

  FloatT test_on_time = toml_value.get<double>("test_on_time"); // 200; // 800
  FloatT test_off_time = toml_value.get<double>("test_off_time"); // 300; // 200
  
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  ctx->verbose = true;
  ctx->backend = "Eigen";

  // Create Agent
  Agent<QMazePolicy, MazePlaceTestPolicy, MazeWorld> agent;
  // agent.seed(123);

  agent.episode_length(150);

  agent.add_pause_time(toml_value.get<double>("pause_time"),
                       toml_value.get<double>("pause_duration"));

  FloatT test_duration = 1853; // turns off plasticity during testing for this duration
  FloatT test_interval = 1200; // time between tests (s)
  std::vector<FloatT> test_times; //{200};
  for (unsigned i = 0; i < 0; ++i) {
    test_times.push_back(test_times[i] + test_duration + test_interval);
  }
  std::cout << "Test times: ";
  for (auto const& t : test_times) {
    std::cout << t << " ";
    agent.add_test_time(t);
  }
  std::cout << std::endl;

  FloatT fwd_move_dist = 0.8;
  FloatT rot_angle = M_PI / 8;

  FloatT fwd_move_time = 0.2; ///6.0; // seconds per forward move
  FloatT angle_move_time = 0.4; ///6.0; // seconds per angular move

  agent.add_AHV(rot_angle / angle_move_time, angle_move_time);
  agent.add_AHV(-rot_angle / angle_move_time, angle_move_time);

  agent.add_FV(fwd_move_dist / fwd_move_time, fwd_move_time);

  const unsigned num_directions = 8;
  for (unsigned i = 0; i < num_directions; ++i) {
    agent.add_velocity(4.0, (1.0/4.0)*i*M_PI);
  }

  agent.load_map(
"xxxxxxxxxxxxxxxxxxxxxxx\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x     o         o     x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x          o          x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x     o         o     x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x          s          x\n"
"xxxxxxxxxxxxxxxxxxxxxxx\n");

  agent.load_tests(
"xxxxxxxxxxxxxxxxxxxxxxx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xpppppppppppppppppppppx\n"
"xxxxxxxxxxxxxxxxxxxxxxx\n");

  int N_per_obj = 80;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 1.0;

  int N_HD = 180;
  int N_per_state = 10;

  // Input neurons:
  AgentVISRateNeurons VIS(ctx, &agent, N_per_obj, sigma_VIS, lambda_VIS, "VIS");
  AgentHDRateNeurons HD(ctx, &agent, N_HD, sigma_VIS, lambda_VIS, "HD");
  AgentFVRateNeurons FV(ctx, &agent, N_per_state, "FV");

  int N_VIS = VIS.size;
  int N_FV = FV.size;

  EigenVector VIS_INH_on = EigenVector::Ones(N_VIS);
  EigenVector VIS_INH_off = EigenVector::Zero(N_VIS);
  DummyRateNeurons VIS_INH(ctx, N_VIS, "VIS_INH");

  // S neurons:
  int N_S = 400;
  FloatT alpha_S = 2.0;
  FloatT beta_S = 0.5;
  FloatT tau_S = 1e-2;
  RateNeurons S(ctx, N_S, "S", alpha_S, beta_S, tau_S);
  //S.enable_homeostasis(0.02, 0.001, true);

  // A neurons:
  int N_A = 400;
  FloatT alpha_A = 2.0;
  FloatT beta_A = 0.5;
  FloatT tau_A = 1e-2;
  RateNeurons A(ctx, N_A, "A", alpha_A, beta_A, tau_A);
  //A.enable_homeostasis(0.02, 0.001, true);

  // SxA neurons:
  int N_SA = 800;
  FloatT alpha_SA = 2.0;
  FloatT beta_SA = 0.5;
  FloatT tau_SA = 1e-2;
  RateNeurons SA(ctx, N_SA, "SA", alpha_SA, beta_SA, tau_SA);
  // SxA.enable_homeostasis(0.05, 0.001);

  printf("%d, %d, %d, %d, %d, %d\n", N_VIS, N_HD, N_FV, N_S, N_A, N_SA);

  // General parameters:
  FloatT axonal_delay = pow(2, -8); // seconds (TODO units)
  FloatT eps = toml_value.get<double>("eps");


  // FV -> A connectivity:
  FloatT FV_A_sparsity = 1.0;
  FloatT FV_A_scaling = toml_value.get<double>("FV_A_scaling");

  // HD -> A connectivity:
  FloatT HD_A_sparsity = 1.0;
  FloatT HD_A_scaling = toml_value.get<double>("HD_A_scaling");

  // A -> SA connectivity:
  FloatT A_SA_sparsity = 0.5;
  FloatT A_SA_scaling = toml_value.get<double>("A_SA_scaling");
  
  // S -> SA connectivity:
  FloatT S_SA_sparsity = 0.05;
  FloatT S_SA_scaling = toml_value.get<double>("S_SA_scaling");

  // SA -> SA connectivity:
  FloatT SA_inhibition = toml_value.get<double>("SA_inhibition");

  // VIS -> S connectivity:
  FloatT VIS_S_sparsity = 0.05;
  FloatT VIS_S_scaling = toml_value.get<double>("VIS_S_scaling");
  FloatT VIS_S_INH_scaling = toml_value.get<double>("VIS_S_INH_scaling");
  FloatT eps_VIS_S = eps;

  // SA -> S connectivity:
  FloatT SA_S_sparsity = 1.0;
  FloatT SA_S_scaling = toml_value.get<double>("SA_S_scaling");

  // S -> S connectivity:
  FloatT S_inhibition = toml_value.get<double>("S_inhibition");


  auto load_weights = [&](RateSynapses& s) {
    s.load_weights(weights_path + std::string("/") + s.label + std::string(".bin"));
  };

  auto save_weights = [&](RateSynapses& s) {
    s.save_weights(output_path + std::string("/") + s.label + std::string(".bin"));
  };

  std::vector<RateSynapses*> weights_to_store;


  // FV -> A connectivity:
  RateSynapses FV_A(ctx, &FV, &A, FV_A_scaling, "FV_A");
  /// We don't save / load these weights:
  // weights_to_store.push_back(&FV_A);
  // if (read_weights) { load_weights(FV_A); } else { ...
  EigenMatrix W_FV_A
    = Eigen::make_random_matrix(N_A, N_FV, 1.0, true, 1.0-FV_A_sparsity, 0, false);
  FV_A.weights(W_FV_A);
  if (FV_A_sparsity < 1.0f) FV_A.weights(W_FV_A);

  PLASTICITY_TYPE plast_FV_A(ctx, &FV_A);

  A.connect_input(&FV_A, &plast_FV_A);


  // HD -> A connectivity:
  RateSynapses HD_A(ctx, &HD, &A, HD_A_scaling, "HD_A");
  /// We don't save / load these weights:
  // weights_to_store.push_back(&HD_A);
  // if (read_weights) { load_weights(HD_A); } else { ...
  EigenMatrix W_HD_A
    = Eigen::make_random_matrix(N_A, N_HD, 1.0, true, 1.0-HD_A_sparsity, 0, false);
  HD_A.weights(W_HD_A);
  if (HD_A_sparsity < 1.0f) HD_A.make_sparse();

  PLASTICITY_TYPE plast_HD_A(ctx, &HD_A);

  A.connect_input(&HD_A, &plast_HD_A);


  // A -> SA connectivity:
  RateSynapses A_SA(ctx, &A, &SA, A_SA_scaling, "A_SA");
  // A_SA.delay(ceil(axonal_delay / timestep));
  weights_to_store.push_back(&A_SA);
  if (read_weights) {
    load_weights(A_SA);
  } else {
    EigenMatrix W_A_SA
      = Eigen::make_random_matrix(N_SA, N_A, 1.0, true, 1.0-A_SA_sparsity, 0, false);
    A_SA.weights(W_A_SA);
  }
  A_SA.make_sparse();

  PLASTICITY_TYPE plast_A_SA(ctx, &A_SA);

  SA.connect_input(&A_SA, &plast_A_SA);



  // S -> SA connectivity:
  RateSynapses S_SA(ctx, &S, &SA, S_SA_scaling, "S_SA");
  S_SA.delay(ceil(axonal_delay / timestep));
  weights_to_store.push_back(&S_SA);
  if (read_weights) {
    load_weights(S_SA);
  } else {
    EigenMatrix W_S_SA
      = Eigen::make_random_matrix(N_SA, N_S, 1.0, true, 1.0-S_SA_sparsity, 0, false);
    S_SA.weights(W_S_SA);
  }
  S_SA.make_sparse();

  PLASTICITY_TYPE plast_S_SA(ctx, &S_SA);

  SA.connect_input(&S_SA, &plast_S_SA);


  // SA -> SA connectivity:
  RateSynapses SA_SA_INH(ctx, &SA, &SA, SA_inhibition, "SA_SA_INH");
  SA_SA_INH.weights(EigenMatrix::Ones(N_SA, N_SA));

  PLASTICITY_TYPE plast_SA_SA_INH(ctx, &SA_SA_INH);

  SA.connect_input(&SA_SA_INH, &plast_SA_SA_INH);


  // VIS -> S connectivity:
  RateSynapses VIS_S(ctx, &VIS, &S, VIS_S_scaling, "VIS_S");
  weights_to_store.push_back(&VIS_S);
  if (read_weights) {
    load_weights(VIS_S);
  } else {
    EigenMatrix W_VIS_S
      // = EigenMatrix::Ones(N_S, N_VIS);
      = Eigen::make_random_matrix(N_S, N_VIS, 1.0, true, 1.0-VIS_S_sparsity, 0, false);
    VIS_S.weights(W_VIS_S);
  }
  VIS_S.make_sparse();

  PLASTICITY_TYPE plast_VIS_S(ctx, &VIS_S);

  S.connect_input(&VIS_S, &plast_VIS_S);

  RateSynapses VIS_S_INH(ctx, &VIS_INH, &S, VIS_S_INH_scaling, "VIS_S_INH");
  VIS_S_INH.weights(EigenMatrix::Ones(N_S, N_VIS));

  PLASTICITY_TYPE plast_VIS_S_INH(ctx, &VIS_S_INH);

  S.connect_input(&VIS_S_INH, &plast_VIS_S_INH);


  // SA -> S connectivity:
  RateSynapses SA_S(ctx, &SA, &S, SA_S_scaling, "SA_S");
  SA_S.delay(ceil(axonal_delay / timestep));
  weights_to_store.push_back(&SA_S);
  if (read_weights) {
    load_weights(SA_S);
  } else {
    EigenMatrix W_SA_S
      = Eigen::make_random_matrix(N_S, N_SA, 1.0, true, 1.0-SA_S_sparsity, 0, false);
    SA_S.weights(W_SA_S);
  }
  SA_S.make_sparse();

  PLASTICITY_TYPE plast_SA_S(ctx, &SA_S);

  S.connect_input(&SA_S, &plast_SA_S);


  // S -> S connectivity:
  RateSynapses S_S_INH(ctx, &S, &S, S_inhibition, "S_S_INH");
  S_S_INH.weights(EigenMatrix::Ones(N_S, N_S));
  // S_S_INH.delay(ceil(0.5*axonal_delay / timestep)); // ?

  PLASTICITY_TYPE plast_S_S_INH(ctx, &S_S_INH);

  S.connect_input(&S_S_INH, &plast_S_S_INH); //, 1.0);


  // Set simulation schedule:
  VIS.t_stop_after = train_time + test_on_time;
  VIS_INH.add_schedule(VIS.t_stop_after, VIS_INH_on);
  VIS_INH.add_schedule(infinity<FloatT>(), VIS_INH_off);

  // No inhibitory plasticity:
  plast_SA_SA_INH.add_schedule(infinity<FloatT>(), 0);
  plast_S_S_INH.add_schedule(infinity<FloatT>(), 0);

  // No plasticity on action inputs:
  plast_FV_A.add_schedule(infinity<FloatT>(), 0);
  plast_HD_A.add_schedule(infinity<FloatT>(), 0);
  plast_A_SA.add_schedule(infinity<FloatT>(), 0);

  // Rest have plasticity on only during training, with params above:
  auto add_plast_schedule = [&](FloatT dur, FloatT rate) {
    plast_VIS_S.add_schedule(dur, rate);
    plast_SA_S.add_schedule(dur, rate);
    plast_S_SA.add_schedule(dur, rate);
    // plast_A_SA.add_schedule(dur, rate);
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


  // Have to construct electrodes after neurons:
  RateElectrodes VIS_elecs(output_path, &VIS);
  RateElectrodes HD_elecs(output_path, &HD);
  RateElectrodes FV_elecs(output_path, &FV);
  RateElectrodes A_elecs(output_path, &A);
  RateElectrodes SA_elecs(output_path, &SA);
  RateElectrodes S_elecs(output_path, &S);


  // Add Agent, Neurons and Electrodes to Model
  model.add(&agent);

  model.add(&VIS);
  model.add(&VIS_INH);

  model.add(&HD);
  model.add(&FV);
  model.add(&A);
  model.add(&SA);
  model.add(&S);

  model.add(&VIS_elecs);

  model.add(&HD_elecs);
  model.add(&FV_elecs);
  model.add(&A_elecs);
  model.add(&SA_elecs);
  model.add(&S_elecs);


  // Set simulation time parameters:
  model.set_simulation_time(train_time + test_on_time + test_off_time, timestep);
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


  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.current_time(), model.dt, model.t_stop, model.timesteps());
}
