#include "Spike/Models/RateModel.hpp"
#include "Spike/Models/RateAgent.hpp"

#include "toml/toml.h"

#include <fenv.h>
#include <omp.h>

// TODO: Add signal handlers

int main(int argc, char *argv[]) {
  Eigen::initParallel();
  omp_set_num_threads(32);
  Eigen::setNbThreads(8);
  std::cout << Eigen::nbThreads() << std::endl;
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  bool read_weights = false;
  std::string weights_path;

  /* TODO: (make this neater and integrated with TOML config)
  if (argc == 2) {
    read_weights = true;
    weights_path = argv[1];
  }
  */

  assert(argc > 1);
  std::string toml_path = argv[1];
  std::ifstream toml_stream(toml_path);
  toml::ParseResult pr = toml::parse(toml_stream);
  auto& toml_value = pr.value;

  std::string output_path = toml_value.get<std::string>("output_path");

  FloatT timestep = pow(2, -9); // seconds (TODO units)
  FloatT buffer_timestep = pow(2, -6);
  FloatT train_time = toml_value.get<double>("train_time"); // 6000; // 300
  if (read_weights) train_time = 0;
  FloatT test_on_time = toml_value.get<double>("test_on_time"); // 200; // 800
  FloatT test_off_time = toml_value.get<double>("test_off_time"); // 300; // 200
  FloatT start_recording_time = 0;
  if (read_weights) start_recording_time = 0;
  
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

  FloatT fwd_move_dist = 0.4;
  FloatT rot_angle = M_PI / 8;

  FloatT fwd_move_time = 0.2; ///6.0; // seconds per forward move
  FloatT angle_move_time = 0.1; ///6.0; // seconds per angular move

  agent.add_AHV(rot_angle / angle_move_time, angle_move_time);
  agent.add_AHV(-rot_angle / angle_move_time, angle_move_time);

  agent.add_FV(fwd_move_dist / fwd_move_time, fwd_move_time);

  for (unsigned i = 0; i < 8; ++i) {
    agent.add_velocity(4.0, (1.0/4.0)*i*M_PI);
  }

  agent.load_map(
"xxxxxxxxxxxxxxxxxxxxxxx\n"
"x                     x\n"
"x s                 s x\n"
"x  o    o    o    o   x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x   o    o    o    o  x\n"
"x                     x\n"
"x                     x\n"
"x          s          x\n"
"x                     x\n"
"x                     x\n"
"x  o    o    o    o   x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x                     x\n"
"x   o    o    o    o  x\n"
"x s                 s x\n"
"x                     x\n"
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

  int N_per_obj = 60;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 1.0;

  int N_HD = 180;
  int N_per_state = 50;

  // Input neurons:
  AgentVISRateNeurons VIS(ctx, &agent, N_per_obj, sigma_VIS, lambda_VIS, "VIS");
  AgentHDRateNeurons HD(ctx, &agent, N_HD, sigma_VIS, lambda_VIS, "HD");
  AgentFVRateNeurons FV(ctx, &agent, N_per_state, "FV");

  int N_VIS = VIS.size;
  int N_FV = FV.size;

  EigenVector VIS_INH_on = EigenVector::Ones(N_VIS);
  EigenVector VIS_INH_off = EigenVector::Zero(N_VIS);
  DummyRateNeurons VIS_INH(ctx, N_VIS, "VIS_INH");

  // PLACE neurons:
  int N_PLACE = 400;
  FloatT alpha_PLACE = 2.0;
  FloatT beta_PLACE = 0.6;
  FloatT tau_PLACE = 1e-2;
  RateNeurons PLACE(ctx, N_PLACE, "PLACE", alpha_PLACE, beta_PLACE, tau_PLACE);
  //PLACE.enable_homeostasis(0.02, 0.001, true);

  // FVxHD neurons:
  int N_FVxHD = 300;
  FloatT alpha_FVxHD = 20.0;
  FloatT beta_FVxHD = 0.3;
  FloatT tau_FVxHD = 1e-2;
  RateNeurons FVxHD(ctx, N_FVxHD, "FVxHD", alpha_FVxHD, beta_FVxHD, tau_FVxHD);

  // PLACExFVxHD neurons:
  int N_PLACExFVxHD = 800;
  FloatT alpha_PLACExFVxHD = 20.0;
  FloatT beta_PLACExFVxHD = 0.3;
  FloatT tau_PLACExFVxHD = 1e-2;
  RateNeurons PLACExFVxHD(ctx, N_PLACExFVxHD, "PLACExFVxHD",
                          alpha_PLACExFVxHD, beta_PLACExFVxHD,
                          tau_PLACExFVxHD);
  // PLACExFVxHD.enable_homeostasis(0.05, 0.001);

  printf("%d, %d, %d, %d, %d, %d\n", N_VIS, N_HD, N_FV, N_PLACE, N_FVxHD, N_PLACExFVxHD);

  // General parameters:
  FloatT axonal_delay = pow(2, -5); // seconds (TODO units)
  FloatT eps = toml_value.get<double>("eps");


  // FV -> FVxHD connectivity:
  FloatT FV_FVxHD_scaling = 270.0 / N_FV;

  // HD -> FVxHD connectivity:
  FloatT HD_FVxHD_sparsity = 0.05;
  FloatT HD_FVxHD_scaling = 450.0 / (HD_FVxHD_sparsity*N_HD);

  // FVxHD -> FVxHD connectivity:
  FloatT FVxHD_inhibition = -360.0 / N_FVxHD;

  // FVxHD -> PLACExFVxHD connectivity:
  FloatT FVxHD_PLACExFVxHD_sparsity = 0.05;
  FloatT FVxHD_PLACExFVxHD_scaling = toml_value.get<double>("FVxHD_PLACExFVxHD_scaling"); // 65.0 / (FVxHD_PLACExFVxHD_sparsity*N_FVxHD); // 700

  // PLACE -> PLACExFVxHD connectivity:
  FloatT PLACE_PLACExFVxHD_sparsity = 0.05;
  FloatT PLACE_PLACExFVxHD_scaling = toml_value.get<double>("PLACE_PLACExFVxHD_scaling"); // 685.0 / (N_PLACE*PLACE_PLACExFVxHD_sparsity); // 580

  // PLACExFVxHD -> PLACExFVxHD connectivity:
  FloatT PLACExFVxHD_inhibition = toml_value.get<double>("PLACExFVxHD_inhibition"); // -231.0 / N_PLACExFVxHD; // -720

  // VIS -> PLACE connectivity:
  FloatT VIS_PLACE_sparsity = 0.05;
  FloatT VIS_PLACE_scaling = toml_value.get<double>("VIS_PLACE_scaling"); // 1601.0 / (N_VIS*VIS_PLACE_sparsity); // 6x6: 2000
  FloatT VIS_PLACE_INH_scaling = toml_value.get<double>("VIS_PLACE_INH_scaling"); // -1.4 / (N_VIS*VIS_PLACE_sparsity);
  FloatT eps_VIS_PLACE = eps;

  // PLACExFVxHD -> PLACE connectivity:
  FloatT PLACExFVxHD_PLACE_sparsity = 1.0;
  FloatT PLACExFVxHD_PLACE_scaling = toml_value.get<double>("PLACExFVxHD_PLACE_scaling");
  // 8977.0 / (N_PLACExFVxHD*PLACExFVxHD_PLACE_sparsity); // 14000

  // PLACE -> PLACE connectivity:
  FloatT PLACE_inhibition = toml_value.get<double>("PLACE_inhibition");
  // -100.0 / N_PLACE; // -1190.0 / N_PLACE;
  /*
  FloatT PLACE_PLACE_sparsity = 0.25;
  FloatT PLACE_PLACE_scaling = 0.0; // 3000.0 / (N_PLACE*PLACE_PLACE_sparsity);
  */


  // FV -> FVxHD connectivity:
  RateSynapses FV_FVxHD(ctx, &FV, &FVxHD,
                        FV_FVxHD_scaling, "FV_FVxHD");
  EigenMatrix W_FV_FVxHD
    = Eigen::make_random_matrix(N_FVxHD, N_FV);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_FV_FVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_FV_FVxHD,
                       N_FVxHD, N_FV);
  }
  FV_FVxHD.weights(W_FV_FVxHD);

  BCMPlasticity plast_FV_FVxHD(ctx, &FV_FVxHD);

  FVxHD.connect_input(&FV_FVxHD, &plast_FV_FVxHD);


  // HD -> FVxHD connectivity:
  RateSynapses HD_FVxHD(ctx, &HD, &FVxHD,
                        HD_FVxHD_scaling, "HD_FVxHD");
  EigenMatrix W_HD_FVxHD
    = Eigen::make_random_matrix(N_FVxHD, N_HD,
                                1.0, true, 1.0-HD_FVxHD_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_HD_FVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_HD_FVxHD,
                       N_FVxHD, N_HD);
  }
  HD_FVxHD.weights(W_HD_FVxHD);
  HD_FVxHD.make_sparse();

  BCMPlasticity plast_HD_FVxHD(ctx, &HD_FVxHD);

  FVxHD.connect_input(&HD_FVxHD, &plast_HD_FVxHD);


  // FVxHD -> FVxHD connectivity:
  RateSynapses FVxHD_FVxHD_INH(ctx, &FVxHD, &FVxHD,
                               FVxHD_inhibition,
                               "FVxHD_FVxHD_INH");
  FVxHD_FVxHD_INH.weights(EigenMatrix::Ones(N_FVxHD, N_FVxHD));

  BCMPlasticity plast_FVxHD_FVxHD_INH(ctx, &FVxHD_FVxHD_INH);

  FVxHD.connect_input(&FVxHD_FVxHD_INH,
                      &plast_FVxHD_FVxHD_INH);


  // FVxHD -> PLACExFVxHD connectivity:
  RateSynapses FVxHD_PLACExFVxHD(ctx, &FVxHD, &PLACExFVxHD,
                                 FVxHD_PLACExFVxHD_scaling,
                                 "FVxHD_PLACExFVxHD");
  // FVxHD_PLACExFVxHD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_FVxHD_PLACExFVxHD
    = Eigen::make_random_matrix(N_PLACExFVxHD, N_FVxHD,
                                1.0, true, 1.0-FVxHD_PLACExFVxHD_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_FVxHD_PLACExFVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_FVxHD_PLACExFVxHD,
                       N_PLACExFVxHD, N_PLACE);
  }
  FVxHD_PLACExFVxHD.weights(W_FVxHD_PLACExFVxHD);
  FVxHD_PLACExFVxHD.make_sparse();

  BCMPlasticity plast_FVxHD_PLACExFVxHD(ctx, &FVxHD_PLACExFVxHD);

  PLACExFVxHD.connect_input(&FVxHD_PLACExFVxHD, &plast_FVxHD_PLACExFVxHD, 1.0);


  // PLACE -> PLACExFVxHD connectivity:
  RateSynapses PLACE_PLACExFVxHD(ctx, &PLACE, &PLACExFVxHD,
                                 PLACE_PLACExFVxHD_scaling,
                                 "PLACE_PLACExFVxHD");
  PLACE_PLACExFVxHD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_PLACE_PLACExFVxHD
    = Eigen::make_random_matrix(N_PLACExFVxHD, N_PLACE,
                                1.0, true, 1.0-PLACE_PLACExFVxHD_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_PLACE_PLACExFVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_PLACE_PLACExFVxHD,
                       N_PLACExFVxHD, N_PLACE);
  }
  PLACE_PLACExFVxHD.weights(W_PLACE_PLACExFVxHD);
  PLACE_PLACExFVxHD.make_sparse();

  BCMPlasticity plast_PLACE_PLACExFVxHD(ctx, &PLACE_PLACExFVxHD);

  PLACExFVxHD.connect_input(&PLACE_PLACExFVxHD, &plast_PLACE_PLACExFVxHD, 1.0);


  // PLACExFVxHD -> PLACExFVxHD connectivity:
  RateSynapses PLACExFVxHD_PLACExFVxHD_INH(ctx, &PLACExFVxHD, &PLACExFVxHD,
                                           PLACExFVxHD_inhibition,
                                           "PLACExFVxHD_PLACExFVxHD_INH");
  PLACExFVxHD_PLACExFVxHD_INH.weights
    (EigenMatrix::Ones(N_PLACExFVxHD, N_PLACExFVxHD));

  BCMPlasticity plast_PLACExFVxHD_PLACExFVxHD_INH
    (ctx, &PLACExFVxHD_PLACExFVxHD_INH);

  PLACExFVxHD.connect_input(&PLACExFVxHD_PLACExFVxHD_INH,
                            &plast_PLACExFVxHD_PLACExFVxHD_INH);


  // VIS -> PLACE connectivity:
  RateSynapses VIS_PLACE(ctx, &VIS, &PLACE, VIS_PLACE_scaling, "VIS_PLACE");
  EigenMatrix W_VIS_PLACE
    // = EigenMatrix::Ones(N_PLACE, N_VIS);
    = Eigen::make_random_matrix(N_PLACE, N_VIS, 1.0, true,
                                1.0-VIS_PLACE_sparsity, 0, false);
  // if (read_weights) {
  //   std::string tmp_path = weights_path + "/W_VIS_PLACE.bin";
  //   Eigen::read_binary(tmp_path.c_str(), W_VIS_PLACE, N_PLACE, N_VIS);
  // }
  VIS_PLACE.weights(W_VIS_PLACE);
  // VIS_PLACE.make_sparse();

  BCMPlasticity plast_VIS_PLACE(ctx, &VIS_PLACE);

  PLACE.connect_input(&VIS_PLACE, &plast_VIS_PLACE);

  RateSynapses VIS_PLACE_INH(ctx, &VIS_INH, &PLACE,
                             VIS_PLACE_INH_scaling, "VIS_PLACE_INH");
  VIS_PLACE_INH.weights(EigenMatrix::Ones(N_PLACE, N_VIS));

  BCMPlasticity plast_VIS_PLACE_INH(ctx, &VIS_PLACE_INH);

  PLACE.connect_input(&VIS_PLACE_INH, &plast_VIS_PLACE_INH);


  // PLACExFVxHD -> PLACE connectivity:
  RateSynapses PLACExFVxHD_PLACE(ctx, &PLACExFVxHD, &PLACE,
                                 PLACExFVxHD_PLACE_scaling,
                                 "PLACExFVxHD_PLACE");
  PLACExFVxHD_PLACE.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_PLACExFVxHD_PLACE
    = Eigen::make_random_matrix(N_PLACE, N_PLACExFVxHD,
                                1.0, true, 1.0-PLACExFVxHD_PLACE_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_PLACExFVxHD_PLACE.bin";
    Eigen::read_binary(tmp_path.c_str(), W_PLACExFVxHD_PLACE,
                       N_PLACE, N_PLACExFVxHD);
  }
  PLACExFVxHD_PLACE.weights(W_PLACExFVxHD_PLACE);
  PLACExFVxHD_PLACE.make_sparse();

  BCMPlasticity plast_PLACExFVxHD_PLACE(ctx, &PLACExFVxHD_PLACE);

  PLACE.connect_input(&PLACExFVxHD_PLACE, &plast_PLACExFVxHD_PLACE);


  // PLACE -> PLACE connectivity:
  RateSynapses PLACE_PLACE_INH(ctx, &PLACE, &PLACE,
                             PLACE_inhibition, "PLACE_PLACE_INH");
  PLACE_PLACE_INH.weights(EigenMatrix::Ones(N_PLACE, N_PLACE));
  // PLACE_PLACE_INH.delay(ceil(0.5*axonal_delay / timestep)); // ?

  BCMPlasticity plast_PLACE_PLACE_INH(ctx, &PLACE_PLACE_INH);

  PLACE.connect_input(&PLACE_PLACE_INH, &plast_PLACE_PLACE_INH); //, 1.0);


  // Set simulation schedule:
  VIS.t_stop_after = train_time + test_on_time;
  VIS_INH.add_schedule(VIS.t_stop_after, VIS_INH_on);
  VIS_INH.add_schedule(infinity<FloatT>(), VIS_INH_off);

  // No inhibitory plasticity:
  plast_FVxHD_FVxHD_INH.add_schedule(infinity<FloatT>(), 0);
  plast_PLACExFVxHD_PLACExFVxHD_INH.add_schedule(infinity<FloatT>(), 0);
  plast_PLACE_PLACE_INH.add_schedule(infinity<FloatT>(), 0);

  // Rest have plasticity on only during training, with params above:
  auto add_plast_schedule = [&](FloatT dur, FloatT rate) {
    plast_VIS_PLACE.add_schedule(dur, rate);
    plast_PLACExFVxHD_PLACE.add_schedule(dur, rate);
    plast_PLACE_PLACExFVxHD.add_schedule(dur, rate);
    plast_FVxHD_PLACExFVxHD.add_schedule(dur, rate);
    plast_FV_FVxHD.add_schedule(dur, rate);
    plast_HD_FVxHD.add_schedule(dur, rate);
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
  RateElectrodes FVxHD_elecs(output_path, &FVxHD);
  RateElectrodes PLACExFVxHD_elecs(output_path, &PLACExFVxHD);
  RateElectrodes PLACE_elecs(output_path, &PLACE);


  // Add Agent, Neurons and Electrodes to Model
  model.add(&agent);

  model.add(&VIS);
  model.add(&VIS_INH);

  model.add(&HD);
  model.add(&FV);
  model.add(&FVxHD);
  model.add(&PLACExFVxHD);
  model.add(&PLACE);

  model.add(&VIS_elecs);

  model.add(&HD_elecs);
  model.add(&FV_elecs);
  model.add(&FVxHD_elecs);
  model.add(&PLACExFVxHD_elecs);
  model.add(&PLACE_elecs);


  // Ensure VIS input is working (hacky fix for weird bug)
  model.add_hook([&]() {
      float t = model.current_time();
      if (t > 4 && t < 5) {
        if (VIS.mean_rate() < 0.01) exit(127);
      }
    });


  // Set simulation time parameters:
  model.set_simulation_time(train_time + test_on_time + test_off_time, timestep);
  model.set_buffer_intervals((float)buffer_timestep); // TODO: Use proper units
  model.set_weights_buffer_interval(ceil(10.0/timestep));
  model.set_buffer_start(start_recording_time);

  agent.save_map(output_path);
  agent.record_history(output_path, round(buffer_timestep/timestep),
                       round(start_recording_time/timestep));


  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.current_time(), model.dt, model.t_stop, model.timesteps());
}
