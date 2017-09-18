#include "Spike/Models/RateModel.hpp"
#include <fenv.h>
#include <omp.h>

#define ENABLE_PLACE

// TODO: Add signal handlers

int main(int argc, char *argv[]) {
  Eigen::initParallel();
  omp_set_num_threads(32);
  Eigen::setNbThreads(8);
  std::cout << Eigen::nbThreads() << std::endl;
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  bool read_weights = false;
  std::string weights_path;

  if (argc == 2) {
    read_weights = true;
    weights_path = argv[1];
  }

  FloatT timestep = pow(2, -10); // seconds (TODO units)
  FloatT buffer_timestep = pow(2, -6);
  FloatT train_time = 4000; // 300
  if (read_weights) train_time = 0;
  FloatT test_on_time = 200;
  FloatT test_off_time = 200;
  FloatT start_recording_time = 0;
  if (read_weights) start_recording_time = 0;
  
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  ctx->verbose = true;
  ctx->backend = "Eigen";

  // Create Agent
  Agent agent;
  agent.seed(123);

  FloatT radius = 2;
  FloatT bound_x = 2;
  FloatT bound_y = bound_x;

  agent.set_boundary(bound_x, bound_y);

  agent.add_proximal_object(0.4*bound_x, 0.4*bound_y);
  agent.add_proximal_object(0.4*bound_x, -0.4*bound_y);
  agent.add_proximal_object(-0.4*bound_x, 0.4*bound_y);
  agent.add_proximal_object(-0.4*bound_x, -0.4*bound_y);

  agent.add_proximal_object(0, 0.6*bound_y);
  agent.add_proximal_object(0, -0.6*bound_y);
  agent.add_proximal_object(0.6*bound_x, 0);
  agent.add_proximal_object(-0.6*bound_x, 0);

  /*
  agent.add_proximal_object(0.8*bound_x, 0.8*bound_y);
  agent.add_proximal_object(0.8*bound_x, -0.8*bound_y);
  agent.add_proximal_object(-0.8*bound_x, 0.8*bound_y);
  agent.add_proximal_object(-0.8*bound_x, -0.8*bound_y);
  /**//*
  std::default_random_engine engine;
  std::uniform_real_distribution<FloatT> scale(-1, 1);
  for (int i = 0; i < 8; ++i) {
    agent.add_proximal_object(scale(engine)*bound_x, scale(engine)*bound_y);
  }
  */

  FloatT fwd_move_dist = 0.4 * radius;
  FloatT rot_angle = M_PI / 2;

  FloatT fwd_move_time = 0.5; ///6.0; // seconds per forward move
  FloatT angle_move_time = 0.5; ///6.0; // seconds per angular move

  agent.p_fwd = 1.0/3.0;

  agent.add_AHV(rot_angle / angle_move_time, angle_move_time);
  agent.add_AHV(-rot_angle / angle_move_time, angle_move_time);

  agent.add_FV(fwd_move_dist / fwd_move_time, fwd_move_time);

  agent.add_test_time(100);
  agent.add_test_time(500);
  agent.add_test_time(1000);
  agent.add_test_time(2000);
  agent.add_test_time(4000);
  agent.set_place_test_params(0.1, 16);
  agent.add_test_position(0.4, 0.4);
  agent.add_test_position(0.4, -0.4);
  agent.add_test_position(-0.4, 0.4);
  agent.add_test_position(-0.4, -0.4);
 
  int N_per_obj = 100;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 1.0;

  int N_HD = 300;
  int N_per_state = 50;

  // Input neurons:
  AgentVISRateNeurons VIS(ctx, &agent, N_per_obj, sigma_VIS, lambda_VIS, "VIS");
  AgentHDRateNeurons HD(ctx, &agent, N_HD, sigma_VIS, lambda_VIS, "HD");
  AgentFVRateNeurons FV(ctx, &agent, N_per_state, "FV");

  int N_VIS = VIS.size;
  int N_FV = FV.size;

  // HD neurons:
  EigenVector VIS_INH_on = EigenVector::Ones(N_VIS);
  EigenVector VIS_INH_off = EigenVector::Zero(N_VIS);
  DummyRateNeurons VIS_INH(ctx, N_VIS, "VIS_INH");

  // PLACE neurons:
  int N_PLACE = N_HD;
  FloatT alpha_PLACE = 20.0;
  FloatT beta_PLACE = 0.6;
  FloatT tau_PLACE = 1e-2;
  RateNeurons PLACE(ctx, N_PLACE, "PLACE", alpha_PLACE, beta_PLACE, tau_PLACE);

  /*
  // FVxHD neurons:
  int N_FVxHD = N_HD * agent.num_FV_states;
  FloatT alpha_FVxHD = 20.0;
  FloatT beta_FVxHD = 1.2;
  FloatT tau_FVxHD = 1e-2;
  RateNeurons FVxHD(ctx, N_FVxHD, "FVxHD", alpha_FVxHD, beta_FVxHD, tau_FVxHD);
  */

  // PLACExFVxHD neurons:
  int N_PLACExFVxHD = N_PLACE * agent.num_FV_states * agent.num_AHV_states;
  FloatT alpha_PLACExFVxHD = 20.0;
  FloatT beta_PLACExFVxHD = 0.6;
  FloatT tau_PLACExFVxHD = 1e-2;
  RateNeurons PLACExFVxHD(ctx, N_PLACExFVxHD, "PLACExFVxHD",
                          alpha_PLACExFVxHD, beta_PLACExFVxHD,
                          tau_PLACExFVxHD);

  // General parameters:
  FloatT axonal_delay = 1e-2; // seconds (TODO units)
  FloatT eps = 0.02;


  // FV -> PLACExFVxHD connectivity:
  FloatT FV_PLACExFVxHD_scaling = 100.0 / N_FV;

  // HD -> PLACExFVxHD connectivity:
  FloatT HD_PLACExFVxHD_sparsity = 0.134;
  FloatT HD_PLACExFVxHD_scaling = 100.0 / (HD_PLACExFVxHD_sparsity*N_HD);

  // PLACE -> PLACExFVxHD connectivity:
  FloatT PLACE_PLACExFVxHD_sparsity = 0.05;
  FloatT PLACE_PLACExFVxHD_scaling = 280.0 / (N_PLACE*PLACE_PLACExFVxHD_sparsity);

  // PLACExFVxHD -> PLACExFVxHD connectivity:
  FloatT PLACExFVxHD_inhibition = -160.0 / N_PLACExFVxHD;

  // VIS -> PLACE connectivity:
  FloatT VIS_PLACE_sparsity = 0.139;
  FloatT VIS_PLACE_scaling = 1600.0 / (N_PLACE*VIS_PLACE_sparsity);
  FloatT VIS_PLACE_INH_scaling = -2.0 / (N_VIS*VIS_PLACE_sparsity);
  FloatT eps_VIS_PLACE = 0.06;

  // PLACExFVxHD -> PLACE connectivity:
  FloatT PLACExFVxHD_PLACE_scaling = 6800.0 / N_PLACExFVxHD;

  // PLACE -> PLACE connectivity:
  FloatT PLACE_inhibition = -600.0 / N_PLACE;

  // FV -> PLACExFVxHD connectivity:
  RateSynapses FV_PLACExFVxHD(ctx, &FV, &PLACExFVxHD,
                              FV_PLACExFVxHD_scaling, "FV_PLACExFVxHD");
  EigenMatrix W_FV_PLACExFVxHD
    = Eigen::make_random_matrix(N_PLACExFVxHD, N_FV);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_FV_PLACExFVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_FV_PLACExFVxHD,
                       N_PLACExFVxHD, N_FV);
  }
  FV_PLACExFVxHD.weights(W_FV_PLACExFVxHD);

  BCMPlasticity plast_FV_PLACExFVxHD(ctx, &FV_PLACExFVxHD);

  PLACExFVxHD.connect_input(&FV_PLACExFVxHD, &plast_FV_PLACExFVxHD);


  // HD -> PLACExFVxHD connectivity:
  RateSynapses HD_PLACExFVxHD(ctx, &HD, &PLACExFVxHD,
                              HD_PLACExFVxHD_scaling, "HD_PLACExFVxHD");
  EigenMatrix W_HD_PLACExFVxHD
    = Eigen::make_random_matrix(N_PLACExFVxHD, N_HD);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_HD_PLACExFVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_HD_PLACExFVxHD,
                       N_PLACExFVxHD, N_HD);
  }
  HD_PLACExFVxHD.weights(W_HD_PLACExFVxHD);

  BCMPlasticity plast_HD_PLACExFVxHD(ctx, &HD_PLACExFVxHD);

  PLACExFVxHD.connect_input(&HD_PLACExFVxHD, &plast_HD_PLACExFVxHD);


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

  PLACExFVxHD.connect_input(&PLACE_PLACExFVxHD, &plast_PLACE_PLACExFVxHD);


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
    = Eigen::make_random_matrix(N_PLACE, N_VIS, 1.0, true,
                                1.0-VIS_PLACE_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_VIS_PLACE.bin";
    Eigen::read_binary(tmp_path.c_str(), W_VIS_PLACE, N_PLACE, N_VIS);
  }
  VIS_PLACE.weights(W_VIS_PLACE);
  VIS_PLACE.make_sparse();

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
    = Eigen::make_random_matrix(N_PLACE, N_PLACExFVxHD);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_PLACExFVxHD_PLACE.bin";
    Eigen::read_binary(tmp_path.c_str(), W_PLACExFVxHD_PLACE,
                       N_PLACE, N_PLACExFVxHD);
  }
  PLACExFVxHD_PLACE.weights(W_PLACExFVxHD_PLACE);

  BCMPlasticity plast_PLACExFVxHD_PLACE(ctx, &PLACExFVxHD_PLACE);

  PLACE.connect_input(&PLACExFVxHD_PLACE, &plast_PLACExFVxHD_PLACE);


  // PLACE -> PLACE connectivity:
  RateSynapses PLACE_PLACE_INH(ctx, &PLACE, &PLACE,
                               PLACE_inhibition, "PLACE_PLACE_INH");
  PLACE_PLACE_INH.weights(EigenMatrix::Ones(N_PLACE, N_PLACE));

  BCMPlasticity plast_PLACE_PLACE_INH(ctx, &PLACE_PLACE_INH);

  PLACE.connect_input(&PLACE_PLACE_INH, &plast_PLACE_PLACE_INH);


  // Set simulation schedule:
  VIS.t_stop_after = train_time + test_on_time;
  VIS_INH.add_schedule(VIS.t_stop_after, VIS_INH_on);
  VIS_INH.add_schedule(infinity<FloatT>(), VIS_INH_off);

  // No inhibitory plasticity:
  plast_PLACE_PLACE_INH.add_schedule(infinity<FloatT>(), 0);
  plast_PLACExFVxHD_PLACExFVxHD_INH.add_schedule(infinity<FloatT>(), 0);

  // Rest have plasticity on only during training, with params above:
  plast_VIS_PLACE.add_schedule(train_time, eps_VIS_PLACE);
  plast_PLACExFVxHD_PLACE.add_schedule(train_time, eps*0.8);
  plast_PLACE_PLACExFVxHD.add_schedule(train_time, eps*1.2);
  plast_FV_PLACExFVxHD.add_schedule(train_time, eps);
  plast_HD_PLACExFVxHD.add_schedule(train_time, eps);

  plast_VIS_PLACE.add_schedule(infinity<FloatT>(), 0);
  plast_PLACExFVxHD_PLACE.add_schedule(infinity<FloatT>(), 0);
  plast_PLACE_PLACExFVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_FV_PLACExFVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_HD_PLACExFVxHD.add_schedule(infinity<FloatT>(), 0);


  // Have to construct electrodes after neurons:
  RateElectrodes VIS_elecs("PLACE_out", &VIS);
  RateElectrodes HD_elecs("PLACE_out", &HD);
  RateElectrodes FV_elecs("PLACE_out", &FV);

  RateElectrodes PLACE_elecs("PLACE_out", &PLACE);
  RateElectrodes PLACExFVxHD_elecs("PLACE_out", &PLACExFVxHD);


  // Add Agent, Neurons and Electrodes to Model
  model.add(&agent);

  model.add(&VIS);
  model.add(&VIS_INH);

  model.add(&HD);
  model.add(&FV);
  model.add(&PLACExFVxHD);
  model.add(&PLACE);

  model.add(&VIS_elecs);

  model.add(&HD_elecs);
  model.add(&FV_elecs);
  model.add(&PLACExFVxHD_elecs);
  model.add(&PLACE_elecs);


  // Set simulation time parameters:
  model.set_simulation_time(train_time + test_on_time + test_off_time, timestep);
  model.set_buffer_intervals((float)buffer_timestep); // TODO: Use proper units
  model.set_weights_buffer_interval(ceil(10.0/timestep));
  model.set_buffer_start(start_recording_time);

  agent.save_map("PLACE_out");
  agent.record_history("PLACE_out", round(buffer_timestep/timestep),
                       round(start_recording_time/timestep));


  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
