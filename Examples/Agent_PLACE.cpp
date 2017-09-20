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

  FloatT timestep = pow(2, -9); // seconds (TODO units)
  FloatT buffer_timestep = pow(2, -6);
  FloatT train_time = 6000; // 300
  if (read_weights) train_time = 0;
  FloatT test_on_time = 800;
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

  FloatT radius = 1.0;
  FloatT bound_x = 1.0;
  FloatT bound_y = bound_x;

  agent.set_boundary(bound_x, bound_y);

  agent.add_proximal_object(-0.75*bound_x, -0.25*bound_y);
  agent.add_proximal_object(-0.25*bound_x, -0.25*bound_y);
  agent.add_proximal_object(0.25*bound_x, -0.25*bound_y);
  agent.add_proximal_object(0.75*bound_x, -0.25*bound_y);

  agent.add_proximal_object(-0.75*bound_x, 0.25*bound_y);
  agent.add_proximal_object(-0.25*bound_x, 0.25*bound_y);
  agent.add_proximal_object(0.25*bound_x, 0.25*bound_y);
  agent.add_proximal_object(0.75*bound_x, 0.25*bound_y);

  agent.add_proximal_object(-0.75*bound_x, -0.75*bound_y);
  agent.add_proximal_object(-0.25*bound_x, -0.75*bound_y);
  agent.add_proximal_object(0.25*bound_x, -0.75*bound_y);
  agent.add_proximal_object(0.75*bound_x, -0.75*bound_y);

  agent.add_proximal_object(-0.75*bound_x, 0.75*bound_y);
  agent.add_proximal_object(-0.25*bound_x, 0.75*bound_y);
  agent.add_proximal_object(0.25*bound_x, 0.75*bound_y);
  agent.add_proximal_object(0.75*bound_x, 0.75*bound_y);

  // agent.add_proximal_object(0.4*bound_x, 0.4*bound_y);
  // agent.add_proximal_object(0.4*bound_x, -0.4*bound_y);
  // agent.add_proximal_object(-0.4*bound_x, 0.4*bound_y);
  // agent.add_proximal_object(-0.4*bound_x, -0.4*bound_y);

  // agent.add_proximal_object(0, 0.6*bound_y);
  // agent.add_proximal_object(0, -0.6*bound_y);
  // agent.add_proximal_object(0.6*bound_x, 0);
  // agent.add_proximal_object(-0.6*bound_x, 0);

  // agent.add_proximal_object(-0.1*bound_x, 0.1*bound_y);
  // agent.add_proximal_object(0.1*bound_x, -0.1*bound_y);

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

  FloatT fwd_move_dist = 0.05;
  FloatT rot_angle = M_PI / 8;

  FloatT fwd_move_time = 0.1; ///6.0; // seconds per forward move
  FloatT angle_move_time = 0.1; ///6.0; // seconds per angular move

  // agent.p_fwd = 1.0/2.0;
  agent.p_fwd = 1.0/3.0;

  agent.add_AHV(rot_angle / angle_move_time, angle_move_time);
  agent.add_AHV(-rot_angle / angle_move_time, angle_move_time);

  agent.add_FV(fwd_move_dist / fwd_move_time, fwd_move_time);

  agent.add_test_time(100);
  agent.add_test_time(500);
  agent.add_test_time(1000);
  agent.add_test_time(2000);
  agent.add_test_time(3000);
  agent.add_test_time(4000);
  agent.add_test_time(5000);
  agent.add_test_time(6000);
  agent.set_place_test_params(0.2*radius, 20);

  agent.add_test_position(-0.5*bound_x, 0.5*bound_y);
  agent.add_test_position(-0.5*bound_x, -0.5*bound_y);
  agent.add_test_position(0.5*bound_x, 0.5*bound_y);
  agent.add_test_position(0.5*bound_x, -0.5*bound_y);
  agent.add_test_position(0, 0);

  // agent.add_test_position(0.75*bound_x, 0.75*bound_x);
  // agent.add_test_position(-0.75*bound_x, -0.75*bound_x);
  // agent.add_test_position(0.4*bound_x, 0.4*bound_x);
  // agent.add_test_position(0.4*bound_x, 0.4*bound_x);
  // agent.add_test_position(0.4*bound_x, -0.4*bound_x);
  // agent.add_test_position(-0.4*bound_x, 0.4*bound_y);
  // agent.add_test_position(-0.4*bound_x, -0.4*bound_y);
 
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
  int N_PLACE = 100;
  FloatT alpha_PLACE = 20.0;
  FloatT beta_PLACE = 0.3;
  FloatT tau_PLACE = 1e-2;
  RateNeurons PLACE(ctx, N_PLACE, "PLACE", alpha_PLACE, beta_PLACE, tau_PLACE);

  // GRID neurons:
  int N_GRID = 300;
  FloatT alpha_GRID = 20.0;
  FloatT beta_GRID = 0.3;
  FloatT tau_GRID = 1e-2;
  RateNeurons GRID(ctx, N_GRID, "GRID", alpha_GRID, beta_GRID, tau_GRID);

  // FVxHD neurons:
  int N_FVxHD = N_HD * agent.num_FV_states;
  FloatT alpha_FVxHD = 20.0;
  FloatT beta_FVxHD = 0.3;
  FloatT tau_FVxHD = 1e-2;
  RateNeurons FVxHD(ctx, N_FVxHD, "FVxHD", alpha_FVxHD, beta_FVxHD, tau_FVxHD);

  // GRIDxFVxHD neurons:
  int N_GRIDxFVxHD = N_GRID * 2 * agent.num_FV_states;
  FloatT alpha_GRIDxFVxHD = 20.0;
  FloatT beta_GRIDxFVxHD = 0.4;
  FloatT tau_GRIDxFVxHD = 1e-2;
  RateNeurons GRIDxFVxHD(ctx, N_GRIDxFVxHD, "GRIDxFVxHD",
                         alpha_GRIDxFVxHD, beta_GRIDxFVxHD,
                         tau_GRIDxFVxHD);

  // General parameters:
  FloatT axonal_delay = 1e-2; // seconds (TODO units)
  FloatT eps = 0.02;


  // FV -> FVxHD connectivity:
  FloatT FV_FVxHD_scaling = 270.0 / N_FV;

  // HD -> FVxHD connectivity:
  FloatT HD_FVxHD_sparsity = 0.05;
  FloatT HD_FVxHD_scaling = 450.0 / (HD_FVxHD_sparsity*N_HD);

  // FVxHD -> FVxHD connectivity:
  FloatT FVxHD_inhibition = -360.0 / N_FVxHD;

  // FVxHD -> GRIDxFVxHD connectivity:
  FloatT FVxHD_GRIDxFVxHD_sparsity = 0.05;
  FloatT FVxHD_GRIDxFVxHD_scaling = 770.0 / (FVxHD_GRIDxFVxHD_sparsity*N_FVxHD);

  // GRID -> GRIDxFVxHD connectivity:
  FloatT GRID_GRIDxFVxHD_sparsity = 0.05;
  FloatT GRID_GRIDxFVxHD_scaling = 500.0 / (N_GRID*GRID_GRIDxFVxHD_sparsity);

  // GRIDxFVxHD -> GRIDxFVxHD connectivity:
  FloatT GRIDxFVxHD_inhibition = -120.0 / N_GRIDxFVxHD;

  // VIS -> GRID connectivity:
  FloatT VIS_GRID_sparsity = 0.05;
  FloatT VIS_GRID_scaling = 500.0 / (N_GRID*VIS_GRID_sparsity);
  FloatT VIS_GRID_INH_scaling = -1.05 / (N_VIS*VIS_GRID_sparsity);
  FloatT eps_VIS_GRID = 0.06;

  // GRIDxFVxHD -> GRID connectivity:
  FloatT GRIDxFVxHD_GRID_scaling = 6000.0 / N_GRIDxFVxHD;

  // GRID -> GRID connectivity:
  FloatT GRID_inhibition = -900.0 / N_GRID;

  // GRID -> PLACE connectivity
  FloatT GRID_PLACE_sparsity = 0.4;
  FloatT GRID_PLACE_scaling = 25000.0 / (N_GRID*GRID_PLACE_sparsity);

  // PLACE -> PLACE connectivity:
  FloatT PLACE_inhibition = -3600.0 / N_PLACE;

  // PLACE -> GRID connectivity
  FloatT PLACE_GRID_sparsity = 0.1;
  FloatT PLACE_GRID_scaling = 180.0 / (N_PLACE*PLACE_GRID_sparsity);


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


  // FVxHD -> GRIDxFVxHD connectivity:
  RateSynapses FVxHD_GRIDxFVxHD(ctx, &FVxHD, &GRIDxFVxHD,
                                 FVxHD_GRIDxFVxHD_scaling,
                                 "FVxHD_GRIDxFVxHD");
  FVxHD_GRIDxFVxHD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_FVxHD_GRIDxFVxHD
    = Eigen::make_random_matrix(N_GRIDxFVxHD, N_GRID,
                                1.0, true, 1.0-FVxHD_GRIDxFVxHD_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_FVxHD_GRIDxFVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_FVxHD_GRIDxFVxHD,
                       N_GRIDxFVxHD, N_GRID);
  }
  FVxHD_GRIDxFVxHD.weights(W_FVxHD_GRIDxFVxHD);
  FVxHD_GRIDxFVxHD.make_sparse();

  BCMPlasticity plast_FVxHD_GRIDxFVxHD(ctx, &FVxHD_GRIDxFVxHD);

  GRIDxFVxHD.connect_input(&FVxHD_GRIDxFVxHD, &plast_FVxHD_GRIDxFVxHD);


  // GRID -> GRIDxFVxHD connectivity:
  RateSynapses GRID_GRIDxFVxHD(ctx, &GRID, &GRIDxFVxHD,
                                 GRID_GRIDxFVxHD_scaling,
                                 "GRID_GRIDxFVxHD");
  GRID_GRIDxFVxHD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_GRID_GRIDxFVxHD
    = Eigen::make_random_matrix(N_GRIDxFVxHD, N_GRID,
                                1.0, true, 1.0-GRID_GRIDxFVxHD_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_GRID_GRIDxFVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_GRID_GRIDxFVxHD,
                       N_GRIDxFVxHD, N_GRID);
  }
  GRID_GRIDxFVxHD.weights(W_GRID_GRIDxFVxHD);
  GRID_GRIDxFVxHD.make_sparse();

  BCMPlasticity plast_GRID_GRIDxFVxHD(ctx, &GRID_GRIDxFVxHD);

  GRIDxFVxHD.connect_input(&GRID_GRIDxFVxHD, &plast_GRID_GRIDxFVxHD);


  // GRIDxFVxHD -> GRIDxFVxHD connectivity:
  RateSynapses GRIDxFVxHD_GRIDxFVxHD_INH(ctx, &GRIDxFVxHD, &GRIDxFVxHD,
                                           GRIDxFVxHD_inhibition,
                                           "GRIDxFVxHD_GRIDxFVxHD_INH");
  GRIDxFVxHD_GRIDxFVxHD_INH.weights
    (EigenMatrix::Ones(N_GRIDxFVxHD, N_GRIDxFVxHD));

  BCMPlasticity plast_GRIDxFVxHD_GRIDxFVxHD_INH
    (ctx, &GRIDxFVxHD_GRIDxFVxHD_INH);

  GRIDxFVxHD.connect_input(&GRIDxFVxHD_GRIDxFVxHD_INH,
                            &plast_GRIDxFVxHD_GRIDxFVxHD_INH);


  // VIS -> GRID connectivity:
  RateSynapses VIS_GRID(ctx, &VIS, &GRID, VIS_GRID_scaling, "VIS_GRID");
  EigenMatrix W_VIS_GRID
    = Eigen::make_random_matrix(N_GRID, N_VIS, 1.0, true,
                                1.0-VIS_GRID_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_VIS_GRID.bin";
    Eigen::read_binary(tmp_path.c_str(), W_VIS_GRID, N_GRID, N_VIS);
  }
  VIS_GRID.weights(W_VIS_GRID);
  VIS_GRID.make_sparse();

  BCMPlasticity plast_VIS_GRID(ctx, &VIS_GRID);

  GRID.connect_input(&VIS_GRID, &plast_VIS_GRID);

  RateSynapses VIS_GRID_INH(ctx, &VIS_INH, &GRID,
                             VIS_GRID_INH_scaling, "VIS_GRID_INH");
  VIS_GRID_INH.weights(EigenMatrix::Ones(N_GRID, N_VIS));

  BCMPlasticity plast_VIS_GRID_INH(ctx, &VIS_GRID_INH);

  GRID.connect_input(&VIS_GRID_INH, &plast_VIS_GRID_INH);


  // GRIDxFVxHD -> GRID connectivity:
  RateSynapses GRIDxFVxHD_GRID(ctx, &GRIDxFVxHD, &GRID,
                                 GRIDxFVxHD_GRID_scaling,
                                 "GRIDxFVxHD_GRID");
  GRIDxFVxHD_GRID.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_GRIDxFVxHD_GRID
    = Eigen::make_random_matrix(N_GRID, N_GRIDxFVxHD);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_GRIDxFVxHD_GRID.bin";
    Eigen::read_binary(tmp_path.c_str(), W_GRIDxFVxHD_GRID,
                       N_GRID, N_GRIDxFVxHD);
  }
  GRIDxFVxHD_GRID.weights(W_GRIDxFVxHD_GRID);

  BCMPlasticity plast_GRIDxFVxHD_GRID(ctx, &GRIDxFVxHD_GRID);

  GRID.connect_input(&GRIDxFVxHD_GRID, &plast_GRIDxFVxHD_GRID);


  // GRID -> GRID connectivity:
  RateSynapses GRID_GRID_INH(ctx, &GRID, &GRID,
                               GRID_inhibition, "GRID_GRID_INH");
  GRID_GRID_INH.weights(EigenMatrix::Ones(N_GRID, N_GRID));

  BCMPlasticity plast_GRID_GRID_INH(ctx, &GRID_GRID_INH);

  GRID.connect_input(&GRID_GRID_INH, &plast_GRID_GRID_INH);


  // GRID -> PLACE connectivity:
  RateSynapses GRID_PLACE(ctx, &GRID, &PLACE,
                                 GRID_PLACE_scaling,
                                 "GRID_PLACE");
  GRID_PLACE.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_GRID_PLACE
    = Eigen::make_random_matrix(N_PLACE, N_GRID,
                                1.0, true, 1.0-GRID_PLACE_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_GRID_PLACE.bin";
    Eigen::read_binary(tmp_path.c_str(), W_GRID_PLACE,
                       N_PLACE, N_GRID);
  }
  GRID_PLACE.weights(W_GRID_PLACE);
  GRID_PLACE.make_sparse();

  BCMPlasticity plast_GRID_PLACE(ctx, &GRID_PLACE);

  PLACE.connect_input(&GRID_PLACE, &plast_GRID_PLACE);


  // PLACE -> PLACE connectivity:
  RateSynapses PLACE_PLACE_INH(ctx, &PLACE, &PLACE,
                               PLACE_inhibition, "PLACE_PLACE_INH");
  PLACE_PLACE_INH.weights(EigenMatrix::Ones(N_PLACE, N_PLACE));

  BCMPlasticity plast_PLACE_PLACE_INH(ctx, &PLACE_PLACE_INH);

  PLACE.connect_input(&PLACE_PLACE_INH, &plast_PLACE_PLACE_INH);


  // PLACE -> GRID connectivity:
  RateSynapses PLACE_GRID(ctx, &PLACE, &GRID,
                          PLACE_GRID_scaling,
                          "PLACE_GRID");
  PLACE_GRID.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_PLACE_GRID
    = Eigen::make_random_matrix(N_GRID, N_PLACE,
                                1.0, true, 1.0-PLACE_GRID_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_PLACE_GRID.bin";
    Eigen::read_binary(tmp_path.c_str(), W_PLACE_GRID,
                       N_GRID, N_PLACE);
  }
  PLACE_GRID.weights(W_PLACE_GRID);
  PLACE_GRID.make_sparse();

  BCMPlasticity plast_PLACE_GRID(ctx, &PLACE_GRID);

  GRID.connect_input(&PLACE_GRID, &plast_PLACE_GRID);


  // Set simulation schedule:
  VIS.t_stop_after = train_time + test_on_time;
  VIS_INH.add_schedule(VIS.t_stop_after, VIS_INH_on);
  VIS_INH.add_schedule(infinity<FloatT>(), VIS_INH_off);

  // No inhibitory plasticity:
  plast_GRID_GRID_INH.add_schedule(infinity<FloatT>(), 0);
  plast_FVxHD_FVxHD_INH.add_schedule(infinity<FloatT>(), 0);
  plast_GRIDxFVxHD_GRIDxFVxHD_INH.add_schedule(infinity<FloatT>(), 0);
  plast_PLACE_PLACE_INH.add_schedule(infinity<FloatT>(), 0);

  // Rest have plasticity on only during training, with params above:
  plast_VIS_GRID.add_schedule(train_time, eps_VIS_GRID);
  plast_GRIDxFVxHD_GRID.add_schedule(train_time, eps*0.8);
  plast_GRID_GRIDxFVxHD.add_schedule(train_time, eps*1.2);
  plast_FVxHD_GRIDxFVxHD.add_schedule(train_time, eps);
  plast_GRID_PLACE.add_schedule(train_time, eps);
  plast_PLACE_GRID.add_schedule(train_time, eps);
  plast_FV_FVxHD.add_schedule(train_time, eps);
  plast_HD_FVxHD.add_schedule(train_time, eps);

  plast_VIS_GRID.add_schedule(infinity<FloatT>(), 0);
  plast_GRIDxFVxHD_GRID.add_schedule(infinity<FloatT>(), 0);
  plast_GRID_GRIDxFVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_FVxHD_GRIDxFVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_GRID_PLACE.add_schedule(infinity<FloatT>(), 0);
  plast_PLACE_GRID.add_schedule(infinity<FloatT>(), 0);
  plast_FV_FVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_HD_FVxHD.add_schedule(infinity<FloatT>(), 0);


  // Have to construct electrodes after neurons:
  RateElectrodes VIS_elecs("PLACE_out", &VIS);
  RateElectrodes HD_elecs("PLACE_out", &HD);
  RateElectrodes FV_elecs("PLACE_out", &FV);
  RateElectrodes FVxHD_elecs("PLACE_out", &FVxHD);

  RateElectrodes GRID_elecs("PLACE_out", &GRID);
  RateElectrodes GRIDxFVxHD_elecs("PLACE_out", &GRIDxFVxHD);

  RateElectrodes PLACE_elecs("PLACE_out", &PLACE);


  // Add Agent, Neurons and Electrodes to Model
  model.add(&agent);

  model.add(&VIS);
  model.add(&VIS_INH);

  model.add(&HD);
  model.add(&FV);
  model.add(&FVxHD);
  model.add(&GRIDxFVxHD);
  model.add(&GRID);
  model.add(&PLACE);

  model.add(&VIS_elecs);

  model.add(&HD_elecs);
  model.add(&FV_elecs);
  model.add(&FVxHD_elecs);
  model.add(&GRIDxFVxHD_elecs);
  model.add(&GRID_elecs);
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
