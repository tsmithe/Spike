#include "Spike/Models/RateModel.hpp"
#include "Spike/Models/RateAgent.hpp"
#include <fenv.h>
#include <omp.h>

//#define ENABLE_PLACE

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
  FloatT train_time = 4000; // 300
  if (read_weights) train_time = 0;
  FloatT test_on_time = 100;
  FloatT test_off_time = 0;
  FloatT start_recording_time = 0;
  if (read_weights) start_recording_time = 0;
  
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  ctx->verbose = true;
  ctx->backend = "Eigen";

  // Create Agent
  Agent<RandomWalkPolicy, ScanWalkTestPolicy> agent;
  // agent.seed(123);

  FloatT radius = 2;
  FloatT bound_x = 2;
  FloatT bound_y = bound_x;

  agent.set_boundary(bound_x, bound_y);

  agent.add_distal_object(0);
  agent.add_distal_object(0.25 * M_PI);
  agent.add_distal_object(0.5 * M_PI);
  agent.add_distal_object(0.75 * M_PI);
  agent.add_distal_object(M_PI);
  agent.add_distal_object(1.25 * M_PI);
  agent.add_distal_object(1.5 * M_PI);
  agent.add_distal_object(1.75 * M_PI);

  // agent.add_distal_object(2.0 * M_PI / 3.0);
  // agent.add_distal_object(4.0 * M_PI / 3.0);

  FloatT start_x = -0.75;
  FloatT stop_x = 0.75;
  FloatT start_y = -0.75;
  FloatT stop_y = 0.75;
  unsigned n_objs_x = 4;
  unsigned n_objs_y = 2;

  for (unsigned i = 0; i < n_objs_x; ++i) {
    for (unsigned j = 0; j < n_objs_y; ++j) {
      agent.add_proximal_object(start_x + (stop_x-start_x)*(FloatT)i/(n_objs_x-1),
                                start_y + (stop_y-start_y)*(FloatT)j/(n_objs_y-1));
    }
  }

  /*
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

  agent.set_row_separation(0.1);

  // // agent.set_place_test_params(0.1, 16);
  // agent.add_test_position(0.4, 0.4);
  // agent.add_test_position(0.4, -0.4);
  // agent.add_test_position(-0.4, 0.4);
  // agent.add_test_position(-0.4, -0.4);
  // agent.add_test_position(0, 0);

  int N_per_obj = 100;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 1.0;

  int N_per_state = 50;

  // Input neurons:
  AgentVISRateNeurons VIS(ctx, &agent, N_per_obj, sigma_VIS, lambda_VIS, "VIS");
  AgentAHVRateNeurons AHV(ctx, &agent, N_per_state, "AHV");
#ifdef ENABLE_PLACE
  AgentFVRateNeurons FV(ctx, &agent, N_per_state, "FV");
#endif

  int N_VIS = VIS.size;
  int N_AHV = AHV.size;
#ifdef ENABLE_PLACE
  int N_FV = FV.size;
#endif

  // HD neurons:
  int N_HD = 300; // N_VIS
  FloatT alpha_HD = 20.0;
  FloatT beta_HD = 0.6;
  FloatT tau_HD = 1e-2;
  RateNeurons HD(ctx, N_HD, "HD", alpha_HD, beta_HD, tau_HD);

  EigenVector VIS_INH_on = EigenVector::Ones(N_VIS);
  EigenVector VIS_INH_off = EigenVector::Zero(N_VIS);
  DummyRateNeurons VIS_INH(ctx, N_VIS, "VIS_INH");

  // AHVxHD neurons:
  int N_AHVxHD = N_HD * agent.num_AHV_states;
  FloatT alpha_AHVxHD = 20.0;
  FloatT beta_AHVxHD = 0.6;
  FloatT tau_AHVxHD = 1e-2;
  RateNeurons AHVxHD(ctx, N_AHVxHD, "AHVxHD",
                     alpha_AHVxHD, beta_AHVxHD, tau_AHVxHD);

#ifdef ENABLE_PLACE
  // PLACE neurons:
  int N_PLACE = N_HD; // N_VIS;
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
#endif

  // General parameters:
  FloatT axonal_delay = 1e-2; // seconds (TODO units)
  FloatT eps = 0.02;


  // VIS -> HD connectivity:
  FloatT VIS_HD_sparsity = 0.05;
  FloatT VIS_HD_scaling = 3200.0 / (N_VIS*VIS_HD_sparsity); // 1600
  FloatT VIS_HD_INH_scaling = -2.0 / (N_VIS*VIS_HD_sparsity); // -1.0
  FloatT eps_VIS_HD = 0.06;

  // AHVxHD -> HD connectivity:
  FloatT AHVxHD_HD_scaling = 6400.0 / (N_AHVxHD*1.0); // 6000

  // HD -> HD connectivity
  FloatT HD_inhibition = -600.0 / N_HD; // 300

  // HD -> AHVxHD connectivity:
  FloatT HD_AHVxHD_sparsity = 0.05;
  FloatT HD_AHVxHD_scaling = 350.0 / (N_HD*HD_AHVxHD_sparsity); // 500

  // AHV -> AHVxHD connectivity:
  FloatT AHV_AHVxHD_scaling = 320.0 / N_AHV; // 240

  // AHVxHD -> AHVxHD connectivity:
  FloatT AHVxHD_inhibition = -100.0 / N_AHVxHD; // -250

#ifdef ENABLE_PLACE
  // FV -> PLACExFVxHD connectivity:
  FloatT FV_PLACExFVxHD_scaling = 90.0 / N_FV;

  // HD -> PLACExFVxHD connectivity:
  FloatT HD_PLACExFVxHD_sparsity = 0.05;
  FloatT HD_PLACExFVxHD_scaling = 130.0 / (HD_PLACExFVxHD_sparsity*N_HD);

  // PLACE -> PLACExFVxHD connectivity:
  FloatT PLACE_PLACExFVxHD_sparsity = 0.05;
  FloatT PLACE_PLACExFVxHD_scaling = 280.0 / (N_PLACE*PLACE_PLACExFVxHD_sparsity);

  // PLACExFVxHD -> PLACExFVxHD connectivity:
  FloatT PLACExFVxHD_inhibition = -160.0 / N_PLACExFVxHD;

  // VIS -> PLACE connectivity:
  FloatT VIS_PLACE_sparsity = VIS_HD_sparsity;
  FloatT VIS_PLACE_scaling = VIS_HD_scaling; // 1100.0 / (N_VIS*0.05);
  FloatT VIS_PLACE_INH_scaling = VIS_HD_INH_scaling; // -2.3 / (N_VIS*0.05);
  FloatT eps_VIS_PLACE = eps_VIS_HD;

  // PLACExFVxHD -> PLACE connectivity:
  FloatT PLACExFVxHD_PLACE_scaling = 6800.0 / N_PLACExFVxHD;

  // PLACE -> PLACE connectivity:
  FloatT PLACE_inhibition = -600.0 / N_PLACE;
#endif

  // VIS -> HD connectivity
  RateSynapses VIS_HD(ctx, &VIS, &HD, VIS_HD_scaling, "VIS_HD");
//#ifdef TRAIN_VIS_HD
  EigenMatrix W_VIS_HD = Eigen::make_random_matrix(N_HD, N_VIS, 1.0, true,
                                                   1.0-VIS_HD_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_VIS_HD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_VIS_HD, N_HD, N_VIS);
  }
//#else
//  EigenMatrix W_VIS_HD = EigenMatrix::Identity(N_HD, N_VIS);
//#endif
  VIS_HD.weights(W_VIS_HD);
  VIS_HD.make_sparse();

  BCMPlasticity plast_VIS_HD(ctx, &VIS_HD);

  HD.connect_input(&VIS_HD, &plast_VIS_HD);

  RateSynapses VIS_HD_INH(ctx, &VIS_INH, &HD,
                          VIS_HD_INH_scaling, "VIS_HD_INH");
  VIS_HD_INH.weights(EigenMatrix::Ones(N_HD, N_VIS));

  BCMPlasticity plast_VIS_HD_INH(ctx, &VIS_HD_INH);

  HD.connect_input(&VIS_HD_INH, &plast_VIS_HD_INH);


  // AHVxHD -> HD connectivity:
  RateSynapses AHVxHD_HD(ctx, &AHVxHD, &HD, AHVxHD_HD_scaling, "AHVxHD_HD");
  AHVxHD_HD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_AHVxHD_HD = Eigen::make_random_matrix(N_HD, N_AHVxHD);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_AHVxHD_HD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_AHVxHD_HD, N_HD, N_AHVxHD);
  }
  AHVxHD_HD.weights(W_AHVxHD_HD);

  BCMPlasticity plast_AHVxHD_HD(ctx, &AHVxHD_HD);

  // RateSynapses AHVxHD_HD_INH(ctx, &AHVxHD, &HD, AHVxHD_inhibition, "AHVxHD_HD_INH");
  // AHVxHD_HD_INH.weights(EigenMatrix::Ones(N_HD, N_AHVxHD));
  // BCMPlasticity plast_AHVxHD_HD_INH(ctx, &AHVxHD_HD_INH);

  HD.connect_input(&AHVxHD_HD, &plast_AHVxHD_HD);


  // HD -> HD connectivity
  RateSynapses HD_HD_INH(ctx, &HD, &HD, HD_inhibition, "HD_HD_INH");
  HD_HD_INH.weights(EigenMatrix::Ones(N_HD, N_HD));

  BCMPlasticity plast_HD_HD_INH(ctx, &HD_HD_INH);

  HD.connect_input(&HD_HD_INH, &plast_HD_HD_INH);


  // HD -> AHVxHD connectivity:
  RateSynapses HD_AHVxHD(ctx, &HD, &AHVxHD, HD_AHVxHD_scaling, "HD_AHVxHD");
  HD_AHVxHD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_HD_AHVxHD = Eigen::make_random_matrix(N_AHVxHD, N_HD, 1.0, true,
                                                      1.0-HD_AHVxHD_sparsity, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_HD_AHVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_HD_AHVxHD, N_AHVxHD, N_HD);
  }
  HD_AHVxHD.weights(W_HD_AHVxHD);
  HD_AHVxHD.make_sparse();

  BCMPlasticity plast_HD_AHVxHD(ctx, &HD_AHVxHD);

  AHVxHD.connect_input(&HD_AHVxHD, &plast_HD_AHVxHD);

  // RateSynapses HD_AHVxHD_INH(ctx, &HD, &AHVxHD,
  //                            HD_inhibition, "HD_AHVxHD_INH");
  // HD_AHVxHD_INH.weights(EigenMatrix::Ones(N_AHVxHD, N_AHVxHD));
  // BCMPlasticity plast_HD_AHVxHD_INH(ctx, &HD_AHVxHD_INH);


  // AHV -> AHVxHD connectivity:
  RateSynapses AHV_AHVxHD(ctx, &AHV, &AHVxHD, AHV_AHVxHD_scaling, "AHV_AHVxHD");
  EigenMatrix W_AHV_AHVxHD = Eigen::make_random_matrix(N_AHVxHD, N_AHV);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_AHV_AHVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_AHV_AHVxHD, N_AHVxHD, N_AHV);
  }
  AHV_AHVxHD.weights(W_AHV_AHVxHD);

  BCMPlasticity plast_AHV_AHVxHD(ctx, &AHV_AHVxHD);

  AHVxHD.connect_input(&AHV_AHVxHD, &plast_AHV_AHVxHD);


  // AHVxHD -> AHVxHD connectivity:
  RateSynapses AHVxHD_AHVxHD_INH(ctx, &AHVxHD, &AHVxHD,
                                 AHVxHD_inhibition, "AHVxHD_AHVxHD_INH");
  AHVxHD_AHVxHD_INH.weights(EigenMatrix::Ones(N_AHVxHD, N_AHVxHD));

  BCMPlasticity plast_AHVxHD_AHVxHD_INH(ctx, &AHVxHD_AHVxHD_INH);

  AHVxHD.connect_input(&AHVxHD_AHVxHD_INH, &plast_AHVxHD_AHVxHD_INH);


#ifdef ENABLE_PLACE
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
#endif


  // Set simulation schedule:
  VIS.t_stop_after = train_time + test_on_time;
  VIS_INH.add_schedule(VIS.t_stop_after, VIS_INH_on);
  VIS_INH.add_schedule(infinity<FloatT>(), VIS_INH_off);

  // No inhibitory plasticity:
  plast_HD_HD_INH.add_schedule(infinity<FloatT>(), 0);
  plast_AHVxHD_AHVxHD_INH.add_schedule(infinity<FloatT>(), 0);
#ifdef ENABLE_PLACE
  plast_PLACE_PLACE_INH.add_schedule(infinity<FloatT>(), 0);
  plast_PLACExFVxHD_PLACExFVxHD_INH.add_schedule(infinity<FloatT>(), 0);
#endif
  // plast_AHVxHD_HD_INH.add_schedule(infinity<FloatT>(), 0);
  // plast_HD_AHVxHD_INH.add_schedule(infinity<FloatT>(), 0);

  // Rest have plasticity on only during training, with params above:
  plast_VIS_HD.add_schedule(train_time, eps_VIS_HD);
  plast_AHVxHD_HD.add_schedule(train_time, eps*0.8);
  plast_HD_AHVxHD.add_schedule(train_time, eps*1.2);
  plast_AHV_AHVxHD.add_schedule(train_time, eps);

#ifdef ENABLE_PLACE
  plast_VIS_PLACE.add_schedule(train_time, eps_VIS_PLACE);
  plast_PLACExFVxHD_PLACE.add_schedule(train_time, eps*0.8);
  plast_PLACE_PLACExFVxHD.add_schedule(train_time, eps*1.2);
  plast_FV_PLACExFVxHD.add_schedule(train_time, eps);
  plast_HD_PLACExFVxHD.add_schedule(train_time, eps);
#endif

  plast_VIS_HD.add_schedule(infinity<FloatT>(), 0);
  plast_AHVxHD_HD.add_schedule(infinity<FloatT>(), 0);
  plast_HD_AHVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_AHV_AHVxHD.add_schedule(infinity<FloatT>(), 0);

#ifdef ENABLE_PLACE
  plast_VIS_PLACE.add_schedule(infinity<FloatT>(), 0);
  plast_PLACExFVxHD_PLACE.add_schedule(infinity<FloatT>(), 0);
  plast_PLACE_PLACExFVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_FV_PLACExFVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_HD_PLACExFVxHD.add_schedule(infinity<FloatT>(), 0);
#endif


  // Have to construct electrodes after neurons:
  RateElectrodes VIS_elecs("HD_VIS_out", &VIS);
  RateElectrodes AHV_elecs("HD_VIS_out", &AHV);
#ifdef ENABLE_PLACE
  RateElectrodes FV_elecs("HD_VIS_out", &FV);
#endif

  RateElectrodes HD_elecs("HD_VIS_out", &HD);
  RateElectrodes AHVxHD_elecs("HD_VIS_out", &AHVxHD);

#ifdef ENABLE_PLACE
  RateElectrodes PLACE_elecs("HD_VIS_out", &PLACE);
  RateElectrodes PLACExFVxHD_elecs("HD_VIS_out", &PLACExFVxHD);
#endif


  // Add Agent, Neurons and Electrodes to Model
  model.add(&agent);

  model.add(&VIS);
  model.add(&VIS_INH);

  model.add(&HD);
  model.add(&AHV);
  model.add(&AHVxHD);

#ifdef ENABLE_PLACE
  model.add(&PLACE);
  model.add(&FV);
  model.add(&PLACExFVxHD);
#endif

  model.add(&VIS_elecs);

  model.add(&HD_elecs);
  model.add(&AHV_elecs);
  model.add(&AHVxHD_elecs);

#ifdef ENABLE_PLACE
  model.add(&FV_elecs);
  model.add(&PLACE_elecs);
  model.add(&PLACExFVxHD_elecs);
#endif


  // Set simulation time parameters:
  model.set_simulation_time(train_time + test_on_time + test_off_time, timestep);
  model.set_buffer_intervals((float)buffer_timestep); // TODO: Use proper units
  model.set_weights_buffer_interval(ceil(10.0/timestep));
  model.set_buffer_start(start_recording_time);

  agent.save_map("HD_VIS_out");
  agent.record_history("HD_VIS_out", round(buffer_timestep/timestep),
                       round(start_recording_time/timestep));


  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
