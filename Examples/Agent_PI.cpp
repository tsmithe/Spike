#include "Spike/Models/RateModel.hpp"
#include <fenv.h>

// TODO: Add signal handlers

int main(int argc, char *argv[]) {
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  bool read_weights = false;
  std::string weights_path;

  if (argc == 2) {
    read_weights = true;
    weights_path = argv[1];
  }

  FloatT timestep = 5e-4; // seconds (TODO units)
  FloatT train_time = 12000;
  if (read_weights) train_time = 0;
  FloatT test_on_time = 10;
  FloatT test_off_time = 20;
  FloatT start_recording_time = 5000;
  if (read_weights) start_recording_time = 0;
  
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  //ctx->verbose = true;
  ctx->backend = "Eigen";

  // Create Agent
  Agent agent;

  FloatT radius = 5;
  FloatT bound_x = sqrt(2) * radius;
  FloatT bound_y = bound_x;

  agent.set_boundary(bound_x, bound_y);

  agent.add_distal_object(0);
  agent.add_distal_object(0.5 * M_PI);
  agent.add_distal_object(M_PI);
  agent.add_distal_object(1.5 * M_PI);

  agent.add_proximal_object(bound_x, bound_y);
  agent.add_proximal_object(bound_x, -bound_y);
  agent.add_proximal_object(-bound_x, bound_y);
  agent.add_proximal_object(-bound_x, -bound_y);

  agent.add_proximal_object(0.5*bound_x, 0.5*bound_y);
  agent.add_proximal_object(0.5*bound_x, -0.5*bound_y);
  agent.add_proximal_object(-0.5*bound_x, 0.5*bound_y);
  agent.add_proximal_object(-0.5*bound_x, -0.5*bound_y);

  FloatT fwd_move_dist = (2.0/3.0) * radius;
  FloatT rot_angle = M_PI / 2;

  FloatT move_time = 1; // second per forward or angular move

  agent.add_AHV(rot_angle, move_time);
  agent.add_AHV(-rot_angle, move_time);

  agent.add_FV(fwd_move_dist, move_time);
 
  int N_per_obj = 100;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 1.0;

  int N_per_state = 50;

  // Input neurons:
  AgentVISRateNeurons VIS(ctx, &agent, N_per_obj, sigma_VIS, lambda_VIS, "VIS");
  AgentAHVRateNeurons AHV(ctx, &agent, N_per_state, "AHV");
  AgentFVRateNeurons FV(ctx, &agent, N_per_state, "FV");

  int N_VIS = VIS.size;
  int N_AHV = AHV.size;
  int N_FV = FV.size;

  // HD neurons:
  int N_HD = N_VIS;
  FloatT alpha_HD = 20.0;
  FloatT beta_HD = 1.0;
  FloatT tau_HD = 1e-2;
  RateNeurons HD(ctx, N_HD, "HD", alpha_HD, beta_HD, tau_HD);

  EigenVector VIS_INH_on = EigenVector::Ones(N_VIS);
  EigenVector VIS_INH_off = EigenVector::Zero(N_VIS);
  DummyRateNeurons VIS_INH(ctx, N_VIS, "VIS_INH");

  // AHVxHD neurons:
  int N_AHVxHD = N_HD * agent.num_AHV_states;
  FloatT alpha_AHVxHD = 20.0;
  FloatT beta_AHVxHD = 1.2;
  FloatT tau_AHVxHD = 1e-2;
  RateNeurons AHVxHD(ctx, N_AHVxHD, "AHVxHD",
                     alpha_AHVxHD, beta_AHVxHD, tau_AHVxHD);

  // PLACE neurons:
  int N_PLACE = N_VIS;
  FloatT alpha_PLACE = 20.0;
  FloatT beta_PLACE = 1.0;
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
  FloatT beta_PLACExFVxHD = 1.2;
  FloatT tau_PLACExFVxHD = 1e-2;
  RateNeurons PLACExFVxHD(ctx, N_PLACExFVxHD, "PLACExFVxHD",
                          alpha_PLACExFVxHD, beta_PLACExFVxHD,
                          tau_PLACExFVxHD);


  // General parameters:
  FloatT axonal_delay = 1e-2; // seconds (TODO units)
  FloatT eps = 0.01;


  // VIS -> HD connectivity:
  FloatT VIS_HD_scaling = 1100.0 / (N_VIS*0.05); // 1600
  FloatT VIS_HD_INH_scaling = -2.3 / (N_VIS*0.05); // -1.0
  FloatT eps_VIS_HD = 0.05;

  RateSynapses VIS_HD(ctx, &VIS, &HD, VIS_HD_scaling, "VIS_HD");
//#ifdef TRAIN_VIS_HD
  EigenMatrix W_VIS_HD = Eigen::make_random_matrix(N_HD, N_VIS, 1.0, true, 0.95, 0, false);
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
  FloatT AHVxHD_HD_scaling = 4500.0 / (N_AHVxHD*1.0); // 6000

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
  FloatT HD_inhibition = -32.0 / N_HD; // 300

  RateSynapses HD_HD_INH(ctx, &HD, &HD, HD_inhibition, "HD_HD_INH");
  HD_HD_INH.weights(EigenMatrix::Ones(N_HD, N_HD));

  BCMPlasticity plast_HD_HD_INH(ctx, &HD_HD_INH);

  HD.connect_input(&HD_HD_INH, &plast_HD_HD_INH);


  // HD -> AHVxHD connectivity:
  FloatT HD_AHVxHD_scaling = 360.0 / (N_HD*0.05); // 500

  RateSynapses HD_AHVxHD(ctx, &HD, &AHVxHD, HD_AHVxHD_scaling, "HD_AHVxHD");
  HD_AHVxHD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_HD_AHVxHD = Eigen::make_random_matrix(N_AHVxHD, N_HD, 1.0, true, 0.95, 0, false);
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
  FloatT AHV_AHVxHD_scaling = 240.0 / N_AHV; // 240

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
  FloatT AHVxHD_inhibition = -120.0 / N_AHVxHD; // -250
  RateSynapses AHVxHD_AHVxHD_INH(ctx, &AHVxHD, &AHVxHD,
                                 AHVxHD_inhibition, "AHVxHD_AHVxHD_INH");
  AHVxHD_AHVxHD_INH.weights(EigenMatrix::Ones(N_AHVxHD, N_AHVxHD));

  BCMPlasticity plast_AHVxHD_AHVxHD_INH(ctx, &AHVxHD_AHVxHD_INH);

  AHVxHD.connect_input(&AHVxHD_AHVxHD_INH, &plast_AHVxHD_AHVxHD_INH);


  // FV -> PLACExFVxHD connectivity:
  FloatT FV_PLACExFVxHD_scaling = 240.0 / N_FV;

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
  FloatT HD_PLACExFVxHD_scaling = 240.0 / N_HD;

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
  FloatT PLACE_PLACExFVxHD_scaling = 360.0 / (N_PLACE*0.05);

  RateSynapses PLACE_PLACExFVxHD(ctx, &PLACE, &PLACExFVxHD,
                                 PLACE_PLACExFVxHD_scaling,
                                 "PLACE_PLACExFVxHD");
  PLACE_PLACExFVxHD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_PLACE_PLACExFVxHD
    = Eigen::make_random_matrix(N_PLACExFVxHD, N_PLACE,
                                1.0, true, 0.95, 0, false);
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
  FloatT PLACExFVxHD_inhibition = -120.0 / N_PLACExFVxHD;
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
  FloatT VIS_PLACE_scaling = VIS_HD_scaling; // 1100.0 / (N_VIS*0.05);
  FloatT VIS_PLACE_INH_scaling = VIS_HD_INH_scaling; // -2.3 / (N_VIS*0.05);
  FloatT eps_VIS_PLACE = eps_VIS_HD;

  RateSynapses VIS_PLACE(ctx, &VIS, &PLACE, VIS_PLACE_scaling, "VIS_PLACE");
  EigenMatrix W_VIS_PLACE
    = Eigen::make_random_matrix(N_PLACE, N_VIS, 1.0, true, 0.95, 0, false);
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
  FloatT PLACExFVxHD_PLACE_scaling = 4500.0 / N_PLACExFVxHD;

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
  FloatT PLACE_inhibition = -32.0 / N_PLACE;

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
  plast_HD_HD_INH.add_schedule(infinity<FloatT>(), 0);
  plast_AHVxHD_AHVxHD_INH.add_schedule(infinity<FloatT>(), 0);
  plast_PLACE_PLACE_INH.add_schedule(infinity<FloatT>(), 0);
  plast_PLACExFVxHD_PLACExFVxHD_INH.add_schedule(infinity<FloatT>(), 0);
  // plast_AHVxHD_HD_INH.add_schedule(infinity<FloatT>(), 0);
  // plast_HD_AHVxHD_INH.add_schedule(infinity<FloatT>(), 0);

  // Rest have plasticity on only during training, with params above:
  plast_VIS_HD.add_schedule(train_time, eps_VIS_HD);
  plast_AHVxHD_HD.add_schedule(train_time, eps*0.8);
  plast_HD_AHVxHD.add_schedule(train_time, eps*1.2);
  plast_AHV_AHVxHD.add_schedule(train_time, eps);

  plast_VIS_PLACE.add_schedule(train_time, eps_VIS_PLACE);
  plast_PLACExFVxHD_PLACE.add_schedule(train_time, eps*0.8);
  plast_PLACE_PLACExFVxHD.add_schedule(train_time, eps*1.2);
  plast_FV_PLACExFVxHD.add_schedule(train_time, eps);
  plast_HD_PLACExFVxHD.add_schedule(train_time, eps);

  plast_VIS_HD.add_schedule(infinity<FloatT>(), 0);
  plast_AHVxHD_HD.add_schedule(infinity<FloatT>(), 0);
  plast_HD_AHVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_AHV_AHVxHD.add_schedule(infinity<FloatT>(), 0);

  plast_VIS_PLACE.add_schedule(infinity<FloatT>(), 0);
  plast_PLACExFVxHD_PLACE.add_schedule(infinity<FloatT>(), 0);
  plast_PLACE_PLACExFVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_FV_PLACExFVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_HD_PLACExFVxHD.add_schedule(infinity<FloatT>(), 0);


  // Have to construct electrodes after neurons:
  RateElectrodes VIS_elecs("HD_VIS_out", &VIS);
  RateElectrodes AHV_elecs("HD_VIS_out", &AHV);
  RateElectrodes FV_elecs("HD_VIS_out", &FV);

  RateElectrodes HD_elecs("HD_VIS_out", &HD);
  RateElectrodes AHVxHD_elecs("HD_VIS_out", &AHVxHD);

  RateElectrodes PLACE_elecs("HD_VIS_out", &PLACE);
  RateElectrodes PLACExFVxHD_elecs("HD_VIS_out", &PLACExFVxHD);


  // Add Agent, Neurons and Electrodes to Model
  model.add(&agent);

  model.add(&VIS);
  model.add(&VIS_INH);

  model.add(&AHV);
  model.add(&FV);

  model.add(&HD);
  model.add(&AHVxHD);

  model.add(&PLACE);
  model.add(&PLACExFVxHD);

  model.add(&VIS_elecs);
  model.add(&AHV_elecs);
  model.add(&FV_elecs);

  model.add(&HD_elecs);
  model.add(&AHVxHD_elecs);

  model.add(&PLACE_elecs);
  model.add(&PLACExFVxHD_elecs);


  // Set simulation time parameters:
  model.set_simulation_time(train_time + test_on_time + test_off_time, timestep);
  model.set_buffer_intervals((float)1e-2); // TODO: Use proper units
  model.set_weights_buffer_interval(ceil(2.0/timestep));
  model.set_buffer_start(start_recording_time);

  agent.record_history("HD_VIS_out", 1e-2, start_recording_time);


  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
