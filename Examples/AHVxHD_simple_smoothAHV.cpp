#include "Spike/Models/RateModel.hpp"
#include "Spike/Models/RateAgent.hpp"
#include <fenv.h>
#include <omp.h>

// TODO: Add signal handlers

#define OUTPUT_PATH "AHVxHD_simple_smoothAHV"

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
  FloatT train_time = 16200;
  if (read_weights) train_time = 0;
  FloatT test_on_time = 0; //300;
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
  Agent<RandomWalkPolicy, HDTestPolicy> agent;
  // agent.seed(123);

  FloatT radius = 2;
  FloatT bound_x = 2;
  FloatT bound_y = bound_x;

  agent.set_boundary(bound_x, bound_y);

  agent.add_distal_object(0);

  FloatT start_x = -0.75;
  FloatT stop_x = 0.75;
  FloatT start_y = -0.75;
  FloatT stop_y = 0.75;
  unsigned n_objs_x = 4;
  unsigned n_objs_y = 2;

  FloatT fwd_move_dist = 0.4 * radius;
  FloatT rot_angle = M_PI / 2;

  FloatT fwd_move_time = 1.0; ///6.0; // seconds per forward move
  FloatT angle_move_time = 1.0; ///6.0; // seconds per angular move

  agent.p_fwd = 0.5 * 1.0/3.0;

  agent.add_AHV(rot_angle / angle_move_time, angle_move_time);
  agent.add_AHV(rot_angle / angle_move_time / 2, angle_move_time);
  agent.add_AHV(rot_angle / angle_move_time / 3, angle_move_time);
  agent.add_AHV(-rot_angle / angle_move_time, angle_move_time);
  agent.add_AHV(-rot_angle / angle_move_time / 2, angle_move_time);
  agent.add_AHV(-rot_angle / angle_move_time / 3, angle_move_time);

  agent.add_FV(fwd_move_dist / fwd_move_time, fwd_move_time);

  // agent.add_test_time(100);
  // agent.add_test_time(500);
  agent.add_test_time(2000);
  agent.add_test_time(4000);
  agent.add_test_time(6000);
  agent.add_test_time(8000);
  agent.add_test_time(10000);
  agent.add_test_time(12000);
  agent.add_test_time(14000);
  agent.add_test_time(16000);
  // agent.set_place_test_params(0.1, 16);
  agent.add_test_position(0, 0);
  /*
  agent.add_test_position(0.4, 0.4);
  agent.add_test_position(0.4, -0.4);
  agent.add_test_position(-0.4, 0.4);
  agent.add_test_position(-0.4, -0.4);
  */

  int N_per_obj = 300;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 1.0;

  int N_per_state = 100;

  // Input neurons:
  AgentVISRateNeurons HD(ctx, &agent, N_per_obj, sigma_VIS, lambda_VIS, "HD");

  agent.set_smooth_AHV(rot_angle / angle_move_time * 2); // NB Need to do this first!
  AgentAHVRateNeurons AHV(ctx, &agent, N_per_state, "AHV");
  // AHV.set_smooth_params(0.1, 1.0, 1.0 / (1.5*M_PI));
  const EigenVector smooth_random = (0.5 * (EigenVector::Random(N_per_state).array() + 1)).matrix();
  auto smooth_sym_base = (0.1 + 0.05 * EigenVector::Random(N_per_state).array()).matrix();
  auto smooth_sym_max = (0.25 + 0.15 * EigenVector::Random(N_per_state).array()).matrix();
  auto smooth_sym_slope = (1.5 / M_PI + 0.4 * EigenVector::Random(N_per_state).array()).matrix();
  auto smooth_asym_neg_base = (0.1 + 0.1 * EigenVector::Random(N_per_state).array()).matrix();
  auto smooth_asym_neg_max = (0.85 + 0.15 * EigenVector::Random(N_per_state).array()).matrix();
  auto smooth_asym_neg_slope = (1.5 / M_PI + 0.4 * EigenVector::Random(N_per_state).array()).matrix();
  auto smooth_asym_pos_base = (0.1 + 0.1 * EigenVector::Random(N_per_state).array()).matrix();
  auto smooth_asym_pos_max = (0.85 + 0.15 * EigenVector::Random(N_per_state).array()).matrix();
  auto smooth_asym_pos_slope = (1.5 / M_PI + 0.4 * EigenVector::Random(N_per_state).array()).matrix();
  AHV.set_smooth_params(smooth_sym_base, smooth_sym_max, smooth_sym_slope,
                        smooth_asym_neg_base, smooth_asym_neg_max, smooth_asym_neg_slope,
                        smooth_asym_pos_base, smooth_asym_pos_max, smooth_asym_pos_slope);

  int N_HD = HD.size;
  int N_AHV = AHV.size;

  // AHVxHD neurons:
  int N_AHVxHD = 600;
  FloatT alpha_AHVxHD = 16.0;
  FloatT beta_AHVxHD = 0.5;
  FloatT tau_AHVxHD = 1e-2;
  RateNeurons AHVxHD(ctx, N_AHVxHD, "AHVxHD",
                     alpha_AHVxHD, beta_AHVxHD, tau_AHVxHD);

  // General parameters:
  FloatT axonal_delay = 0; //1e-2; // seconds (TODO units)
  FloatT eps = 0.03;


  // HD -> AHVxHD connectivity:
  FloatT HD_AHVxHD_sparsity = 0.05;
  FloatT HD_AHVxHD_scaling = 200.0 / (N_HD*HD_AHVxHD_sparsity); // 500

  // AHV -> AHVxHD connectivity:
  FloatT AHV_AHVxHD_sparsity = 0.4;
  FloatT AHV_AHVxHD_scaling = 320.0 / (N_AHV*AHV_AHVxHD_sparsity); // 240

  // AHVxHD -> AHVxHD connectivity:
  FloatT AHVxHD_inhibition = -160.0 / N_AHVxHD; // -250

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
  EigenMatrix W_AHV_AHVxHD = Eigen::make_random_matrix(N_AHVxHD, N_AHV, 1.0, true,
                                                      1.0-AHV_AHVxHD_sparsity, 0, false);
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


  // No inhibitory plasticity:
  plast_AHVxHD_AHVxHD_INH.add_schedule(infinity<FloatT>(), 0);
  // plast_AHVxHD_HD_INH.add_schedule(infinity<FloatT>(), 0);
  // plast_HD_AHVxHD_INH.add_schedule(infinity<FloatT>(), 0);

  // Rest have plasticity on only during training, with params above:
  plast_HD_AHVxHD.add_schedule(train_time, eps*1.2);
  plast_AHV_AHVxHD.add_schedule(train_time, eps);

  plast_HD_AHVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_AHV_AHVxHD.add_schedule(infinity<FloatT>(), 0);

  // Have to construct electrodes after neurons:
  RateElectrodes AHV_elecs(OUTPUT_PATH, &AHV);

  RateElectrodes HD_elecs(OUTPUT_PATH, &HD);
  RateElectrodes AHVxHD_elecs(OUTPUT_PATH, &AHVxHD);


  // Add Agent, Neurons and Electrodes to Model
  model.add(&agent);

  model.add(&HD);
  model.add(&AHV);
  model.add(&AHVxHD);

  model.add(&HD_elecs);
  model.add(&AHV_elecs);
  model.add(&AHVxHD_elecs);


  // Set simulation time parameters:
  model.set_simulation_time(train_time + test_on_time + test_off_time, timestep);
  model.set_buffer_intervals((float)buffer_timestep); // TODO: Use proper units
  model.set_weights_buffer_interval(ceil(10.0/timestep));
  model.set_buffer_start(start_recording_time);

  agent.save_map(OUTPUT_PATH);
  agent.record_history(OUTPUT_PATH, round(buffer_timestep/timestep),
                       round(start_recording_time/timestep));


  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}