#include "Spike/Models/RateModel.hpp"
#include "Spike/Models/RateAgent.hpp"
#include <fenv.h>
#include <omp.h>

#define OUTPUT_PATH "PLACE_detour"

// TODO: Add signal handlers

int main(int argc, char *argv[]) {
  Eigen::initParallel();
  omp_set_num_threads(20);
  Eigen::setNbThreads(20);
  std::cout << Eigen::nbThreads() << std::endl;
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  bool read_weights = false;
  std::string weights_path;

  if (argc == 2) {
    read_weights = true;
    weights_path = argv[1];
  }

  FloatT timestep = pow(2, -9); // seconds (TODO units)
  FloatT buffer_timestep = pow(2, -4);
  FloatT train_time = 40000;
  if (read_weights) train_time = 0;
  FloatT test_on_time = 0; // 800
  FloatT test_off_time = 0; // 200
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

  for (unsigned i = 0; i < 4; ++i) {
    agent.add_velocity(2.5, (1.0/2.0)*i*M_PI);
  }

  agent.load_map(
"xx*xx          \n"
"x   x          \n"
"x   x          \n"
"*   xxxxx*xxxxx\n"
"x             *\n"
"x             x\n"
"x             x\n"
"x   xxx*xxx   x\n"
"*   x     x   *\n"
"x   x     x   x\n"
"x   *     *   x\n"
"x   x     x   x\n"
"*   x     x   *\n"
"x   xxx*xxx   x\n"
"x             x\n"
"x             x\n"
"x             *\n"
"*   xxxxx*xxxxx\n"
"x   x          \n"
"x   x          \n"
"x s x          \n"
"xx*xx         #\n"
);

  agent.load_tests(
"xx*xx          \n"
"xpppx          \n"
"xpppx          \n"
"*pppxxxxx*xxxxx\n"
"xppppppppppppp*\n"
"xpppppppppppppx\n"
"xpppppppppppppx\n"
"xpppxxx*xxxpppx\n"
"*pppx     xppp*\n"
"xpppx     xpppx\n"
"xppp*     *pppx\n"
"xpppx     xpppx\n"
"*pppx     xppp*\n"
"xpppxxx*xxxpppx\n"
"xpppppppppppppx\n"
"xpppppppppppppx\n"
"xppppppppppppp*\n"
"*pppxxxxx*xxxxx\n"
"xpppx          \n"
"xpppx          \n"
"xpppx          \n"
"xx*xx         #\n"
);

  const FloatT reward_intro_time = 1050;
  bool rewarded = false;
  model.add_hook([&]() {
      if (rewarded || agent.current_time() < reward_intro_time) return;
      rewarded = true;
      agent.load_rewards(
"xx*xx          \n"
"x+++x          \n"
"x+++x          \n"
"*+++xxxxx*xxxxx\n"
"x             *\n"
"x             x\n"
"x             x\n"
"x   xxx*xxx   x\n"
"*   x     x   *\n"
"x   x     x   x\n"
"x   *     *   x\n"
"x   x     x   x\n"
"*   x     x   *\n"
"x   xxx*xxx   x\n"
"x             x\n"
"x             x\n"
"x             *\n"
"*   xxxxx*xxxxx\n"
"x   x          \n"
"x   x          \n"
"x s x          \n"
"xx*xx         #\n",
{{'+', 20.0}});
      printf("\tIntroduced reward\n");
      agent.set_restart_bound(0.05, 3, -1);
    });

  const FloatT barrier_intro_time = 2215;
  bool new_map = false;
  model.add_hook([&]() {
      if (new_map || agent.current_time() < barrier_intro_time) return;
      new_map = true;
      agent.load_map(
"xx*xx          \n"
"x   x          \n"
"x   x          \n"
"*   xxxxx*xxxxx\n"
"x             *\n"
"x             x\n"
"x             x\n"
"x   xxx*xxx   x\n"
"*   x     x   *\n"
"x   x     x   x\n"
"x...*     *   x\n"
"x   x     x   x\n"
"*   x     x   *\n"
"x   xxx*xxx   x\n"
"x             x\n"
"x             x\n"
"x             *\n"
"*   xxxxx*xxxxx\n"
"x   x          \n"
"x   x          \n"
"x s x          \n"
"xx*xx         #\n"
);
      agent.load_rewards(
"xx*xx          \n"
"x+++x          \n"
"x+++x          \n"
"*+++xxxxx*xxxxx\n"
"x             *\n"
"x             x\n"
"x             x\n"
"x   xxx*xxx   x\n"
"*   x     x   *\n"
"x   x     x   x\n"
"x...*     *   x\n"
"x   x     x   x\n"
"*   x     x   *\n"
"x   xxx*xxx   x\n"
"x             x\n"
"x             x\n"
"x             *\n"
"*   xxxxx*xxxxx\n"
"x   x          \n"
"x   x          \n"
"x s x          \n"
"xx*xx         #\n",
{{'+', 50.0}});
      printf("\tIntroduced barrier\n");
      agent.save_map(OUTPUT_PATH, ".new");
    });

  agent.set_rl_params(10, 0.5, -100.0, 0.66, 0.85);


  FloatT fwd_move_dist = 0.4;
  FloatT rot_angle = M_PI / 8;

  FloatT fwd_move_time = 0.1; ///6.0; // seconds per forward move
  FloatT angle_move_time = 0.1; ///6.0; // seconds per angular move

  agent.add_AHV(rot_angle / angle_move_time, angle_move_time);
  agent.add_AHV(-rot_angle / angle_move_time, angle_move_time);

  agent.add_FV(fwd_move_dist / fwd_move_time, fwd_move_time);

  agent.episode_length(150);

  FloatT test_duration = 580; // turns off plasticity during testing for this duration
  FloatT test_interval = 750; // time between tests (s)
  std::vector<FloatT> test_times{450};
  for (unsigned i = 0; i < 10; ++i) {
    test_times.push_back(test_times[i] + test_duration + test_interval);
  }
  std::cout << "Test times: ";
  for (auto const& t : test_times) {
    std::cout << t << " ";
    agent.add_test_time(t);
  }
  std::cout << std::endl;
 
  int N_per_obj = 60;
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

  printf("VIS: %d; FV: %d\n", N_VIS, N_FV);

  /*
  EigenVector VIS_INH_on = EigenVector::Ones(N_VIS);
  EigenVector VIS_INH_off = EigenVector::Zero(N_VIS);
  DummyRateNeurons VIS_INH(ctx, N_VIS, "VIS_INH");
  */

  // PLACE neurons:
  int N_PLACE = 400;
  FloatT alpha_PLACE = 20.0;
  FloatT beta_PLACE = 0.8;
  FloatT tau_PLACE = 1e-2;
  RateNeurons PLACE(ctx, N_PLACE, "PLACE", alpha_PLACE, beta_PLACE, tau_PLACE);

  // GRID neurons:
  int N_GRID = 20;
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

  // PLACExFVxHD neurons:
  int N_PLACExFVxHD = 200 * 2 * agent.num_FV_states;
  FloatT alpha_PLACExFVxHD = 20.0;
  FloatT beta_PLACExFVxHD = 0.3;
  FloatT tau_PLACExFVxHD = 1e-2;
  RateNeurons PLACExFVxHD(ctx, N_PLACExFVxHD, "PLACExFVxHD",
                         alpha_PLACExFVxHD, beta_PLACExFVxHD,
                         tau_PLACExFVxHD);


  // General parameters:
  FloatT axonal_delay = pow(2, -5); // seconds (TODO units)
  FloatT eps = 0.03;


  // FV -> FVxHD connectivity:
  FloatT FV_FVxHD_scaling = 270.0 / N_FV;

  // HD -> FVxHD connectivity:
  FloatT HD_FVxHD_sparsity = 0.05;
  FloatT HD_FVxHD_scaling = 480.0 / (HD_FVxHD_sparsity*N_HD);

  // FVxHD -> FVxHD connectivity:
  FloatT FVxHD_inhibition = -360.0 / N_FVxHD;

  // FVxHD -> PLACExFVxHD connectivity:
  FloatT FVxHD_PLACExFVxHD_sparsity = 0.05;
  FloatT FVxHD_PLACExFVxHD_scaling = 800.0 / (FVxHD_PLACExFVxHD_sparsity*N_FVxHD);

  // PLACE -> PLACExFVxHD connectivity:
  FloatT PLACE_PLACExFVxHD_sparsity = 0.05;
  FloatT PLACE_PLACExFVxHD_scaling = 1000.0 / (N_PLACE*PLACE_PLACExFVxHD_sparsity);

  // PLACExFVxHD -> PLACExFVxHD connectivity:
  FloatT PLACExFVxHD_inhibition = -501.0 / N_PLACExFVxHD;

  // VIS -> PLACE connectivity:
  FloatT VIS_PLACE_sparsity = 0.05;
  FloatT VIS_PLACE_scaling = 2050.0 / (N_VIS*VIS_PLACE_sparsity); // 6x6: 2000
  FloatT VIS_PLACE_INH_scaling = -12.0 / (N_VIS*VIS_PLACE_sparsity);

  // PLACExFVxHD -> PLACE connectivity:
  FloatT PLACExFVxHD_PLACE_scaling = 9000.0 / N_PLACExFVxHD; // ???14000

  // PLACE -> PLACE connectivity:
  FloatT PLACE_inhibition = -1800.0 / N_PLACE;

  // PLACE -> GRID connectivity:
  FloatT PLACE_GRID_sparsity = 1;
  FloatT PLACE_GRID_scaling = 12000.0 / (N_PLACE*PLACE_GRID_sparsity);

  // VIS -> GRID connectivity:
  FloatT VIS_GRID_sparsity = 0.3;
  FloatT VIS_GRID_scaling = 400.0 / (N_VIS*VIS_PLACE_sparsity); // 6x6: 2000

  // GRID -> GRID connectivity:
  // FloatT GRID_GRID_sparsity = 1;
  // FloatT GRID_GRID_scaling = 100.0 / (N_GRID*GRID_GRID_sparsity);
  FloatT GRID_inhibition = -1000.0 / N_GRID;


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

  PLACExFVxHD.connect_input(&FVxHD_PLACExFVxHD, &plast_FVxHD_PLACExFVxHD);


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

  RateSynapses VIS_PLACE_INH(ctx, &VIS, &PLACE,
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


  // PLACE -> GRID connectivity:
  RateSynapses PLACE_GRID(ctx, &PLACE, &GRID,
                          PLACE_GRID_scaling, "PLACE_GRID");
  PLACE_GRID.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_PLACE_GRID
    = Eigen::make_random_matrix(N_GRID, N_PLACE);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_PLACE_GRID.bin";
    Eigen::read_binary(tmp_path.c_str(), W_PLACE_GRID,
                       N_GRID, N_PLACE);
  }
  PLACE_GRID.weights(W_PLACE_GRID);

  GHAPlasticity plast_PLACE_GRID(ctx, &PLACE_GRID);

  GRID.connect_input(&PLACE_GRID, &plast_PLACE_GRID);


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


  // GRID -> GRID connectivity:
  /*
  RateSynapses GRID_GRID(ctx, &GRID, &GRID,
                         GRID_GRID_scaling, "GRID_GRID");
  GRID_GRID.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_GRID_GRID
    = Eigen::make_random_matrix(N_GRID, N_GRID);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_GRID_GRID.bin";
    Eigen::read_binary(tmp_path.c_str(), W_GRID_GRID,
                       N_GRID, N_GRID);
  }
  GRID_GRID.weights(W_GRID_GRID);

  BCMPlasticity plast_GRID_GRID(ctx, &GRID_GRID);

  GRID.connect_input(&GRID_GRID, &plast_GRID_GRID);
  */

  RateSynapses GRID_GRID_INH(ctx, &GRID, &GRID,
                             GRID_inhibition, "GRID_GRID_INH");
  GRID_GRID_INH.weights(EigenMatrix::Ones(N_GRID, N_GRID));

  BCMPlasticity plast_GRID_GRID_INH(ctx, &GRID_GRID_INH);

  GRID.connect_input(&GRID_GRID_INH, &plast_GRID_GRID_INH);



  // Set simulation schedule:
  VIS.t_stop_after = train_time + test_on_time;
  // VIS_INH.add_schedule(VIS.t_stop_after, VIS_INH_on);
  // VIS_INH.add_schedule(infinity<FloatT>(), VIS_INH_off);

  // No inhibitory plasticity:
  plast_FVxHD_FVxHD_INH.add_schedule(infinity<FloatT>(), 0);
  plast_PLACExFVxHD_PLACExFVxHD_INH.add_schedule(infinity<FloatT>(), 0);
  plast_PLACE_PLACE_INH.add_schedule(infinity<FloatT>(), 0);
  plast_GRID_GRID_INH.add_schedule(infinity<FloatT>(), 0);

  // Rest have plasticity on only during training, with params above:
  auto add_plast_schedule = [&](FloatT dur, FloatT rate) {
    plast_VIS_PLACE.add_schedule(dur, rate);
    plast_PLACExFVxHD_PLACE.add_schedule(dur, rate);
    plast_PLACE_PLACExFVxHD.add_schedule(dur, rate);
    plast_FVxHD_PLACExFVxHD.add_schedule(dur, rate);
    plast_FV_FVxHD.add_schedule(dur, rate);
    plast_HD_FVxHD.add_schedule(dur, rate);
    plast_PLACE_GRID.add_schedule(dur, rate);
    plast_VIS_GRID.add_schedule(dur, rate);
    // plast_GRID_GRID.add_schedule(dur, rate);
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
  total_time += test_duration;
  FloatT remaining_train_time = train_time - total_time;

  add_plast_schedule(remaining_train_time, eps);
  add_plast_schedule(infinity<FloatT>(), 0);

  // Have to construct electrodes after neurons:
  RateElectrodes VIS_elecs(OUTPUT_PATH, &VIS);
  RateElectrodes HD_elecs(OUTPUT_PATH, &HD);
  RateElectrodes FV_elecs(OUTPUT_PATH, &FV);
  RateElectrodes FVxHD_elecs(OUTPUT_PATH, &FVxHD);
  RateElectrodes PLACExFVxHD_elecs(OUTPUT_PATH, &PLACExFVxHD);
  RateElectrodes PLACE_elecs(OUTPUT_PATH, &PLACE);
  RateElectrodes GRID_elecs(OUTPUT_PATH, &GRID);


  // Add Agent, Neurons and Electrodes to Model
  model.add(&agent);

  model.add(&VIS);
  model.add(&HD);
  model.add(&FV);
  model.add(&FVxHD);
  model.add(&PLACExFVxHD);
  model.add(&PLACE);
  // model.add(&GRID);

  model.add(&VIS_elecs);
  model.add(&HD_elecs);
  model.add(&FV_elecs);
  model.add(&FVxHD_elecs);
  model.add(&PLACExFVxHD_elecs);
  model.add(&PLACE_elecs);
  // model.add(&GRID_elecs);

  // Set simulation time parameters:
  model.set_simulation_time(train_time + test_on_time + test_off_time, timestep);
  model.set_buffer_intervals((float)buffer_timestep); // TODO: Use proper units
  model.set_weights_buffer_interval(ceil(120.0/timestep));
  model.set_buffer_start(start_recording_time);

  agent.save_map(OUTPUT_PATH);
  agent.record_history(OUTPUT_PATH, round(buffer_timestep/timestep),
                       round(start_recording_time/timestep));


  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.current_time(), model.dt, model.t_stop, model.timesteps());
}
