#include "Spike/Models/RateModel.hpp"
#include <fenv.h>

// TODO: Add signal handlers

#define ENABLE_COMB

int main(int argc, char *argv[]) {
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  bool read_weights = false;
  std::string weights_path;

  if (argc == 2) {
    read_weights = true;
    weights_path = argv[1];
  }

  FloatT timestep = 5e-4; // seconds (TODO units)
  FloatT train_time = 800;
  if (read_weights) train_time = 0;
  FloatT test_on_time = 10;
  FloatT test_off_time = 20;
  
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  //ctx->verbose = true;
  ctx->backend = "Eigen";

  // Set parameters
  int N_ROT = 50;
  int N_NOROT = 50;
  int N_AHV = N_ROT + N_NOROT;

  EigenVector ROT_off = EigenVector::Zero(N_AHV);
  ROT_off.head(N_NOROT) = EigenVector::Ones(N_NOROT);
  EigenVector ROT_on = EigenVector::Ones(N_AHV);
  ROT_on.head(N_NOROT) = EigenVector::Zero(N_NOROT);

  int N_VIS = 200;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 1.0;
  FloatT revs_per_sec = 1;

  int N_HD = 400;
  FloatT alpha_HD = 20.0;
  FloatT beta_HD = 0.6;
  FloatT tau_HD = 1e-2;

  EigenVector HD_VIS_INH_on = EigenVector::Ones(N_HD);
  EigenVector HD_VIS_INH_off = EigenVector::Zero(N_HD);

  int N_AHVxHD = 800;
  FloatT alpha_AHVxHD = 20.0;
  FloatT beta_AHVxHD = 0.8;
  FloatT tau_AHVxHD = 1e-2;

  FloatT VIS_HD_scaling = 780.0 / (N_VIS*0.05); // 1200 / (N_VIS*0.05);

  FloatT VIS_INH_scaling = -1.0 / (N_VIS*0.05); // -160 / N_HD;
  FloatT HD_inhibition = -300.0 / N_HD; // 300

  FloatT AHVxHD_HD_scaling = 6000.0 / (N_AHVxHD*1.0); // 5000

  FloatT HD_AHVxHD_scaling = 500.0 / (N_HD*0.05); // 8000
  FloatT AHV_AHVxHD_scaling = 240.0 / N_AHV; // 500
  FloatT AHVxHD_inhibition = -250.0 / N_AHVxHD; // 20000

  if (argc == 3) {
    /*
    AHVxHD_HD_scaling = atof(argv[1]) / N_AHVxHD;
    HD_AHVxHD_scaling = atof(argv[2]) / (N_HD*0.05);
    AHVxHD_inhibition = -atof(argv[3]) / N_AHVxHD;
    */
    VIS_INH_scaling = -atof(argv[1]) / (N_VIS*0.05);
    HD_inhibition = -atof(argv[2]) / N_HD;

    std::cout << argv[1] << ", " << argv[2] /*<< ", " << argv[3] /*<< ", " << argv[4]*/ << "\n";
  } else if (argc > 2 && argc != 3) {
    printf("ERROR COUNTING ARGV\n");
    return 64;
  }

  FloatT axonal_delay = 1e-2; // seconds (TODO units)

  FloatT eps_VIS_HD = 0.2;
  FloatT eps = 0.12;

   // Construct neurons
  DummyRateNeurons AHV(ctx, N_AHV, "AHV");

  InputDummyRateNeurons VIS(ctx, N_VIS, "VIS", sigma_VIS, lambda_VIS);
  InputDummyRateNeurons VIS_cf(ctx, N_VIS, "VIS_cf", sigma_VIS, lambda_VIS);

  RateNeurons HD(ctx, N_HD, "HD", alpha_HD, beta_HD, tau_HD);

  DummyRateNeurons HD_VIS_INH(ctx, N_HD, "HD_VIS_INH");

  RateNeurons AHVxHD(ctx, N_AHVxHD, "AHVxHD",
                     alpha_AHVxHD, beta_AHVxHD, tau_AHVxHD);

  // Construct synapses
  RateSynapses HD_HD(ctx, &HD, &HD, HD_inhibition, "HD_HD");

  RateSynapses VIS_HD(ctx, &VIS, &HD, VIS_HD_scaling, "VIS_HD");

  RateSynapses VIS_INH_HD(ctx, &HD_VIS_INH, &HD, VIS_INH_scaling, "VIS_INH_HD");
  // RateSynapses VIS_INH_HD(ctx, &HD, &HD, VIS_INH_scaling, "VIS_INH_HD");
 
  RateSynapses HD_AHVxHD(ctx, &HD, &AHVxHD, HD_AHVxHD_scaling, "HD_AHVxHD");

  RateSynapses AHVxHD_HD(ctx, &AHVxHD, &HD, AHVxHD_HD_scaling, "AHVxHD_HD");
  RateSynapses AHVxHD_AHVxHD(ctx, &AHVxHD, &AHVxHD,
                             AHVxHD_inhibition, "AHVxHD_AHVxHD");
  RateSynapses AHV_AHVxHD(ctx, &AHV, &AHVxHD, AHV_AHVxHD_scaling, "AHV_AHVxHD");

  // Set initial weights

  // -- Fixed weights:
  HD_HD.weights(EigenMatrix::Ones(N_HD, N_HD));

  VIS_INH_HD.weights(EigenMatrix::Ones(N_HD, N_HD));

  AHVxHD_AHVxHD.weights(EigenMatrix::Ones(N_AHVxHD, N_AHVxHD));

  // -- Variable weights:
  EigenMatrix W_VIS_HD = Eigen::make_random_matrix(N_HD, N_VIS, 1.0, true, 0.95, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_VIS_HD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_VIS_HD, N_HD, N_VIS);
  }
  VIS_HD.weights(W_VIS_HD);
  VIS_HD.make_sparse();

  AHVxHD_HD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_AHVxHD_HD = Eigen::make_random_matrix(N_HD, N_AHVxHD);
  // EigenMatrix W_AHVxHD_HD = Eigen::make_random_matrix(N_HD, N_AHVxHD, 1.0, true, 0.4, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_AHVxHD_HD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_AHVxHD_HD, N_HD, N_AHVxHD);
  }
  AHVxHD_HD.weights(W_AHVxHD_HD);
  // AHVxHD_HD.make_sparse();

  HD_AHVxHD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_HD_AHVxHD = Eigen::make_random_matrix(N_AHVxHD, N_HD, 1.0, true, 0.95, 0, false);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_HD_AHVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_HD_AHVxHD, N_AHVxHD, N_HD);
  }
  HD_AHVxHD.weights(W_HD_AHVxHD);
  HD_AHVxHD.make_sparse();

  EigenMatrix W_AHV_AHVxHD = Eigen::make_random_matrix(N_AHVxHD, N_AHV);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_AHV_AHVxHD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_AHV_AHVxHD, N_AHVxHD, N_AHV);
  }
  AHV_AHVxHD.weights(W_AHV_AHVxHD);

  // Construct plasticity
  RatePlasticity plast_HD_HD(ctx, &HD_HD);
  RatePlasticity plast_VIS_HD(ctx, &VIS_HD);
  RatePlasticity plast_VIS_INH_HD(ctx, &VIS_INH_HD);
  RatePlasticity plast_AHVxHD_AHVxHD(ctx, &AHVxHD_AHVxHD);
  RatePlasticity plast_AHVxHD_HD(ctx, &AHVxHD_HD);
  RatePlasticity plast_HD_AHVxHD(ctx, &HD_AHVxHD);
  RatePlasticity plast_AHV_AHVxHD(ctx, &AHV_AHVxHD);

  // Connect synapses and plasticity to neurons
  HD.connect_input(&HD_HD, &plast_HD_HD);
  HD.connect_input(&VIS_HD, &plast_VIS_HD);
  HD.connect_input(&VIS_INH_HD, &plast_VIS_INH_HD);
  HD.connect_input(&AHVxHD_HD, &plast_AHVxHD_HD);
  AHVxHD.connect_input(&AHVxHD_AHVxHD, &plast_AHVxHD_AHVxHD);
  AHVxHD.connect_input(&HD_AHVxHD, &plast_HD_AHVxHD);
  AHVxHD.connect_input(&AHV_AHVxHD, &plast_AHV_AHVxHD);

  // Set up schedule
  // + cycle between ROT_on and ROT_off every 0.2s, until VIS.t_stop_after
  // + after VIS.t_stop_after, turn off plasticity
  AHV.add_schedule(0.37, ROT_on);
  VIS.add_schedule(0.37, revs_per_sec);
  VIS_cf.add_schedule(0.37, revs_per_sec);

  /*
#ifdef ENABLE_COMB
  AHV.add_schedule(0.37, ROT_off);
  VIS.add_schedule(0.37, 0);
  VIS_cf.add_schedule(0.37, 0);
#endif
  */

  VIS.t_stop_after = train_time + test_on_time;

  HD_VIS_INH.add_schedule(VIS.t_stop_after, HD_VIS_INH_on);
  /*
  //plast_HD_HD.add_schedule(VIS.t_stop_after, eps);
  //plast_AHVxHD_AHVxHD.add_schedule(VIS.t_stop_after, eps);
  plast_VIS_HD.add_schedule(10, 1.0);
  plast_VIS_HD.add_schedule(20, 0.5);
  plast_VIS_HD.add_schedule(70, eps_VIS_HD*2);
  plast_VIS_HD.add_schedule(2, 0);
  plast_VIS_HD.add_schedule(100, eps_VIS_HD);
  plast_VIS_HD.add_schedule(2, 0);
  plast_VIS_HD.add_schedule(292, eps_VIS_HD);
  plast_VIS_HD.add_schedule(2, 0);
  plast_VIS_HD.add_schedule(VIS.t_stop_after-500, eps_VIS_HD*0.6);
  plast_VIS_HD.add_schedule(2, 0);
  */
  plast_VIS_HD.add_schedule(train_time, eps_VIS_HD);
#ifdef ENABLE_COMB
  plast_AHVxHD_HD.add_schedule(train_time, eps);
  plast_HD_AHVxHD.add_schedule(train_time, eps*1.25);

  plast_AHV_AHVxHD.add_schedule(train_time, eps*0.8);

  HD_VIS_INH.add_schedule(infinity<FloatT>(), HD_VIS_INH_off);
#endif

  plast_VIS_HD.add_schedule(infinity<FloatT>(), 0);
  plast_HD_HD.add_schedule(infinity<FloatT>(), 0);
  plast_AHVxHD_AHVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_AHVxHD_HD.add_schedule(infinity<FloatT>(), 0);
  plast_HD_AHVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_AHV_AHVxHD.add_schedule(infinity<FloatT>(), 0);

  // Have to construct electrodes after neurons:
  RateElectrodes VIS_elecs("HD_VIS_out", &VIS);
  RateElectrodes VIS_cf_elecs("HD_VIS_out", &VIS_cf);
  RateElectrodes HD_elecs("HD_VIS_out", &HD);
  RateElectrodes AHVxHD_elecs("HD_VIS_out", &AHVxHD);
  RateElectrodes AHV_elecs("HD_VIS_out", &AHV);

  // Add Neurons and Electrodes to Model
  model.add(&VIS);
  model.add(&VIS_cf);
  model.add(&HD);
  model.add(&HD_VIS_INH);
#ifdef ENABLE_COMB
  model.add(&AHVxHD);
  model.add(&AHV);
#endif

  model.add(&VIS_elecs);
  model.add(&VIS_cf_elecs);
  model.add(&HD_elecs);
#ifdef ENABLE_COMB
  model.add(&AHVxHD_elecs);
  model.add(&AHV_elecs);
#endif

  // Set simulation time parameters:
  model.set_simulation_time(train_time + test_on_time + test_off_time, timestep);
  model.set_buffer_intervals((float)1e-2); // TODO: Use proper units
  model.set_weights_buffer_interval(ceil(2.0/timestep));

  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
