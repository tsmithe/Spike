#include "Spike/Models/RateModel.hpp"
#include <fenv.h>

// TODO: Add signal handlers

int main(int argc, char *argv[]) {
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  bool read_weights = false;
  std::string weights_path;
  if (argc > 1) {
    read_weights = true;
    weights_path = argv[1];
  }

  FloatT timestep = 5e-4; // seconds (TODO units)
  
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  ctx->verbose = true;
  ctx->backend = "Eigen";

  // Set parameters
  int N_ROT = 500;
  int N_NOROT = 500;
  int N_AHV = N_ROT + N_NOROT;

  EigenVector ROT_off = EigenVector::Zero(N_AHV);
  ROT_off.head(N_NOROT) = EigenVector::Ones(N_NOROT);
  EigenVector ROT_on = EigenVector::Ones(N_AHV);
  ROT_on.head(N_NOROT) = EigenVector::Zero(N_NOROT);

  int N_VIS = 500;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 453.0;
  FloatT revs_per_sec = 1;

  int N_HD = 500;
  FloatT alpha_HD = 20.0;
  FloatT beta_HD = 0.2;
  FloatT tau_HD = 1e-2;

  EigenVector HD_VIS_INH_on = EigenVector::Ones(N_HD);
  EigenVector HD_VIS_INH_off = EigenVector::Zero(N_HD);

  FloatT VIS_INH_scaling = -400.0;

  FloatT HD_inhibition = -300.0 / N_HD; // 500


  int N_AHVxHD = N_AHV;
  FloatT alpha_AHVxHD = 20.0;
  FloatT beta_AHVxHD = 0.3;
  FloatT tau_AHVxHD = 1e-2;

  FloatT AHVxHD_HD_scaling = 22000.0 / N_AHVxHD; // 5000

  FloatT axonal_delay = 1e-2; // seconds (TODO units)

  FloatT HD_AHVxHD_scaling = 11500.0 / N_HD; // 8000

  FloatT AHV_AHVxHD_scaling = 520.0 / N_AHV; // 500

  FloatT AHVxHD_inhibition = -550.0 / N_AHVxHD; // 20000

  FloatT eps = 0.1;

  // NOTA BENE
  // 
  // The following assert reflects assumptions of the prewired model.
  // In particular, we assume that the HD cells are over-ridden by the VIS
  // input, and that the tuning of VIS is identical to the tuning of HD in
  // this case (see VIS_tuning above, and its use below as a surrogate for
  // HD_tuning).
  assert(N_VIS == N_HD);

  // Construct neurons
  DummyRateNeurons AHV(ctx, N_AHV, "AHV");

  InputDummyRateNeurons VIS(ctx, N_VIS, "VIS", sigma_VIS, lambda_VIS);
  RateNeurons HD(ctx, N_HD, "HD", alpha_HD, beta_HD, tau_HD);

  DummyRateNeurons HD_VIS_INH(ctx, N_HD, "HD_VIS_INH");

  RateNeurons AHVxHD(ctx, N_AHVxHD, "AHVxHD",
                     alpha_AHVxHD, beta_AHVxHD, tau_AHVxHD);

  // Construct synapses
  RateSynapses HD_HD(ctx, &HD, &HD, HD_inhibition, "HD_HD");

  RateSynapses VIS_HD(ctx, &VIS, &HD, 1, "VIS_HD");

  RateSynapses VIS_INH_HD(ctx, &HD_VIS_INH, &HD, VIS_INH_scaling, "VIS_INH_HD");
 
  RateSynapses HD_AHVxHD(ctx, &HD, &AHVxHD, HD_AHVxHD_scaling, "HD_AHVxHD");

  RateSynapses AHVxHD_HD(ctx, &AHVxHD, &HD, AHVxHD_HD_scaling, "AHVxHD_HD");
  RateSynapses AHVxHD_AHVxHD(ctx, &AHVxHD, &AHVxHD,
                             AHVxHD_inhibition, "AHVxHD_AHVxHD");
  RateSynapses AHV_AHVxHD(ctx, &AHV, &AHVxHD, AHV_AHVxHD_scaling, "AHV_AHVxHD");

  // Set initial weights

  // -- Fixed weights:
  HD_HD.weights(EigenMatrix::Ones(N_HD, N_HD));

  VIS_HD.weights(EigenMatrix::Identity(N_HD, N_VIS));
  VIS_HD.make_sparse();

  VIS_INH_HD.weights(EigenMatrix::Identity(N_HD, N_HD));
  VIS_INH_HD.make_sparse();

  AHVxHD_AHVxHD.weights(EigenMatrix::Ones(N_AHVxHD, N_AHVxHD));

  // -- Variable weights:
  AHVxHD_HD.delay(ceil(axonal_delay / timestep));
  EigenMatrix W_AHVxHD_HD = Eigen::make_random_matrix(N_HD, N_AHVxHD);
  if (read_weights) {
    std::string tmp_path = weights_path + "/W_AHVxHD_HD.bin";
    Eigen::read_binary(tmp_path.c_str(), W_AHVxHD_HD, N_HD, N_AHVxHD);
  }
  AHVxHD_HD.weights(W_AHVxHD_HD);

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
  AHV.add_schedule(0.23, ROT_on);
  VIS.add_schedule(0.23, revs_per_sec);

  AHV.add_schedule(0.23, ROT_off);
  VIS.add_schedule(0.23, 0);

  VIS.t_stop_after = 1000*2; // 2 seconds per full rotation

  HD_VIS_INH.add_schedule(VIS.t_stop_after, HD_VIS_INH_on);
  //plast_HD_HD.add_schedule(VIS.t_stop_after, eps);
  //plast_AHVxHD_AHVxHD.add_schedule(VIS.t_stop_after, eps);
  plast_AHVxHD_HD.add_schedule(VIS.t_stop_after, eps);
  plast_HD_AHVxHD.add_schedule(VIS.t_stop_after, eps);
  plast_AHV_AHVxHD.add_schedule(VIS.t_stop_after, eps);

  HD_VIS_INH.add_schedule(infinity<FloatT>(), HD_VIS_INH_off);
  plast_HD_HD.add_schedule(infinity<FloatT>(), 0);
  plast_AHVxHD_AHVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_AHVxHD_HD.add_schedule(infinity<FloatT>(), 0);
  plast_HD_AHVxHD.add_schedule(infinity<FloatT>(), 0);
  plast_AHV_AHVxHD.add_schedule(infinity<FloatT>(), 0);

  // Have to construct electrodes after neurons:
  RateElectrodes VIS_elecs("HD_out", &VIS);
  RateElectrodes HD_elecs("HD_out", &HD);
  RateElectrodes AHVxHD_elecs("HD_out", &AHVxHD);
  RateElectrodes AHV_elecs("HD_out", &AHV);

  // Add Neurons and Electrodes to Model
  model.add(&VIS);
  model.add(&HD);
  model.add(&HD_VIS_INH);
  model.add(&AHVxHD);
  model.add(&AHV);

  model.add(&VIS_elecs);
  model.add(&HD_elecs);
  model.add(&AHVxHD_elecs);
  model.add(&AHV_elecs);

  // Set simulation time parameters:
  model.set_simulation_time(VIS.t_stop_after + 12, timestep);
  model.set_buffer_intervals((float)1e-2); // TODO: Use proper units
  model.set_weights_buffer_interval(ceil(1.0/timestep));

  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
