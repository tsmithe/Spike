#include "Spike/Models/RateModel.hpp"
#include <fenv.h>

// TODO: Add signal handlers

int main() {
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  FloatT timestep = 5e-4; // seconds (TODO units)
  
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  ctx->verbose = true;
  ctx->backend = "Eigen";

  // Set up some Neurons, Synapses and Electrodes
  int N_ROT = 500;
  int N_NOROT = 500;
  int N_AHV = N_ROT + N_NOROT;
  EigenVector ROT_off = EigenVector::Zero(N_AHV);
  ROT_off.head(N_NOROT) = EigenVector::Ones(N_NOROT);
  EigenVector ROT_on = EigenVector::Ones(N_AHV);
  ROT_on.head(N_NOROT) = EigenVector::Zero(N_NOROT);
  DummyRateNeurons AHV(ctx, N_AHV, "AHV");
  AHV.add_rate(0.6, ROT_on);
  AHV.add_rate(0.6, ROT_off);

  int N_VIS = 500;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 450.0;
  FloatT revs_per_sec = 1;
  InputDummyRateNeurons VIS(ctx, N_VIS, "VIS", sigma_VIS, lambda_VIS);
  VIS.add_rate(0.6, revs_per_sec);
  VIS.add_rate(0.6, 0);
  VIS.t_stop_after = 50*1.2;
  auto& VIS_tuning = VIS.theta_pref;

  int N_HD = 500;
  FloatT alpha_HD = 4.2;
  FloatT beta_HD = 0.2;
  FloatT tau_HD = 1e-2;
  RateNeurons HD(ctx, N_HD, "HD", alpha_HD, beta_HD, tau_HD);

  DummyRateNeurons HD_VIS_INH(ctx, N_HD, "HD_VIS_INH");
  EigenVector HD_VIS_INH_on = EigenVector::Ones(N_HD);
  EigenVector HD_VIS_INH_off = EigenVector::Zero(N_HD);
  HD_VIS_INH.add_rate(VIS.t_stop_after, HD_VIS_INH_on);
  HD_VIS_INH.add_rate(100.0, HD_VIS_INH_off);

  int N_AHVxHD = N_AHV;
  FloatT alpha_AHVxHD = 113.0;
  FloatT beta_AHVxHD = 0.3;
  FloatT tau_AHVxHD = 1e-2;
  RateNeurons AHVxHD(ctx, N_AHVxHD, "AHVxHD",
                     alpha_AHVxHD, beta_AHVxHD, tau_AHVxHD);
  
  FloatT HD_inhibition = -1;
  RateSynapses HD_HD(ctx, &HD, &HD, HD_inhibition/N_HD, "HD_HD");
  EigenMatrix HD_HD_W = EigenMatrix::Ones(N_HD, N_HD);
  HD_HD.weights(HD_HD_W);

  RateSynapses VIS_HD(ctx, &VIS, &HD, 1, "VIS_HD");
  VIS_HD.weights(EigenMatrix::Identity(N_HD, N_VIS));
  VIS_HD.make_sparse();

  RateSynapses VIS_INH_HD(ctx, &HD_VIS_INH, &HD, -420.0, "VIS_INH_HD");
  VIS_INH_HD.weights(EigenMatrix::Identity(N_HD, N_HD));
  VIS_INH_HD.make_sparse();

  // NOTA BENE
  // 
  // The following assert reflects assumptions of the prewired model.
  // In particular, we assume that the HD cells are over-ridden by the VIS
  // input, and that the tuning of VIS is identical to the tuning of HD in
  // this case (see VIS_tuning above, and its use below as a surrogate for
  // HD_tuning).
  assert(N_VIS == N_HD);
  
  FloatT axonal_delay = 1e-2; // seconds (TODO units)
  FloatT HD_AHVxHD_scaling = 2400.0 / (0.05*N_HD);
  RateSynapses HD_AHVxHD(ctx, &HD, &AHVxHD, HD_AHVxHD_scaling, "HD_AHVxHD");
  HD_AHVxHD.delay(ceil(axonal_delay / timestep));
  HD_AHVxHD.weights(Eigen::make_random_matrix(N_AHVxHD, N_HD, 1.0, true,
                                              0.95, 0, false));
  HD_AHVxHD.make_sparse();

  FloatT AHVxHD_HD_scaling = 10000.0 / N_AHVxHD;
  RateSynapses AHVxHD_HD(ctx, &AHVxHD, &HD, AHVxHD_HD_scaling, "AHVxHD_HD");
  AHVxHD_HD.delay(ceil(axonal_delay / timestep));
  AHVxHD_HD.weights(Eigen::make_random_matrix(N_HD, N_AHVxHD));

  FloatT AHVxHD_inhibition = -1000.0;
  RateSynapses AHVxHD_AHVxHD(ctx, &AHVxHD, &AHVxHD,
                             AHVxHD_inhibition/N_AHVxHD, "AHVxHD_AHVxHD");
  AHVxHD_AHVxHD.weights(EigenMatrix::Ones(N_AHVxHD, N_AHVxHD));

  FloatT AHV_AHVxHD_scaling = 5000.0 / N_AHV;
  RateSynapses AHV_AHVxHD(ctx, &AHV, &AHVxHD, AHV_AHVxHD_scaling, "AHV_AHVxHD");
  AHV_AHVxHD.weights(Eigen::make_random_matrix(N_AHVxHD, N_AHV));

  float eps = 0.1;
  RatePlasticity plast_HD_HD(ctx, &HD_HD, eps);
  RatePlasticity plast_VIS_HD(ctx, &VIS_HD, 0);
  RatePlasticity plast_VIS_INH_HD(ctx, &VIS_INH_HD, 0);
  RatePlasticity plast_AHVxHD_AHVxHD(ctx, &AHVxHD_AHVxHD, eps);
  RatePlasticity plast_AHVxHD_HD(ctx, &AHVxHD_HD, eps);
  RatePlasticity plast_HD_AHVxHD(ctx, &HD_AHVxHD, eps);
  RatePlasticity plast_AHV_AHVxHD(ctx, &AHV_AHVxHD, eps);

  HD.connect_input(&HD_HD, &plast_HD_HD);
  HD.connect_input(&VIS_HD, &plast_VIS_HD);
  HD.connect_input(&VIS_INH_HD, &plast_VIS_INH_HD);
  HD.connect_input(&AHVxHD_HD, &plast_AHVxHD_HD);
  AHVxHD.connect_input(&AHVxHD_AHVxHD, &plast_AHVxHD_AHVxHD);
  AHVxHD.connect_input(&HD_AHVxHD, &plast_HD_AHVxHD);
  AHVxHD.connect_input(&AHV_AHVxHD, &plast_AHV_AHVxHD);

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
