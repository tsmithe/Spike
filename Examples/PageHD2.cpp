#include "Spike/Models/RateModel.hpp"
#include <fenv.h>

// TODO: Add signal handlers

int main() {
  feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  FloatT timestep = 5e-4; // seconds (TODO units)
  FloatT simulation_time = 3.6; // 4.1; // seconds (TODO units)
  
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
  EigenVector ROT_on = EigenVector::Ones(N_ROT);
  ROT_off.head(N_NOROT) = EigenVector::Zero(N_NOROT);
  DummyRateNeurons AHV(ctx, N_AHV, "AHV");
  AHV.add_rate(0.6, ROT_off);
  AHV.add_rate(2, ROT_on);
  AHV.add_rate(1, ROT_off);

  int N_VIS = 500;
  FloatT sigma_VIS = M_PI / 9;
  FloatT lambda_VIS = 8;
  FloatT revs_per_sec_VIS = 0; // NB: Initialisation is static
  InputDummyRateNeurons VIS(ctx, N_VIS, "VIS",
                            sigma_VIS, lambda_VIS, revs_per_sec_VIS);
  VIS.t_stop_after = 0.1;
  auto& VIS_tuning = VIS.theta_pref;

  int N_HD = 500;
  FloatT alpha_HD = 4.2;
  FloatT beta_HD = 0.7;
  FloatT tau_HD = 1e-2;
  RateNeurons HD(ctx, N_HD, "HD", alpha_HD, beta_HD, tau_HD);

  int N_AHVxHD = N_AHV;
  FloatT alpha_AHVxHD = 16.0;
  FloatT beta_AHVxHD = 0.2;
  FloatT tau_AHVxHD = 1e-2;
  RateNeurons AHVxHD(ctx, N_AHVxHD, "AHVxHD",
                     alpha_AHVxHD, beta_AHVxHD, tau_AHVxHD);
  
  FloatT HD_inhibition = -5;
  RateSynapses HD_HD(ctx, &HD, &HD, HD_inhibition/N_HD, "HD_HD");
  {
    EigenMatrix HD_HD_W = EigenMatrix::Ones(N_HD, N_HD);
    HD_HD.weights(HD_HD_W);
  }

  RateSynapses VIS_HD(ctx, &VIS, &HD, 1, "VIS_HD");
  VIS_HD.weights(EigenMatrix::Identity(N_HD, N_VIS));

  // NOTA BENE
  // 
  // The following assert reflects assumptions of the prewired model.
  // In particular, we assume that the HD cells are over-ridden by the VIS
  // input, and that the tuning of VIS is identical to the tuning of HD in
  // this case (see VIS_tuning above, and its use below as a surrogate for
  // HD_tuning).
  assert(N_VIS == N_HD);
  
  FloatT axonal_delay = 1.1e-2; // seconds (TODO units)
  FloatT HD_AHVxHD_scaling = 130.0 / N_HD;
  RateSynapses HD_AHVxHD(ctx, &HD, &AHVxHD, HD_AHVxHD_scaling, "HD_AHVxHD");
  HD_AHVxHD.delay(ceil(axonal_delay / timestep));
  {
    EigenMatrix HD_AHVxHD_W = EigenMatrix::Zero(N_AHVxHD, N_HD);
    HD_AHVxHD.weights(HD_AHVxHD_W);
  }
  HD_AHVxHD.make_sparse();

  FloatT AHVxHD_HD_scaling = 165.0 / N_AHVxHD;
  RateSynapses AHVxHD_HD(ctx, &AHVxHD, &HD, AHVxHD_HD_scaling, "AHVxHD_HD");
  AHVxHD_HD.delay(ceil(axonal_delay / timestep));
  {
    EigenMatrix AHVxHD_HD_W = EigenMatrix::Zero(N_HD, N_AHVxHD);
    // HD_AHVxHD.get_weights(HD_AHVxHD_W);
    AHVxHD_HD.weights(AHVxHD_HD_W);
  }

  FloatT AHVxHD_inhibition = -5.7;
  RateSynapses AHVxHD_AHVxHD(ctx, &AHVxHD, &AHVxHD,
                             AHVxHD_inhibition/N_AHVxHD, "AHVxHD_AHVxHD");
  AHVxHD_AHVxHD.weights(EigenMatrix::Ones(N_AHVxHD, N_AHVxHD));

  FloatT AHV_AHVxHD_scaling = 8.3 / N_AHV;
  RateSynapses AHV_AHVxHD(ctx, &AHV, &AHVxHD, AHV_AHVxHD_scaling, "AHV_AHVxHD");
  EigenMatrix AHV_AHVxHD_W = EigenMatrix::Zero(N_AHVxHD, N_AHV);
  AHV_AHVxHD.weights(AHV_AHVxHD_W);

  float eps = 0.01;
  RatePlasticity plast_HD_HD(ctx, &HD_HD, eps);
  RatePlasticity plast_VIS_HD(ctx, &VIS_HD, 0);
  RatePlasticity plast_AHVxHD_AHVxHD(ctx, &AHVxHD_AHVxHD, eps);
  RatePlasticity plast_AHVxHD_HD(ctx, &AHVxHD_HD, eps);
  RatePlasticity plast_HD_AHVxHD(ctx, &HD_AHVxHD, eps);
  RatePlasticity plast_AHV_AHVxHD(ctx, &AHV_AHVxHD, eps);

  HD.connect_input(&HD_HD, &plast_HD_HD);
  HD.connect_input(&VIS_HD, &plast_VIS_HD);
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
  model.add(&AHVxHD);
  model.add(&AHV);

  model.add(&VIS_elecs);
  model.add(&HD_elecs);
  model.add(&AHVxHD_elecs);
  model.add(&AHV_elecs);

  // Set simulation time parameters:
  model.set_simulation_time(simulation_time, timestep);
  model.set_buffer_intervals((float)2e-3); // TODO: Use proper units
  model.set_weights_buffer_interval(100);

  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
