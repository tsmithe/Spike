#include "Spike/Models/RateModel.hpp"
#include <fenv.h>

// TODO: Add signal handlers

int main() {
  feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  FloatT timestep = 5e-4; // seconds (TODO units)
  FloatT simulation_time = 0.6; // 4.1; // seconds (TODO units)
  
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  ctx->verbose = true;

  // Set up some Neurons, Synapses and Electrodes
  int N_ROT = 1;
  EigenVector ROT_on = EigenVector::Ones(N_ROT);
  EigenVector ROT_off = EigenVector::Zero(N_ROT);
  DummyRateNeurons ROT(ctx, N_ROT, "ROT");
  ROT.add_rate(0.6, ROT_off);
  ROT.add_rate(2, ROT_on);
  ROT.add_rate(1, ROT_off);

  int N_NOROT = 1;
  EigenVector NOROT_on = EigenVector::Ones(N_NOROT);
  EigenVector NOROT_off = EigenVector::Zero(N_NOROT);
  DummyRateNeurons NOROT(ctx, N_NOROT, "NOROT");
  NOROT.add_rate(0.6, NOROT_on);
  NOROT.add_rate(2, NOROT_off);
  NOROT.add_rate(1, NOROT_on);

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

  int N_ROTxHD = 500;
  int N_NOROTxHD = 500;
  int N_AHVxHD = N_ROTxHD + N_NOROTxHD;
  FloatT alpha_AHVxHD = 16.0;
  FloatT beta_AHVxHD = 0.2;
  FloatT tau_AHVxHD = 1e-2;
  RateNeurons AHVxHD(ctx, N_AHVxHD, "AHVxHD",
                     alpha_AHVxHD, beta_AHVxHD, tau_AHVxHD);

  
  FloatT HD_inhibition = -5;
  RateSynapses HD_HD(ctx, &HD, &HD, HD_inhibition/N_HD, "HD_HD");
  {
    EigenMatrix HD_HD_W = EigenMatrix::Ones(N_HD, N_HD);
    for (int i = 0; i < N_HD; ++i) {
      for (int j = 0; j < N_HD; ++j) {
        FloatT tmp_tuning_diff = fabs(VIS_tuning(i) - VIS_tuning(j));
        FloatT s_ij = fmin(tmp_tuning_diff, (2 * M_PI - tmp_tuning_diff));
        HD_HD_W(i, j) += 5*expf(-(s_ij*s_ij)/(sigma_VIS*sigma_VIS));
      }
    }
    HD_HD.weights(HD_HD_W);
  }

  RateSynapses VIS_HD(ctx, &VIS, &HD, 1, "VIS_HD");
  VIS_HD.weights(EigenMatrix::Identity(N_HD, N_VIS));

  // NOTA BENE
  // 
  // The following asserts reflect assumptions of the prewired model.
  // For instance, we assume that the HD cells are over-ridden by the VIS
  // input, and that the tuning of VIS is identical to the tuning of HD in
  // this case (see VIS_tuning above, and its use below as a surrogate for
  // HD_tuning). Similarly, we assume equal 'ideal' tuning in the AHVxHD
  // cell layer, replicated identically between the NOROT and ROT populations.
  //
  // Of course, this can be generalized: just need to replicate the
  // computations that produce the VIS_tuning vector for each case.
  assert(N_VIS == N_HD);
  assert(N_VIS == N_NOROTxHD);
  assert(N_VIS == N_ROTxHD);
  
  FloatT axonal_delay = 1.1e-2; // seconds (TODO units)
  FloatT HD_AHVxHD_scaling = 130.0 / N_HD;
  RateSynapses HD_AHVxHD(ctx, &HD, &AHVxHD, HD_AHVxHD_scaling, "HD_AHVxHD");
  HD_AHVxHD.delay(ceil(axonal_delay / timestep));
  FloatT sigma_HD_AHVxHD = sigma_VIS;
  FloatT V = M_PI; // radians per second (angular velocity)
  FloatT O = 3*V*axonal_delay; // offset in radians due to delay
  {
    EigenMatrix HD_AHVxHD_W = EigenMatrix::Zero(N_AHVxHD, N_HD);
    /*
    auto HD_ROTxHD_W = HD_AHVxHD_W.topRows(N_ROTxHD);
    for (int i = 0; i < N_ROTxHD; ++i) {
      for (int j = 0; j < N_HD; ++j) {
        FloatT tmp_tuning_diff = fabs(VIS_tuning(i) - (VIS_tuning(j) + O));
        FloatT s_ij = fmin(tmp_tuning_diff, (2 * M_PI - tmp_tuning_diff));
        HD_ROTxHD_W(i, j) = expf(-(s_ij*s_ij)
                                 /(2*sigma_HD_AHVxHD*sigma_HD_AHVxHD));
      }
    }
    */
    auto HD_NOROTxHD_W = HD_AHVxHD_W.bottomRows(N_NOROTxHD);
    for (int i = 0; i < N_NOROTxHD; ++i) {
      for (int j = 0; j < N_HD; ++j) {
        FloatT tmp_tuning_diff = fabs(VIS_tuning(i) - (VIS_tuning(j) + O));
        FloatT s_ij = fmin(tmp_tuning_diff, (2 * M_PI - tmp_tuning_diff));
        HD_NOROTxHD_W(i, j) = expf(-(s_ij*s_ij)
                                   /(1.5*sigma_HD_AHVxHD*sigma_HD_AHVxHD));
      }
    }
    HD_AHVxHD.weights(HD_AHVxHD_W);
  }

  FloatT AHVxHD_HD_scaling = 165.0 / N_AHVxHD;
  RateSynapses AHVxHD_HD(ctx, &AHVxHD, &HD, AHVxHD_HD_scaling, "AHVxHD_HD");
  AHVxHD_HD.delay(ceil(axonal_delay / timestep));
  AHVxHD_HD.weights(EigenMatrix(HD_AHVxHD.weights().transpose()));
  // {
  //   EigenMatrix AHVxHD_HD_W = EigenMatrix::Zero(N_HD, N_AHVxHD);
  //   /*
  //   auto HD_ROTxHD_W = HD_AHVxHD_W.topRows(N_ROTxHD);
  //   for (int i = 0; i < N_ROTxHD; ++i) {
  //     for (int j = 0; j < N_HD; ++j) {
  //       FloatT tmp_tuning_diff = fabs(VIS_tuning(i) - (VIS_tuning(j) + O));
  //       FloatT s_ij = fmin(tmp_tuning_diff, (2 * M_PI - tmp_tuning_diff));
  //       HD_ROTxHD_W(i, j) = expf(-(s_ij*s_ij)
  //                                /(2*sigma_HD_AHVxHD*sigma_HD_AHVxHD));
  //     }
  //   }
  //   */
  //   auto NOROTxHD_HD_W = AHVxHD_HD_W.rightCols(N_NOROTxHD);
  //   for (int i = 0; i < N_HD; ++i) {
  //     for (int j = 0; j < N_NOROTxHD; ++j) {
  //       FloatT tmp_tuning_diff = fabs(VIS_tuning(j) - VIS_tuning(j) - O);
  //       FloatT s_ij = fmin(tmp_tuning_diff, (2 * M_PI - tmp_tuning_diff));
  //       NOROTxHD_HD_W(i, j) = expf(-(s_ij*s_ij)
  //                                  /(2*sigma_HD_AHVxHD*sigma_HD_AHVxHD));
  //     }
  //   }
  //   //AHVxHD_HD.weights(AHVxHD_HD_W);
  // }

  FloatT AHVxHD_inhibition = -5.7;
  RateSynapses AHVxHD_AHVxHD(ctx, &AHVxHD, &AHVxHD,
                             AHVxHD_inhibition/N_AHVxHD, "AHVxHD_AHVxHD");
  AHVxHD_AHVxHD.weights(EigenMatrix::Ones(N_AHVxHD, N_AHVxHD));

  FloatT ROT_AHVxHD_scaling = 8.3 / N_ROT;
  RateSynapses ROT_AHVxHD(ctx, &ROT, &AHVxHD, ROT_AHVxHD_scaling, "ROT_AHVxHD");
  EigenMatrix ROT_AHVxHD_W = EigenMatrix::Zero(N_AHVxHD, N_ROT);
  ROT_AHVxHD_W.topRows(N_ROTxHD) = EigenMatrix::Ones(N_ROTxHD, N_ROT);
  ROT_AHVxHD.weights(ROT_AHVxHD_W);

  FloatT NOROT_AHVxHD_scaling = 8.3 / N_NOROT;
  RateSynapses NOROT_AHVxHD(ctx, &NOROT, &AHVxHD,
                            NOROT_AHVxHD_scaling, "NOROT_AHVxHD");
  EigenMatrix NOROT_AHVxHD_W = EigenMatrix::Zero(N_AHVxHD, N_NOROT);
  NOROT_AHVxHD_W.bottomRows(N_NOROTxHD) = EigenMatrix::Ones(N_NOROTxHD, N_NOROT);
  NOROT_AHVxHD.weights(NOROT_AHVxHD_W);


  float eps = 0;
  RatePlasticity plast_HD_HD(ctx, &HD_HD, eps);
  RatePlasticity plast_VIS_HD(ctx, &VIS_HD, 0);
  // plast_VIS_HD.multipliers(EigenMatrix::Ones(N_HD, N_VIS));
  RatePlasticity plast_AHVxHD_AHVxHD(ctx, &AHVxHD_AHVxHD, eps);
  RatePlasticity plast_AHVxHD_HD(ctx, &AHVxHD_HD, eps);
  RatePlasticity plast_HD_AHVxHD(ctx, &HD_AHVxHD, eps);
  RatePlasticity plast_ROT_AHVxHD(ctx, &ROT_AHVxHD, eps);
  RatePlasticity plast_NOROT_AHVxHD(ctx, &NOROT_AHVxHD, eps);

  HD.connect_input(&HD_HD, &plast_HD_HD);
  HD.connect_input(&VIS_HD, &plast_VIS_HD);
  HD.connect_input(&AHVxHD_HD, &plast_AHVxHD_HD);
  AHVxHD.connect_input(&AHVxHD_AHVxHD, &plast_AHVxHD_AHVxHD);
  AHVxHD.connect_input(&HD_AHVxHD, &plast_HD_AHVxHD);
  AHVxHD.connect_input(&ROT_AHVxHD, &plast_ROT_AHVxHD);
  AHVxHD.connect_input(&NOROT_AHVxHD, &plast_NOROT_AHVxHD);

  // Have to construct electrodes after neurons:
  RateElectrodes VIS_elecs("HD_out", &VIS);
  RateElectrodes HD_elecs("HD_out", &HD);
  RateElectrodes AHVxHD_elecs("HD_out", &AHVxHD);
  RateElectrodes ROT_elecs("HD_out", &ROT);
  RateElectrodes NOROT_elecs("HD_out", &NOROT);

  // Add Neurons and Electrodes to Model
  model.add(&VIS);
  model.add(&HD);
  model.add(&AHVxHD);
  model.add(&ROT);
  model.add(&NOROT);

  model.add(&VIS_elecs);
  model.add(&HD_elecs);
  model.add(&AHVxHD_elecs);
  model.add(&ROT_elecs);
  model.add(&NOROT_elecs);

  // Set simulation time parameters:
  model.set_simulation_time(simulation_time, timestep);
  model.set_buffer_intervals((float)2e-3); // TODO: Use proper units
  model.set_weights_buffer_interval(100);

  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
