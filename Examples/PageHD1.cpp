#include "Spike/Models/RateModel.hpp"
#include <fenv.h>

// TODO: Add signal handlers

int main() {
  feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  ctx->verbose = true;

  // Set up some Neurons, Synapses and Electrodes
  EigenVector neurons0_on = EigenVector::Random(10);
  EigenVector neurons0_off = EigenVector::Zero(10);
  DummyRateNeurons ROT(ctx, 10, "ROT", 0, 0.1,
                       neurons0_on, neurons0_off);
  DummyRateNeurons NOROT(ctx, 10, "NOROT", 0, 0.1,
                         neurons0_off, neurons0_on);

  InputDummyRateNeurons VIS(ctx, 500, "VIS",
                            1.5*M_PI/18, 100, 0.25);

  RateNeurons HD(ctx, 500, "HD", 0, 1, 0.1);
  RateNeurons AHVxHD(ctx, 1000, "AHVxHD", 0, 1, 0.1);

  RateSynapses VIS_HD(ctx, &VIS, &HD, 1, "VIS_HD");
  VIS_HD.weights(0.2 * Eigen::make_random_matrix(HD.size,
                                                 VIS.size,
                                                 true, 0, 0.1));

  RateSynapses HD_HD(ctx, &HD, &HD, 1, "HD_HD");
  HD_HD.weights(0.2 * Eigen::make_random_matrix(HD.size,
                                                HD.size,
                                                true, 0, 0.1));

  RateSynapses AHVxHD_HD(ctx, &AHVxHD, &HD, 1, "AHVxHD_HD");
  AHVxHD_HD.weights(0.35 * Eigen::make_random_matrix(HD.size,
                                                     AHVxHD.size,
                                                     true, 0, 0));
  AHVxHD_HD.delay(100);

  RateSynapses HD_AHVxHD(ctx, &HD, &AHVxHD, 1, "HD_AHVxHV");
  HD_AHVxHD.weights(0.2 * Eigen::make_random_matrix(AHVxHD.size,
                                                    HD.size,
                                                    true, 0, 0.1));
  HD_AHVxHD.delay(100);

  RateSynapses ROT_AHVxHD(ctx, &ROT, &AHVxHD, 1, "ROT_AHVxHD");
  ROT_AHVxHD.weights(0.3 * Eigen::make_random_matrix(AHVxHD.size,
                                                     ROT.size,
                                                     true, 0, 0));

  RateSynapses NOROT_AHVxHD(ctx, &NOROT, &AHVxHD, 1, "NOROT_AHVxHD");
  NOROT_AHVxHD.weights(0.3 * Eigen::make_random_matrix(AHVxHD.size,
                                                       NOROT.size,
                                                       true, 0, 0));

  float eps = 0.0001;
  RatePlasticity plast_VIS_HD(ctx, &VIS_HD, 0);
  RatePlasticity plast_HD_HD(ctx, &HD_HD, eps);
  // plast_VIS_HD.multipliers(EigenMatrix::Ones(HD.size, VIS.size));
  RatePlasticity plast_AHVxHD_HD(ctx, &AHVxHD_HD, eps);
  RatePlasticity plast_HD_AHVxHD(ctx, &HD_AHVxHD, eps);
  RatePlasticity plast_ROT_AHVxHD(ctx, &ROT_AHVxHD, eps);
  RatePlasticity plast_NOROT_AHVxHD(ctx, &NOROT_AHVxHD, eps);

  HD.connect_input(&VIS_HD, &plast_VIS_HD);
  HD.connect_input(&HD_HD, &plast_HD_HD);
  HD.connect_input(&AHVxHD_HD, &plast_AHVxHD_HD);
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
  model.set_simulation_time(10, 1e-3);
  model.set_buffer_intervals((float)0.05); // TODO: Use proper units
  model.set_weights_buffer_interval(100);

  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
