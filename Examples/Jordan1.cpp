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

  // Set parameters:
  int N_STATE = 50;
  FloatT alpha_STATE = 0.0;
  FloatT beta_STATE = 1.0;
  FloatT tau_STATE = 1e-2;

  int N_ACT = 50;
  FloatT alpha_ACT = 0.0;
  FloatT beta_ACT = 1.0;
  FloatT tau_ACT = 1e-2;

  int N_STATExACT = N_STATE * N_ACT;
  FloatT alpha_STATExACT = 0.0;
  FloatT beta_STATExACT = 1.0;
  FloatT tau_STATExACT = 1e-2;

  FloatT STATE_STATExACT_scaling = 1;
  FloatT ACT_STATExACT_scaling = 1;
  FloatT STATExACT_STATExACT_scaling = 1;

  FloatT eps = 0.1;

  // Construct neurons
  RateNeurons STATE(ctx, N_STATE, "STATE", alpha_STATE, beta_STATE, tau_STATE);
  RateNeurons ACT(ctx, N_ACT, "ACT", alpha_ACT, beta_ACT, tau_ACT);
  RateNeurons STATExACT(ctx, N_STATExACT, "STATExACT",
                        alpha_STATExACT, beta_STATExACT, tau_STATExACT);

  // Construct synapses
  RateSynapses STATE_STATExACT(ctx, &STATE, &STATExACT,
                               STATE_STATExACT_scaling, "STATE_STATExACT");
  RateSynapses ACT_STATExACT(ctx, &STATE, &STATExACT,
                             STATE_STATExACT_scaling, "STATE_STATExACT");
  RateSynapses STATExACT_STATExACT(ctx, &STATExACT, &STATExACT,
                                   STATExACT_STATExACT_scaling,
                                   "STATExACT_STATExACT");

  // Set initial weights
  STATE_STATExACT.weights(Eigen::make_random_matrix(N_STATExACT, N_STATE));
  ACT_STATExACT.weights(Eigen::make_random_matrix(N_STATExACT, N_ACT));
  STATExACT_STATExACT.weights(Eigen::make_random_matrix(N_STATExACT,
                                                        N_STATExACT));
  // STATExACT_STATExACT.make_sparse();
  // STATExACT_STATExACT.delay(ceil(axonal_delay / timestep));

  // Construct plasticity
  RatePlasticity plast_STATE_STATExACT(ctx, &STATE_STATExACT);
  RatePlasticity plast_ACT_STATExACT(ctx, &ACT_STATExACT);
  RatePlasticity plast_STATExACT_STATExACT(ctx, &STATExACT_STATExACT);

  // Connect synapses and plasticity to neurons
  STATExACT.connect_input(&STATE_STATExACT, &plast_STATE_STATExACT);
  STATExACT.connect_input(&ACT_STATExACT, &plast_ACT_STATExACT);
  STATExACT.connect_input(&STATExACT_STATExACT, &plast_STATExACT_STATExACT);

  // Set up schedule
  // ...

  // Have to construct electrodes after neurons:
  RateElectrodes STATE_elecs("BEHAV_out", &STATE);
  RateElectrodes ACT_elecs("BEHAV_out", &ACT);
  RateElectrodes STATExACT_elecs("BEHAV_out", &STATExACT);

  // Add Neurons and Electrodes to Model
  model.add(&STATE);
  model.add(&ACT);
  model.add(&STATExACT);

  model.add(&STATE_elecs);
  model.add(&ACT_elecs);
  model.add(&STATExACT_elecs);

  // Set simulation time parameters:
  model.set_simulation_time(12, timestep);
  model.set_buffer_intervals((float)1e-2); // TODO: Use proper units
  model.set_weights_buffer_interval(ceil(1.0/timestep));

  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
