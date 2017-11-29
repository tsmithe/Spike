#include "Spike/Models/RateModel.hpp"
#include <fenv.h>

// TODO: Add signal handlers

int main() {
  //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

  FloatT timestep = 5e-4; // seconds (TODO units)
  
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  Agent agent;
  agent.velocity_scaling = 100;
  model.add(&agent);

  // Tell Spike to talk
  ctx->verbose = true;
  ctx->backend = "Eigen";

  // Set parameters:
  int N_STATE = ceil(agent.bound_x * agent.bound_y);

  int N_ACT = 10;
  FloatT alpha_ACT = 4.0;
  FloatT beta_ACT = 0.8;
  FloatT tau_ACT = 1e-2;

  int N_STATExACT = N_STATE * N_ACT;
  FloatT alpha_STATExACT = 12.0;
  FloatT beta_STATExACT = 0.2;
  FloatT tau_STATExACT = 1e-2;

  FloatT STATE_STATExACT_scaling = 83.0 / N_STATE;
  FloatT ACT_STATExACT_scaling = 83.0 / N_ACT;
  FloatT STATExACT_STATExACT_scaling = 150.0 / N_STATExACT;
  FloatT STATExACT_ACT_scaling = 30.0 / N_STATExACT;
  FloatT RAND_ACT_scaling = 10.0 / N_ACT;

  // FloatT axonal_delay = 1e-2;
  FloatT eps = 0.1;

  // Construct neurons
  AgentSenseRateNeurons STATE(ctx, &agent, "STATE");
  RandomDummyRateNeurons RAND_ACT(ctx, N_ACT, "RAND_ACT");
  RateNeurons ACT(ctx, N_ACT, "ACT", alpha_ACT, beta_ACT, tau_ACT);
  RateNeurons STATExACT(ctx, N_STATExACT, "STATExACT",
                        alpha_STATExACT, beta_STATExACT, tau_STATExACT);

  agent.connect_actor(&ACT);

  // Construct synapses
  RateSynapses STATE_STATExACT(ctx, &STATE, &STATExACT,
                               STATE_STATExACT_scaling, "STATE_STATExACT");
  RateSynapses ACT_STATExACT(ctx, &ACT, &STATExACT,
                             ACT_STATExACT_scaling, "ACT_STATExACT");
  RateSynapses STATExACT_STATExACT(ctx, &STATExACT, &STATExACT,
                                   STATExACT_STATExACT_scaling,
                                   "STATExACT_STATExACT");
  RateSynapses STATExACT_ACT(ctx, &STATExACT, &ACT,
                             STATExACT_ACT_scaling, "STATExACT_ACT");
  RateSynapses RAND_ACT_ACT(ctx, &RAND_ACT, &ACT, RAND_ACT_scaling, "RAND_ACT");

  // Set initial weights
  STATE_STATExACT.weights(Eigen::make_random_matrix(N_STATExACT, N_STATE));
  ACT_STATExACT.weights(Eigen::make_random_matrix(N_STATExACT, N_ACT));
  STATExACT_STATExACT.weights(Eigen::make_random_matrix(N_STATExACT,
                                                        N_STATExACT));
  // STATExACT_STATExACT.make_sparse();
  // STATExACT_STATExACT.delay(ceil(axonal_delay / timestep));
  STATExACT_ACT.weights(Eigen::make_random_matrix(N_ACT, N_STATExACT));
  RAND_ACT_ACT.weights(EigenMatrix::Identity(N_ACT, N_ACT));
  RAND_ACT_ACT.make_sparse();

  // Construct plasticity
  RatePlasticity plast_STATE_STATExACT(ctx, &STATE_STATExACT);
  RatePlasticity plast_ACT_STATExACT(ctx, &ACT_STATExACT);
  RatePlasticity plast_STATExACT_STATExACT(ctx, &STATExACT_STATExACT);
  RatePlasticity plast_STATExACT_ACT(ctx, &STATExACT_ACT);
  RatePlasticity plast_RAND_ACT_ACT(ctx, &RAND_ACT_ACT);

  // Connect synapses and plasticity to neurons
  STATExACT.connect_input(&STATE_STATExACT, &plast_STATE_STATExACT);
  STATExACT.connect_input(&ACT_STATExACT, &plast_ACT_STATExACT);
  STATExACT.connect_input(&STATExACT_STATExACT, &plast_STATExACT_STATExACT);
  ACT.connect_input(&STATExACT_ACT, &plast_STATExACT_ACT);
  ACT.connect_input(&RAND_ACT_ACT, &plast_RAND_ACT_ACT);

  // Set up schedule
  RAND_ACT.t_stop_after = 100;

  plast_STATE_STATExACT.add_schedule(RAND_ACT.t_stop_after, eps);
  plast_ACT_STATExACT.add_schedule(RAND_ACT.t_stop_after, eps);
  plast_STATExACT_STATExACT.add_schedule(RAND_ACT.t_stop_after, eps);
  plast_STATExACT_ACT.add_schedule(RAND_ACT.t_stop_after, eps);

  plast_STATE_STATExACT.add_schedule(infinity<FloatT>(), 0);
  plast_ACT_STATExACT.add_schedule(infinity<FloatT>(), 0);
  plast_STATExACT_STATExACT.add_schedule(infinity<FloatT>(), 0);
  plast_STATExACT_ACT.add_schedule(infinity<FloatT>(), 0);

  // Have to construct electrodes after neurons:
  RateElectrodes STATE_elecs("BEHAV_out", &STATE);
  RateElectrodes ACT_elecs("BEHAV_out", &ACT);
  RateElectrodes RAND_ACT_elecs("BEHAV_out", &RAND_ACT);
  RateElectrodes STATExACT_elecs("BEHAV_out", &STATExACT);

  // Add Neurons and Electrodes to Model
  model.add(&STATE);
  model.add(&ACT);
  model.add(&RAND_ACT);
  model.add(&STATExACT);

  model.add(&STATE_elecs);
  model.add(&ACT_elecs);
  model.add(&RAND_ACT_elecs);
  model.add(&STATExACT_elecs);

  // Set simulation time parameters:
  model.set_simulation_time(2*RAND_ACT.t_stop_after, timestep);
  model.set_buffer_intervals((float)1e-2); // TODO: Use proper units
  model.set_weights_buffer_interval(ceil(0.25/timestep));

  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
