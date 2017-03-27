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
  /*
  EigenVector neurons0_on = EigenVector::Random(100);
  EigenVector neurons0_off = EigenVector::Zero(100);
  DummyRateNeurons neurons0(ctx, 100, "dummy_input");
  neurons0.add_rate(0.05, neurons0_off);
  neurons0.add_rate(0.1, neurons0_on);
  neurons0.add_rate(0.05, neurons0_off);
  InputDummyRateNeurons neurons3(ctx, 400, "rot_input",
                                 1.5*M_PI/9, 10, 0.25);
  */

  RateNeurons neurons(ctx, /*1000,*/ "test_neurons"/*, 0, 1, 0.1*/);
  //RateNeurons neurons2(ctx,  /*800,*/ "test_neurons2"/*, 0, 1, 0.1*/);

  RateNeuronGroup neurons1 = {"test_neurons1", 1000, 0, 1, 0.1};
  RateNeuronGroup neurons2 = {"test_neurons2",  800, 0, 1, 0.1};

  neurons.add_group(&neurons1);
  neurons.add_group(&neurons2);
  
  /*
  RateSynapses synapses01(ctx, &neurons0, &neurons1, 0.1, "01");
  synapses01.weights(0.2 * Eigen::make_random_matrix(neurons1.size,
                                                     neurons0.size,
                                                     true, 0, 0.1));
  RateSynapses synapses32(ctx, &neurons3, &neurons2, 1, "32");
  synapses32.weights(0.25 * EigenMatrix::Identity(neurons2.size,
                                                  neurons3.size));
  */

  RateSynapses synapses(ctx, &neurons);
  // RateSynapses synapses2(ctx, &neurons2);

  RateSynapseGroup synapses11 = {&neurons1, &neurons1};
  synapses.add_group(&synapses11);
  synapses11.weights(0.2 * Eigen::make_random_matrix(neurons1.size,
                                                     neurons1.size,
                                                     true, 0, 0.1));
  RateSynapseGroup synapses12 = {&neurons1, &neurons2};
  synapses.add_group(&synapses12);
  synapses12.weights(0.35 * Eigen::make_random_matrix(neurons2.size,
                                                      neurons1.size,
                                                      true, 0, 0));
  // synapses12.delay(100);
  RateSynapseGroup synapses21 = {&neurons2, &neurons1};
  synapses.add_group(&synapses21);
  synapses21.weights(0.25 * Eigen::make_random_matrix(neurons1.size,
                                                     neurons2.size,
                                                     true, 0, 0));
  // synapses21.delay(100);
  RateSynapseGroup synapses22 = {&neurons2, &neurons2};
  synapses.add_group(&synapses22);
  synapses22.weights(0.1 * Eigen::make_random_matrix(neurons2.size,
                                                     neurons2.size,
                                                     true, 0, -0.5));

  /*
  float eps = 0.001;
  RatePlasticity plasticity01(ctx, &synapses01, 0);
  RatePlasticity plasticity11(ctx, &synapses11, eps);
  RatePlasticity plasticity12(ctx, &synapses12, eps);
  plasticity12.multipliers(EigenMatrix::Ones(neurons2.size, neurons1.size));
  RatePlasticity plasticity21(ctx, &synapses21, eps);
  RatePlasticity plasticity22(ctx, &synapses22, eps);
  RatePlasticity plasticity32(ctx, &synapses32, 0);
  */

  /*
  neurons1.connect_input(&synapses01, &plasticity01);
  neurons1.connect_input(&synapses11, &plasticity11);
  neurons2.connect_input(&synapses12, &plasticity12);
  neurons1.connect_input(&synapses21, &plasticity21);
  neurons2.connect_input(&synapses22, &plasticity22);
  neurons2.connect_input(&synapses32, &plasticity32);
  */

  neurons.connect_input(&synapses);

  // Have to construct electrodes after neurons:
  RateElectrodes electrodes("tmp_out", &neurons);
  // RateElectrodes electrodes1("tmp_out", &neurons1);
  // RateElectrodes electrodes2("tmp_out", &neurons2);
  // RateElectrodes electrodes3("tmp_out", &neurons3);

  // Add Neurons and Electrodes to Model
  model.add(&neurons);
  // model.add(&neurons0);
  // model.add(&neurons1);
  // model.add(&neurons2);
  // model.add(&neurons3);

  model.add(&electrodes);
  // model.add(&electrodes0);
  // model.add(&electrodes1);
  // model.add(&electrodes2);
  // model.add(&electrodes3);

  // Set simulation time parameters:
  model.set_simulation_time(100, 2e-3);
  model.set_buffer_intervals((float)0.1); // TODO: Use proper units
  model.set_weights_buffer_interval(500);

  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
