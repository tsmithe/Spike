#include "Spike/Models/RateModel.hpp"

// TODO: Add signal handlers

int main() {
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  ctx->verbose = true;

  int N = 1000;

  // Set up some Neurons, Synapses and Electrodes
  RateNeurons neurons1(ctx, N, "test_neurons1", 0, 1, 0.1);
  RateNeurons neurons2(ctx, N, "test_neurons2", 0, 1, 0.1);

  RateSynapses synapses11(ctx, &neurons1, &neurons1, "11");
  synapses11.weights(0.2 * Eigen::make_random_matrix(neurons1.size,
                                                     neurons1.size,
                                                     true, 0, 0.2));
  RateSynapses synapses12(ctx, &neurons1, &neurons2, "12");
  synapses12.weights(0.35 * Eigen::make_random_matrix(neurons1.size,
                                                      neurons2.size,
                                                      true, 0, 0));
  RateSynapses synapses21(ctx, &neurons2, &neurons1, "21");
  synapses21.weights(0.3 * Eigen::make_random_matrix(neurons2.size,
                                                     neurons1.size,
                                                     true, 0, 0));
  RateSynapses synapses22(ctx, &neurons2, &neurons2, "22");
  synapses22.weights(0.25 * Eigen::make_random_matrix(neurons2.size,
                                                      neurons2.size,
                                                      true, 0, -0.5));

  RatePlasticity plasticity11(ctx, &synapses11);
  RatePlasticity plasticity12(ctx, &synapses12);
  RatePlasticity plasticity21(ctx, &synapses21);
  RatePlasticity plasticity22(ctx, &synapses22);

  neurons1.connect_input(&synapses11, &plasticity11);
  neurons2.connect_input(&synapses12, &plasticity12);
  neurons1.connect_input(&synapses21, &plasticity21);
  neurons2.connect_input(&synapses22, &plasticity22);

  // Have to construct electrodes after neurons:
  RateElectrodes electrodes1("tmp_out", &neurons1);
  RateElectrodes electrodes2("tmp_out", &neurons2);

  // Add Neurons and Electrodes to Model
  model.add(&neurons1);
  model.add(&neurons2);

  model.add(&electrodes1);
  model.add(&electrodes2);

  // Set simulation time parameters:
  model.set_simulation_time(10, 1e-3);
  model.set_buffer_intervals((float)0.01); // TODO: Use proper units
  model.set_weights_buffer_interval(1000);

  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
