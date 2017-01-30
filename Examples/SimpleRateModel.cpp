#include "Spike/Models/RateModel.hpp"

// TODO: Add signal handlers

int main() {
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Tell Spike to talk
  ctx->verbose = true;

  // Set up some Neurons, Synapses and Electrodes
  EigenVector neurons0_on = EigenVector::Random(100);
  EigenVector neurons0_off = EigenVector::Zero(100);
  DummyRateNeurons neurons0(ctx, 100, "dummy_input", 0, 1,
                            neurons0_on, neurons0_off);

  RateNeurons neurons1(ctx, 1000, "test_neurons1", 0, 1, 0.1);
  RateNeurons neurons2(ctx,  800, "test_neurons2", 0, 1, 0.1);

  RateSynapses synapses01(ctx, &neurons0, &neurons1, 1, "01");
  synapses01.weights(0.5 * Eigen::make_random_matrix(neurons1.size,
                                                     neurons0.size,
                                                     true, 0, 0.1));

  RateSynapses synapses11(ctx, &neurons1, &neurons1, 1, "11");
  synapses11.weights(0.2 * Eigen::make_random_matrix(neurons1.size,
                                                     neurons1.size,
                                                     true, 0, 0.2));
  RateSynapses synapses12(ctx, &neurons1, &neurons2, 1, "12");
  synapses12.weights(0.35 * Eigen::make_random_matrix(neurons2.size,
                                                      neurons1.size,
                                                      true, 0, 0));
  RateSynapses synapses21(ctx, &neurons2, &neurons1, 1, "21");
  synapses21.weights(0.3 * Eigen::make_random_matrix(neurons1.size,
                                                     neurons2.size,
                                                     true, 0, 0));
  synapses21.delay(10);
  RateSynapses synapses22(ctx, &neurons2, &neurons2, 1, "22");
  synapses22.weights(0.25 * Eigen::make_random_matrix(neurons2.size,
                                                      neurons2.size,
                                                      true, 0, -0.5));

  float eps = 0; // .001;
  RatePlasticity plasticity01(ctx, &synapses01, 0);
  RatePlasticity plasticity11(ctx, &synapses11, eps);
  RatePlasticity plasticity12(ctx, &synapses12, eps);
  // plasticity12.multipliers(EigenMatrix::Ones(neurons2.size, neurons1.size));
  RatePlasticity plasticity21(ctx, &synapses21, eps);
  RatePlasticity plasticity22(ctx, &synapses22, eps);

  // neurons1.connect_input(&synapses01, &plasticity01);
  neurons1.connect_input(&synapses11, &plasticity11);
  neurons2.connect_input(&synapses12, &plasticity12);
  neurons1.connect_input(&synapses21, &plasticity21);
  neurons2.connect_input(&synapses22, &plasticity22);

  // Have to construct electrodes after neurons:
  RateElectrodes electrodes1("tmp_out", &neurons1);
  RateElectrodes electrodes2("tmp_out", &neurons2);

  // Add Neurons and Electrodes to Model
  model.add(&neurons0);
  model.add(&neurons1);
  model.add(&neurons2);

  model.add(&electrodes1);
  model.add(&electrodes2);

  // Set simulation time parameters:
  model.set_simulation_time(10, 1e-3);
  model.set_buffer_intervals((float)0.01); // TODO: Use proper units
  model.set_weights_buffer_interval(100);

  // Run!
  model.start();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
}
