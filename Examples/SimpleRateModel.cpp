#include "Spike/Models/RateModel.hpp"

// TODO: Add signal handlers

int main() {
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  // Set up some Neurons, Synapses and Electrodes
  RateNeurons neurons(ctx, 1000, "test_neurons");
  neurons.rate_buffer_interval = 100;

  RateSynapses synapses(ctx, &neurons, &neurons, "rc");
  synapses.activation_buffer_interval = 100;
  synapses.weights_buffer_interval = 100;

  RatePlasticity plasticity(ctx, &synapses);

  neurons.connect_input(&synapses, &plasticity);

  // Have to construct electrodes after setting up neurons:
  RateElectrodes electrodes(ctx, "tmp_out", &neurons);

  // Add Neurons and Electrodes to Model
  model.add(&neurons);
  model.add(&electrodes);

  // Run!
  model.set_simulation_time(10, 1e-4);
  model.start();
}
