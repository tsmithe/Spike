#include "Spike/Models/RateModel.hpp"

// TODO: Add signal handlers

int main() {
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  std::cout << "0\n";

  // Set up some Neurons, Synapses and Electrodes
  RateNeurons* neurons = new RateNeurons(ctx, 1000, "test_neurons");
  neurons->rate_buffer_interval = 100;

  std::cout << "1\n";
  
  RateSynapses* synapses = new RateSynapses(ctx, neurons, neurons, "rc");
  synapses->activation_buffer_interval = 100;
  synapses->weights_buffer_interval = 100;

  std::cout << "2\n";

  RatePlasticity* plasticity = new RatePlasticity(ctx, synapses);

  std::cout << "3\n";

  neurons->init_backend(ctx);
  neurons->reset_state();
  synapses->init_backend(ctx);
  plasticity->init_backend(ctx);
  plasticity->reset_state();

  neurons->connect_input(synapses, plasticity);
  synapses->reset_state();

  // Have to construct electrodes after setting up neurons:
  RateElectrodes* electrodes = new RateElectrodes(/*ctx,*/ "tmp_out", neurons);

  std::cout << "4\n";

  // Add Neurons and Electrodes to Model
  model.add(neurons);
  model.add(electrodes);

  std::cout << "5\n";

  // Run!
  model.set_simulation_time(10, 1e-4);
  model.start();
  model.wait_for_simulation();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
  std::cout << "6\n";
}
