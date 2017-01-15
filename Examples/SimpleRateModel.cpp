#include "Spike/Models/RateModel.hpp"

// TODO: Add signal handlers

int main() {
  // Create Model
  RateModel model;
  Context* ctx = model.context;

  std::cout << "0\n";

  // Set up some Neurons, Synapses and Electrodes
  RateNeurons* neurons1 = new RateNeurons(ctx, 1000, "test_neurons1");
  neurons1->rate_buffer_interval = 100;

  RateNeurons* neurons2 = new RateNeurons(ctx, 1000, "test_neurons2");
  neurons2->rate_buffer_interval = 100;

  std::cout << "1\n";
  
  RateSynapses* synapses11 = new RateSynapses(ctx, neurons1, neurons1, "rc");
  RateSynapses* synapses12 = new RateSynapses(ctx, neurons1, neurons2, "rc");
  RateSynapses* synapses21 = new RateSynapses(ctx, neurons2, neurons1, "rc");
  RateSynapses* synapses22 = new RateSynapses(ctx, neurons2, neurons2, "rc");

  synapses11->activation_buffer_interval = 100;
  synapses11->weights_buffer_interval = 100;

  synapses12->activation_buffer_interval = 100;
  synapses12->weights_buffer_interval = 100;

  synapses21->activation_buffer_interval = 100;
  synapses21->weights_buffer_interval = 100;

  synapses22->activation_buffer_interval = 100;
  synapses22->weights_buffer_interval = 100;

  std::cout << "2\n";

  RatePlasticity* plasticity11 = new RatePlasticity(ctx, synapses11);
  RatePlasticity* plasticity12 = new RatePlasticity(ctx, synapses12);
  RatePlasticity* plasticity21 = new RatePlasticity(ctx, synapses21);
  RatePlasticity* plasticity22 = new RatePlasticity(ctx, synapses22);

  std::cout << "3\n";

  neurons1->init_backend(ctx);
  neurons1->reset_state();
  neurons2->init_backend(ctx);
  neurons2->reset_state();

  synapses11->init_backend(ctx);
  synapses12->init_backend(ctx);
  synapses21->init_backend(ctx);
  synapses22->init_backend(ctx);

  plasticity11->init_backend(ctx);
  plasticity11->reset_state();
  plasticity12->init_backend(ctx);
  plasticity12->reset_state();
  plasticity21->init_backend(ctx);
  plasticity21->reset_state();
  plasticity22->init_backend(ctx);
  plasticity22->reset_state();

  neurons1->connect_input(synapses11, plasticity11);
  neurons2->connect_input(synapses12, plasticity12);
  neurons1->connect_input(synapses21, plasticity21);
  neurons2->connect_input(synapses22, plasticity22);

  synapses11->reset_state();
  synapses12->reset_state();
  synapses21->reset_state();
  synapses22->reset_state();

  // Have to construct electrodes after setting up neurons:
  RateElectrodes* electrodes1 = new RateElectrodes(/*ctx,*/ "tmp_out", neurons1);
  RateElectrodes* electrodes2 = new RateElectrodes(/*ctx,*/ "tmp_out", neurons2);

  std::cout << "4\n";

  // Add Neurons and Electrodes to Model
  model.add(neurons1);
  model.add(neurons2);
  model.add(electrodes1);
  model.add(electrodes2);

  std::cout << "5\n";

  // Run!
  model.set_simulation_time(10, 1e-4);
  model.start();
  model.wait_for_simulation();

  printf("%f, %f, %f; %d\n", model.t, model.dt, model.t_stop, model.timesteps);
  std::cout << "6\n";
}
