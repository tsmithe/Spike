// Synapse Class Header
// Synapse.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#ifndef SYNAPSES_H
#define SYNAPSES_H

#include "../Neurons/Neurons.h"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


enum CONNECTIVITY_TYPE
{
    CONNECTIVITY_TYPE_ALL_TO_ALL,
    CONNECTIVITY_TYPE_ONE_TO_ONE,
    CONNECTIVITY_TYPE_RANDOM,
    CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE,
    CONNECTIVITY_TYPE_SINGLE
};



struct synapse_parameters_struct {
	synapse_parameters_struct(): max_number_of_connections_per_pair(1), gaussian_synapses_per_postsynaptic_neuron(10), gaussian_synapses_standard_deviation(10.0), weight_range_bottom(0.0), weight_range_top(1.0), connectivity_type(CONNECTIVITY_TYPE_ALL_TO_ALL)  {}

	int max_number_of_connections_per_pair;
	int pairwise_connect_presynaptic;
	int pairwise_connect_postsynaptic;
	int gaussian_synapses_per_postsynaptic_neuron;
	float gaussian_synapses_standard_deviation;
	float weight_range_bottom;
	float weight_range_top;
	float random_connectivity_probability;
	int connectivity_type;

};


class Synapses {

public:

	// Constructor/Destructor
	Synapses();
	~Synapses();

	int total_number_of_synapses;

	curandState_t* d_states_for_random_number_generation;

	// Full Matrices
	int* presynaptic_neuron_indices;
	int* postsynaptic_neuron_indices; 
	float* synaptic_efficacies_or_weights;

	int * d_temp_presynaptic_neuron_indices;
	int* d_temp_postsynaptic_neuron_indices; 
	float* d_temp_synaptic_efficacies_or_weights;

	int *original_synapse_indices;

	// Device pointers
	int* d_presynaptic_neuron_indices;
	int* d_postsynaptic_neuron_indices;
	float* d_synaptic_efficacies_or_weights;

	int temp_number_of_synapses_in_last_group;
	int largest_synapse_group_size;
	int old_largest_number_of_blocks_x;

	bool neuron_indices_set_up_on_device;

	bool print_synapse_group_details;

	// Synapse Functions
	virtual void AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						float timestep,
						synapse_parameters_struct * synapse_params);

	virtual void allocate_device_pointers();
	virtual void copy_constants_and_initial_efficacies_to_device();


	virtual void set_threads_per_block_and_blocks_per_grid(int threads);
	virtual void increment_number_of_synapses(int increment);
	virtual void shuffle_synapses();

	dim3 number_of_synapse_blocks_per_grid;
	dim3 threads_per_block;

	
};

__global__ void compute_yes_no_connection_matrix_for_groups(bool * d_yes_no_connection_vector, 
														int pre_width, 
														int post_width, 
														int post_height, 
														float sigma, 
														int total_pre_neurons, 
														int total_post_neurons);

__global__ void set_up_neuron_indices_and_weights_for_yes_no_connection_matrix(bool * d_yes_no_connection_vector, 
																			int pre_width, 
																			int post_width, 
																			int post_height, 
																			int total_pre_neurons, 
																			int total_post_neurons, 
																			int * d_presynaptic_neuron_indices, 
																			int * d_postsynaptic_neuron_indices);

__global__ void set_neuron_indices_by_sampling_from_normal_distribution(int total_number_of_new_synapses, 
																		int postsynaptic_group_id, 
																		int poststart, 
																		int prestart, 
																		int post_width, 
																		int post_height, 
																		int pre_width, 
																		int pre_height, 
																		int number_of_new_synapses_per_postsynaptic_neuron, 
																		int number_of_postsynaptic_neurons_in_group, 
																		int * d_presynaptic_neuron_indices, 
																		int * d_postsynaptic_neuron_indices, 
																		float * d_synaptic_efficacies_or_weights, 
																		float standard_deviation_sigma, 
																		bool presynaptic_group_is_input,
																		curandState_t* d_states);

// GAUSS random number generator
double randn (double mu, double sigma);
#endif