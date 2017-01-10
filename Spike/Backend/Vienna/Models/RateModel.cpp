#include "RateModel.hpp"

namespace Backend {
  namespace Vienna {
    void RateNeurons::prepare() {
      for (const auto& front_dendrite_pair : frontend()->dendrites) {
        // NB: Here assume backend is Vienna throughout:
        ::Backend::Vienna::RateSynapses* synapses
          = dynamic_cast<::Backend::Vienna::RateSynapses*>
              (front_dendrite_pair.first->backend());
        ::Backend::Vienna::RatePlasticity* plasticity
          = dynamic_cast<::Backend::Vienna::RatePlasticity*>
              (front_dendrite_pair.second->backend());
        _dendrites.push_back(std::make_pair(synapses, plasticity));
      }
      reset_state();
    }

    void RateNeurons::reset_state() {
      rates = viennacl::vector<float>(frontend()->size);
    }

    void RateNeurons::push_data_front() {
      viennacl::copy(rates, frontend()->rates);
    }

    void RateNeurons::pull_data_back() {
      viennacl::copy(frontend()->rates, rates);
    }

    void RateNeurons::update_rate(float dt) {
      viennacl::vector<float> total_activation
        = viennacl::zero_vector<float>(frontend()->size);
      for (const auto& dendrite_pair : _dendrites)
        total_activation += dendrite_pair.first->activation;

      // TODO: Generalize transfer function
      rates += dt * viennacl::linalg::element_tanh(total_activation);
    }

    void RateSynapses::prepare() {
      neurons_pre = dynamic_cast<::Backend::Vienna::RateNeurons*>
        (frontend()->neurons_pre->backend());
      weights = viennacl::zero_matrix<float>(frontend()->neurons_pre->size,
                                             frontend()->neurons_post->size);
      reset_state();
    }

    void RateSynapses::reset_state() {
      activation = viennacl::zero_vector<float>(frontend()->neurons_post->size);
    }

    void RateSynapses::push_data_front() {
      viennacl::copy(activation, frontend()->activation);
    }

    void RateSynapses::pull_data_back() {
      viennacl::copy(frontend()->activation, activation);
    }

    void RateSynapses::update_activation(float dt) {
      // TODO:: Generalise activation function
      activation = viennacl::linalg::prod(weights, neurons_pre->rates);
    }

    void RatePlasticity::prepare() {
      synapses = dynamic_cast<::Backend::Vienna::RateSynapses*>
        (frontend()->synapses->backend());
    }

    void RatePlasticity::reset_state() {
      // TODO
    }

    void RatePlasticity::push_data_front() {
      viennacl::copy(synapses->weights, frontend()->synapses->weights);
    }

    void RatePlasticity::pull_data_back() {
      viennacl::copy(frontend()->synapses->weights, synapses->weights);
    }

    void RatePlasticity::apply_plasticity(float dt) {
      // TODO
    }

    void RateElectrodes::prepare() {
      // TODO
    }

    void RateElectrodes::reset_state() {
      // TODO
    }

    void RateElectrodes::push_data_front() {
      // TODO
    }

    void RateElectrodes::pull_data_back() {
      // TODO
    }

    /*
    void RateModel::prepare() {
      // TODO
    }

    void RateModel::reset_state() {
      // TODO
    }

    void RateModel::push_data_front() {
      // TODO
    }

    void RateModel::pull_data_back() {
      // TODO
    }
    */
  }
}
