#include "EventModel.hpp"

EventModel::EventModel(Context* ctx) {
  init_backend(ctx);
}

void EventModel::reset_state() {
}

EventNeurons::EventNeurons(Context* ctx, int size_, std::string label_)
  : size(size_), label(label_) {
  // TODO: This is hacky!
  if (ctx == nullptr)
    return;

  init_backend(ctx);
  // reset_state();

  if (ctx->verbose) {
    std::cout << "Spike: Created RateNeurons with size "
              << size << " and label '" << label << "'.\n";
  }
}

EventNeurons::~EventNeurons() {
}

void EventNeurons::reset_state() {
}

