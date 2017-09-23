#include "RateBase.hpp"

BufferWriter global_writer;

namespace Eigen {
std::mt19937 global_random_generator;
}

SpikeException::SpikeException(std::string msg) : _msg(msg) {}
const char* SpikeException::what() const noexcept {
  return _msg.c_str();
}

BufferWriter::~BufferWriter() {
  if (running)
    stop();
  if (othread.joinable())
    othread.join();
  // file.close();
}

void BufferWriter::add_buffer(EigenBuffer* buf) {
  buffers.push_back(buf);
}

void BufferWriter::write_output() {
  for (auto buffer : buffers) {
    while (buffer->size() > 0) {
      auto& front = buffer->front();
      // int timestep = front.first; // TODO: perhaps write this out, too?

      auto data = front.second.data();
      int n_bytes = front.second.size() * sizeof(decltype(front.second)::Scalar);

      buffer->file.write((char*) data, n_bytes);

      buffer->pop_front();
    }
  }
}

void BufferWriter::block_until_empty() const {
  for (auto buffer : buffers) {
    while (buffer->size() > 0)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void BufferWriter::write_loop() {
  while (running) {
    // TODO: why 200ms?
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    write_output();
    time_since_last_flush += 0.2;
    if (time_since_last_flush > 10) {
      for (auto buffer : buffers) {
        buffer->file.flush();
      }
      time_since_last_flush = 0;
    }
  }
}

void BufferWriter::start() {
  if (running)
    return;

  running = true;
  othread = std::thread(&BufferWriter::write_loop, this);
}

void BufferWriter::stop() {
  if (!running)
    return;

  running = false;
  if (othread.joinable())
    othread.join();

  for (auto buffer : buffers) {
    buffer->file.flush();
  }
}

