#pragma once

#include "Spike/RecordingElectrodes/NetworkStateArchiveRecordingElectrodes.hpp"
#include "RecordingElectrodes.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class NetworkStateArchiveRecordingElectrodes :
      public virtual ::Backend::CUDA::RecordingElectrodesCommon,
      public ::Backend::NetworkStateArchiveRecordingElectrodes {
    public:
      virtual void reset_state() {
        // TODO (if necessary...)
      }

      virtual void copy_state_to_front(::NetworkStateArchiveRecordingElectrodes* front);
    };
  }
}