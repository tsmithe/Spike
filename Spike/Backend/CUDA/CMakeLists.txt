if(BUILD_SHARED_LIBS)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
endif()

add_subdirectory(Helpers)
add_subdirectory(Models)
add_subdirectory(Neurons)
add_subdirectory(RecordingElectrodes)
add_subdirectory(SpikeAnalyser)
add_subdirectory(STDP)
add_subdirectory(Synapses)

