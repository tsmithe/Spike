include_directories(BEFORE "${EIGEN_INCLUDE_DIR}")
add_library (Models OBJECT
  SpikingModel.cpp
  FourLayerVisionSpikingModel.cpp
  RateModel.cpp
  )
