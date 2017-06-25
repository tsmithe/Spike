add_library(SpikeSYCLModels STATIC
  EventModel.cpp
  )
if(COMPUTECPP_PACKAGE_ROOT_DIR)
  add_sycl_to_target(
    EventModel.cpp ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/EventModel.cpp
    )
endif()
