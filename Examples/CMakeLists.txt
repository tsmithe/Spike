foreach(example
    HD_simple-path-integration.cpp
    HD_distal-only.cpp
    HD_distal-plus-proximal.cpp
    HD_proximal-only.cpp
    HD_no-place.cpp
    PLACE_proximal-only.cpp
    PLACE_distal-only.cpp
    )
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} Spike) # ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
  if (BUILD_WITH_CUDA)
    target_link_libraries(${example} ${CUDA_LIBRARIES})
    set_target_properties(${experiment} PROPERTIES COMPILE_FLAGS "-DSPIKE_WITH_CUDA")
  endif()
endforeach()
