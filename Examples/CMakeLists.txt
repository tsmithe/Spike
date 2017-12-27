foreach(example
    HD_simple-path-integration
    HD_inhib_simple-path-integration
    HD_distal-only
    HD_distal-only_smoothAHV
    HD_distal-only_path-integration
    HD_distal-plus-proximal
    HD_proximal-only
    HD_no-place
    PLACE_proximal-only
    PLACE_distal-only
    )
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} Spike) # ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
  if (BUILD_WITH_CUDA)
    target_link_libraries(${example} ${CUDA_LIBRARIES})
    set_target_properties(${experiment} PROPERTIES COMPILE_FLAGS "-DSPIKE_WITH_CUDA")
  endif()
endforeach()
