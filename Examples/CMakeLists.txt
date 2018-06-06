include_directories(BEFORE SYSTEM "${PROJECT_SOURCE_DIR}/Submodules/tinytoml/include")

foreach(example
    AHVxHD_simple_smoothAHV

    HD_simple-path-integration
    HD_inhib_simple-path-integration

    HD_distal-only
    HD_distal-only_smoothAHV
    HD_distal-only_path-integration
    HD_distal-plus-proximal

    HD_proximal-only
    HD_no-place

    PLACE_proximal-only
    PLACE_proximal-only2
    PLACE_proximal-only3
    PLACE_distal-only

    PLACE_Qpolicy_proximal
    PLACE_linear
    PLACE_square
    PLACE_compartments
    PLACE_detour
    )
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} Spike) # ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
  if (BUILD_WITH_CUDA)
    target_link_libraries(${example} ${CUDA_LIBRARIES})
    set_target_properties(${experiment} PROPERTIES COMPILE_FLAGS "-DSPIKE_WITH_CUDA")
  endif()
endforeach()
