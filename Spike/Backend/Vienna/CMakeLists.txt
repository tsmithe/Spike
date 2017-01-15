add_definitions(-DVIENNACL_WITH_EIGEN)
if(BUILD_WITH_OPENCL)
  add_definitions(-DVIENNACL_WITH_OPENCL)
endif()
# if(BUILD_WITH_CUDA)
#   add_definitions(-DVIENNACL_WITH_CUDA)
# endif()

add_subdirectory(Models)

