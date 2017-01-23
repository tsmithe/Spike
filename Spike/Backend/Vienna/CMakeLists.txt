add_definitions(-DVIENNACL_WITH_EIGEN)

if(BUILD_WITH_OPENCL)
  add_definitions(-DVIENNACL_WITH_OPENCL)
endif()
# if(BUILD_WITH_CUDA)
#   add_definitions(-DVIENNACL_WITH_CUDA)
# endif()

if(OPENMP_FOUND)
  add_definitions(-DVIENNACL_WITH_OPENMP)
endif()

add_subdirectory(Models)

