add_subdirectory(Dummy)
if(BUILD_WITH_EIGEN)
  add_subdirectory(Eigen)
endif()
if(BUILD_WITH_CUDA)
  add_subdirectory(CUDA)
endif()
if(BUILD_WITH_SYCL)
  add_subdirectory(SYCL)
endif()
if(BUILD_WITH_VIENNACL)
  add_subdirectory(Vienna)
endif()

add_library (Backend OBJECT
  Context.cpp
  )
