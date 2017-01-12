#pragma once
#include <cassert>
#include <iostream>

#ifndef NDEBUG
#include <cxxabi.h>
// From http://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c#comment63837522_81870 :
#define TYPEID_NAME(x) abi::__cxa_demangle(typeid((x)).name(), NULL, NULL, NULL)
#endif

#define SPIKE_ADD_BACKEND_GETSET(TYPE, SUPER)           \
  void backend(std::shared_ptr<Backend::TYPE>&& ptr) {  \
    _backend = ptr;                                     \
    SUPER::backend(ptr);                                \
  }                                                     \
  inline Backend::TYPE* backend() const {               \
    assert(_backend != nullptr &&                       \
           "Need to have backend initialized!");        \
    return (Backend::TYPE*)_backend.get();              \
  }                                                     \
  inline void prepare_backend() {                       \
    prepare_backend_early();                            \
    backend()->prepare();                               \
    prepare_backend_late();                             \
  }

#define SPIKE_MAKE_BACKEND_CONSTRUCTOR(TYPE)            \
  TYPE(::TYPE* front) {                                 \
    _frontend = (void*)front;                           \
    std::cout << "TODO@@@ " << TYPEID_NAME(this) << " @ " << this << " with front " << _frontend << "\n"; \
  }

#define SPIKE_ADD_FRONTEND_GETTER(TYPE)                 \
  inline ::TYPE* frontend() const {                     \
    assert(_frontend != nullptr &&                      \
           "Need to have backend initialized!");        \
    return (::TYPE*)_frontend;                          \
  }

#ifdef SPIKE_WITH_CUDA
#define SPIKE_MAKE_INIT_BACKEND(TYPE)                                   \
  void TYPE::init_backend(Context* ctx) {                               \
    ::Backend::TYPE* ptr = nullptr;                                     \
    switch (ctx->device) {                                              \
    case Backend::SPIKE_DEVICE_DUMMY:                                   \
      ptr = new Backend::Dummy::TYPE(this);                             \
      break;                                                            \
    case Backend::SPIKE_DEVICE_VIENNA:                                  \
      ptr = new Backend::Vienna::TYPE(this);                            \
      break;                                                            \
    case Backend::SPIKE_DEVICE_CUDA:                                    \
      ptr = new Backend::CUDA::TYPE(this);                              \
      break;                                                            \
    default:                                                            \
      assert("Unsupported backend" && false);                           \
    };                                                                  \
    backend(std::shared_ptr<::Backend::TYPE>(ptr));                     \
    backend()->context = ctx;                                           \
    prepare_backend();                                                  \
  }
#else
#define SPIKE_MAKE_INIT_BACKEND(TYPE)                                   \
  void TYPE::init_backend(Context* ctx) {                               \
    switch (ctx->device) {                                              \
    case Backend::SPIKE_DEVICE_DUMMY:                                   \
      backend(std::make_shared<Backend::Dummy::TYPE>(this));            \
      break;                                                            \
    case Backend::SPIKE_DEVICE_VIENNA:                                  \
      backend(std::make_shared<Backend::Vienna::TYPE>(this));           \
      break;                                                            \
    default:                                                            \
      assert("Unsupported backend" && false);                           \
    };                                                                  \
    backend()->context = ctx;                                           \
    std::cout << "TODO??? " << backend() << ": " << dynamic_cast<::Backend::Vienna::TYPE*>(backend())->frontend() << "\n"; \
    backend()->_frontend = (void*)this;                                   \
    std::cout << "TODO??? " << backend() << ": " << dynamic_cast<::Backend::Vienna::TYPE*>(backend())->frontend() << "\n"; \
    prepare_backend();                                                  \
  }
#endif

#define SPIKE_MAKE_STUB_INIT_BACKEND(TYPE)                             \
  void TYPE::init_backend(Context* ctx) {                              \
    assert("This type's backend cannot be instantiated!" && false);    \
  }
