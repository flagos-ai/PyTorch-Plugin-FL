# Shim: FlagGemsConfig.cmake calls find_dependency(Torch MODULE), but PyTorch
# only ships TorchConfig.cmake (config mode). Since find_package(Torch) was already
# called in the parent CMakeLists.txt, just verify the target exists and mark as found.
if(NOT TARGET torch)
  message(FATAL_ERROR "FindTorch.cmake shim: torch target not found. Ensure find_package(Torch) was called before FlagGems.")
endif()

# Set variables expected by find_dependency
set(Torch_FOUND TRUE)
set(Torch_VERSION "${TORCH_VERSION}")
set(Torch_DIR "${Torch_DIR}")

# Ensure Torch::Torch alias exists (some PyTorch versions only create 'torch')
if(NOT TARGET Torch::Torch)
  add_library(Torch::Torch ALIAS torch)
endif()
