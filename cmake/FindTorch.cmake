# Shim: FlagGemsConfig.cmake calls find_dependency(Torch MODULE), but PyTorch
# only ships TorchConfig.cmake (config mode). Redirect to config-mode lookup.
if(NOT TARGET Torch::Torch)
  find_package(Torch CONFIG REQUIRED)
endif()
set(Torch_FOUND TRUE)
