#include <ATen/core/Generator.h>
#include <ATen/Context.h>

#include "torch_hpim/csrc/_logging/Logger.h"


// pseudo-random number generator ?
const at::Generator& default_generator(c10::DeviceIndex device_index) {
  return at::globalContext().defaultGenerator(at::Device(c10::DeviceType::PrivateUse1, device_index));;
}
