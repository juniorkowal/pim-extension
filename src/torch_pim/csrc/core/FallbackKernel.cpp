#include <ATen/native/CPUFallback.h>

#include "src/torch_pim/csrc/_logging/Logger.h"


void pim_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    show_info("UPMEM fallback: Operator '" << op.schema().operator_name() << "' not supported. Switching to CPU.");
    at::native::cpu_fallback(op, stack);
}


TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&pim_cpu_fallback>());
}

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m){ // because we only do inference
    m.fallback(torch::CppFunction::makeFallthrough());
}
