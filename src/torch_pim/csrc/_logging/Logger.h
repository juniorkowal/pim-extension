#pragma once

#include <iostream>

#define LOGGING

#ifdef LOGGING
#define show_info(x) std::cout << "[CPP] [INFO] " << x << std::endl
#else
#define show_info(x)
#endif
