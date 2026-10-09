
#pragma once

#ifndef TURBINECORE_OVERLOADS_H
#define TURBINECORE_OVERLOADS_H

#include "wind_turbine_core/WalberlaDataTypes.h"
#include "wind_turbine_core/ProjectDefines.h"

#include <cmath>

namespace turbine_core {

    namespace math {

        HOST_DEVICE_PREFIX FORCEINLINE
        double max(const double x, const double y) {
            return ::fmax(x, y);
        }

        HOST_DEVICE_PREFIX FORCEINLINE
        float max(const float x, const float y) {
            return ::fmaxf(x, y);
        }

        HOST_DEVICE_PREFIX FORCEINLINE
        double min(const double x, const double y) {
            return ::fmin(x, y);
        }

        HOST_DEVICE_PREFIX FORCEINLINE
        float min(const float x, const float y) {
            return ::fminf(x, y);
        }

        HOST_DEVICE_PREFIX FORCEINLINE
        uint_t max(const uint_t x, const uint_t y) {
            return x > y ? x : y;
        }

        HOST_DEVICE_PREFIX FORCEINLINE
        uint_t min(const uint_t x, const uint_t y) {
            return x > y ? y : x;
        }

        HOST_DEVICE_PREFIX FORCEINLINE
        cell_idx_t max(const cell_idx_t x, const cell_idx_t y) {
            return x > y ? x : y;
        }

        HOST_DEVICE_PREFIX FORCEINLINE
        cell_idx_t min(const cell_idx_t x, const cell_idx_t y) {
            return x > y ? y : x;
        }

    }
}


#endif //TURBINECORE_OVERLOADS_H
