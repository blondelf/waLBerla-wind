
#pragma once

#ifndef TURBINECORE_SYNCHRONISATIONMODE_H
#define TURBINECORE_SYNCHRONISATIONMODE_H

#include <core/mpi/MPIManager.h>

namespace turbine_core {

    namespace mpi {

        enum SynchronisationMode {
            SYNC_ALL,
            SYNC_MACRO,
            SYNC_FORCE,
            SYNC_GEOMETRY,
            SYNC_CONTROL
        };

        HOST_DEVICE_PREFIX walberla::mpi::MPISize itemSize( const SynchronisationMode mode ) {
            switch (mode) {
                case SYNC_ALL:
                    printf("Not implemented.\n"); // TODO NOT IMPLEMENTED
                    return -1;
                case SYNC_MACRO:
                    return 7 * sizeof(real_t);
                case SYNC_FORCE:
                    return 3 * sizeof(real_t);
                case SYNC_GEOMETRY:
                    return 4 * 3 * sizeof(real_t) + 2 * 4 * sizeof(real_t);
                case SYNC_CONTROL:
                    printf("Not implemented.\n");
                    return -1;
                default:
                    printf("Not implemented.\n");
                    return -1;
            }
        }

    }

}

#endif //TURBINECORE_SYNCHRONISATIONMODE_H
