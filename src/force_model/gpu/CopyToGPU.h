
#pragma once

#ifndef TURBINECORE_FORCEMODEL_COPYTOGPU_H
#define TURBINECORE_FORCEMODEL_COPYTOGPU_H

#include <cassert>
#include <vector>

#include "wind_turbine_core/ProjectDefines.h"

#include "point3/ActuatorData.h"

namespace turbine_core {

    namespace force_model {

        class ForceModel;
        class None;
        template< typename DensityAndVelocityInterpolator_T, typename ForceDistributor_T, typename TipLossModel_T >
        class ActuatorLineModel;
        template< typename DensityAndVelocityInterpolator_T, typename ForceDistributor_T >
        class ActuatorDiskModel;

        class TipLossModel;

        namespace gpu {

            GLOBAL_PREFIX void deleteForceModelOnGPU( ForceModel ** gpuPtr ) {
                delete (*gpuPtr);
            }

            template<class ForceModel_T>
            struct CopyToGPU {

                HOST_PREFIX CopyToGPU() {

                    gpuMalloc((void**)&devicePtr, sizeof(ForceModel_T*));
                    gpuErrchk(gpuPeekAtLastError())
                    gpuErrchk(gpuDeviceSynchronize())

                }

                HOST_PREFIX ~CopyToGPU() {
                    deleteForceModelOnGPU<<<1,1>>>((ForceModel**)devicePtr);
                    gpuErrchk(gpuPeekAtLastError())
                    gpuErrchk(gpuDeviceSynchronize())
                    gpuFree(devicePtr);
                }

                ForceModel_T ** devicePtr{nullptr};

            };

            // template prototypes
            template<typename ForceModel_T>
            HOST_PREFIX void copy( CopyToGPU<ForceModel_T> &, const std::shared_ptr<ForceModel_T> & );

            template<typename DensityAndVelocityInterpolator_T, typename ForceDistributor_T, template<typename...> class ForceModel_T>
            HOST_PREFIX void copy( CopyToGPU<ForceModel_T<DensityAndVelocityInterpolator_T,ForceDistributor_T>> &,
                                   const std::shared_ptr<ForceModel_T<DensityAndVelocityInterpolator_T,ForceDistributor_T>> & );


            // kernel

            template< typename DensityAndVelocityInterpolator_T, typename ForceDistributor_T, typename TipLossModel_T >
            GLOBAL_PREFIX void newActuatorLineModelOnGPU( ActuatorLineModel<DensityAndVelocityInterpolator_T,ForceDistributor_T,TipLossModel_T> ** gpuPtr, const real_t elementLength,
                                                          const uint_t nPoints, ActuatorData * _data, TipLossModel_T* tipLossModel)
            {
                //TODO might want to think of something cleverer!
                auto data = (ActuatorData*) malloc(nPoints * sizeof(ActuatorData));
                for(uint_t i = 0; i < nPoints; ++i) {
                    new (static_cast<ActuatorData*>(data) + i) ActuatorData(_data[i].width, _data[i].polar);
                }

                (*gpuPtr) = new ActuatorLineModel<DensityAndVelocityInterpolator_T, ForceDistributor_T, TipLossModel_T>(elementLength, nPoints, data, *tipLossModel);

                for(uint_t i = 0; i < nPoints; ++i) {
                    (static_cast<ActuatorData*>(data) + i)->~ActuatorData();
                }
                free(data);
            }

            template< typename DensityAndVelocityInterpolator_T, typename ForceDistributor_T >
            GLOBAL_PREFIX void newActuatorDiskModelOnGPU( ActuatorDiskModel<DensityAndVelocityInterpolator_T,ForceDistributor_T> ** gpuPtr, const real_t radius, const real_t dragCoefficient )
            {
                (*gpuPtr) = new ActuatorDiskModel<DensityAndVelocityInterpolator_T,ForceDistributor_T>(radius, dragCoefficient);
            }

            // template specialisation
            template<> HOST_PREFIX void copy( CopyToGPU<ForceModel>& cpyToGPU, const std::shared_ptr<ForceModel> & cpuPtr ) {
                WALBERLA_ABORT("Abstract base class cannot be copied.")
            }

            // template specialisations

             template< typename DensityAndVelocityInterpolator_T, typename ForceDistributor_T, typename TipLossModel_T >
             HOST_PREFIX void copy( CopyToGPU<ActuatorLineModel<DensityAndVelocityInterpolator_T,ForceDistributor_T,TipLossModel_T>>& cpyToGPU,
                                    const std::shared_ptr<ActuatorLineModel<DensityAndVelocityInterpolator_T,ForceDistributor_T,TipLossModel_T>> & cpuPtr ) {

                 uint_t nPoints = cpuPtr->nPoints_;

                 // copy actuator data
                 ActuatorData * d_actuatorData;
                 gpuMalloc((void**)&d_actuatorData, nPoints * sizeof(ActuatorData) );
                 gpuMemcpy(d_actuatorData, cpuPtr->points_, nPoints * sizeof(ActuatorData), gpuMemcpyHostToDevice);

                 // copy tip loss model
                 TipLossModel_T * d_tipLossModel;
                 gpuMalloc((void**)&d_tipLossModel, sizeof(TipLossModel_T) );
                 gpuMemcpy(d_tipLossModel, &cpuPtr->tiplossModel_, sizeof(TipLossModel_T), gpuMemcpyHostToDevice);

                 newActuatorLineModelOnGPU<<<1,1>>>(cpyToGPU.devicePtr, cpuPtr->elementLength(), nPoints, d_actuatorData, d_tipLossModel);
                 gpuErrchk(gpuPeekAtLastError())
                 gpuErrchk(gpuDeviceSynchronize())

                 gpuFree(d_actuatorData);

                 gpuFree(d_tipLossModel);

                 gpuErrchk(gpuPeekAtLastError())
                 gpuErrchk(gpuDeviceSynchronize())
             }


            template< typename DensityAndVelocityInterpolator_T, typename ForceDistributor_T >
            HOST_PREFIX void copy( CopyToGPU<ActuatorDiskModel<DensityAndVelocityInterpolator_T,ForceDistributor_T>>& cpyToGPU,
                                   const std::shared_ptr<ActuatorDiskModel<DensityAndVelocityInterpolator_T,ForceDistributor_T>> & cpuPtr ) {

                // copy actuator data
                // ActuatorData * d_actuatorData;
                // gpuMalloc((void**)&d_actuatorData, nPoints * sizeof(ActuatorData) );
                // gpuMemcpy(d_actuatorData, cpuPtr->point_, nPoints * sizeof(ActuatorData), gpuMemcpyHostToDevice);

                newActuatorDiskModelOnGPU<<<1,1>>>(cpyToGPU.devicePtr, cpuPtr->radius_, cpuPtr->drag_);
                gpuErrchk(gpuPeekAtLastError())
                gpuErrchk(gpuDeviceSynchronize())

                // gpuFree(d_actuatorData);

            }
        }

    }

}

#endif //TURBINECORE_FORCEMODEL_COPYTOGPU_H
