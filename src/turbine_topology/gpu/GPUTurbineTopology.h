#pragma once

#ifndef TURBINECORE_GPUTURBINETOPOLOGY_H
#define TURBINECORE_GPUTURBINETOPOLOGY_H

#include <iostream>

#include <domain_decomposition/IBlock.h>
#include <gpu/GPUField.h>
#include <gpu/ParallelStreams.h>
//#include <gpu/NVTX.h>
#include <gpu/GPURAII.h>
#include <core/mpi/MPIWrapper.h>
#include <core/mpi/Reduce.h>
#include <core/mpi/BufferSystem.h>

#include "wind_turbine_core/ProjectDefines.h"
#include "wind_turbine_core/WalberlaDataTypes.h"

#include "component/ComponentType.h"
#include "component/Component.h"

#include "domain/TurbineDomain.h"

#include "discretisation/gpu/CopyToGPU.h"
#include "force_model/gpu/CopyToGPU.h"
#include "control_model/gpu/CopyToGPU.h"

#include "mpi/CustomMemoryBuffer.h"
#include "mpi/GPUMemoryAllocator.h"
#include "mpi/CPUMemoryAllocator.h"

#include "walberla_helper/field/Field.h"
#include "walberla_helper/field/Projectors.h"

#include "wind_turbine_core/math/Vector3.h"
#include "discretisation/Disk.h"
#include "conversion/Conversion.h"

#include "GPUTurbineTopology_GlobalFunctions.h"

namespace turbine_core {

    namespace topology {

        namespace gpu {

            template< typename T >
            class GPUTurbineTopology {

            public:

                using HostBuffer_T = mpi::CustomMemoryBuffer<mpi::GPUHostMemoryAllocator>;
                using DeviceBuffer_T = mpi::CustomMemoryBuffer<mpi::GPUDeviceMemoryAllocator>;
                using ElementType_T = HostBuffer_T::ElementType;
                using CPUBufferSystem_T = walberla::mpi::GenericBufferSystem<HostBuffer_T, HostBuffer_T>;
                using GPUBufferSystem_T = walberla::mpi::GenericBufferSystem<DeviceBuffer_T, DeviceBuffer_T>;
                using ReceiverInfo_T = std::map<walberla::mpi::MPIRank, walberla::mpi::MPISize>;

                static constexpr bool isGPU = true;

                HOST_DEVICE_PREFIX GPUTurbineTopology()
                {}
                HOST_DEVICE_PREFIX GPUTurbineTopology(const GPUTurbineTopology & ) = delete;
                HOST_DEVICE_PREFIX ~GPUTurbineTopology() {}

                HOST_PREFIX void callback( const Component::Function & function, const ComponentType & type, const uint_t timestep = 0 ) {

                    globalCallbackFct<<<1,1>>>(d_this, function, type, timestep);
                    TURBINE_GPU_CHECK()

                }

                HOST_PREFIX void callback( const Component::Output & function, const std::filesystem::path & baseFolder,
                                           const uint_t timestep, const ComponentType & ) {

                    // host code below reads managed memory (internal::nComponents, nPoints); on
                    // pre-Pascal GPUs this requires an idle device
                    WALBERLA_GPU_CHECK(gpuDeviceSynchronize())

                    if(!turbineCharacteristicsSet_) {
                        getTurbineCharacteristics();
                    }

                    WALBERLA_ROOT_SECTION() {

                        uint_t totalPoints = 0;
                        for (uint_t c = 0; c < internal::nComponents; ++c) {
                            totalPoints += internal::nPoints[c];
                        }

                        uint_t nDataItems = 0;
                        switch (function) {
                            case Component::Output::GNUPLOT :
                                nDataItems = 1;
                                break;
                            case Component::Output::ORIENTATIONS :
                                nDataItems = 4;
                                break;
                            default:
                                break;
                        }

                        std::vector<Vector3<real_t>> h_vectors(totalPoints * nDataItems);

                        Vector3<real_t> * d_vectors;
                        gpuMalloc((void **) &d_vectors, totalPoints * nDataItems * sizeof(Vector3<real_t>));
                        TURBINE_GPU_CHECK()
                        getOutputDataOnGPU<<<1, 1>>>(d_this, function, d_vectors);
                        TURBINE_GPU_CHECK()

                        gpuMemcpy(h_vectors.data(), d_vectors, totalPoints * nDataItems * sizeof(Vector3<real_t>),
                                   gpuMemcpyDeviceToHost);
                        gpuFree(d_vectors);
                        TURBINE_GPU_CHECK()

                        std::string filename;
                        switch (function) {
                            case Component::Output::GNUPLOT :
                                filename = "Gnuplot_" + std::to_string(timestep) + ".txt";
                                break;
                            case Component::Output::ORIENTATIONS :
                                filename = "Orientations_" + std::to_string(timestep) + ".txt";
                                break;
                            case Component::Output::FORCES : WALBERLA_ABORT("Not implemented.")
                        }

                        auto filepath = baseFolder / filename;
                        std::ofstream os(filepath, std::ios::app);

                        if (!os.is_open()) WALBERLA_ABORT("Could not open file " << filepath.string() << ".")

                        uint_t currentIdx = 0;

                        for (uint_t c = 0; c < internal::nComponents; ++c) {
                            for (uint_t p = 0; p < internal::nPoints[c]; ++p) {

                                // do not print every value to avoid cluttering of plot
                                if ((p % 3 != 0) && p != internal::nPoints[c] - 1)
                                    continue;

                                const uint_t idx = (currentIdx + p) * nDataItems;
                                auto & position = h_vectors[idx];

                                switch (function) {
                                    case Component::Output::GNUPLOT :
                                        os << position[0] << "\t" << position[1] << "\t" << position[2] << "\n";
                                        break;
                                    case Component::Output::ORIENTATIONS :
                                        os << position[0] << "\t" << position[1] << "\t" << position[2] << "\t"
                                           << h_vectors[idx + 1][0] << "\t" << h_vectors[idx + 1][1] << "\t"
                                           << h_vectors[idx + 1][2] << "\tx\n";
                                        os << position[0] << "\t" << position[1] << "\t" << position[2] << "\t"
                                           << h_vectors[idx + 2][0] << "\t" << h_vectors[idx + 2][1] << "\t"
                                           << h_vectors[idx + 2][2] << "\ty\n";
                                        os << position[0] << "\t" << position[1] << "\t" << position[2] << "\t"
                                           << h_vectors[idx + 3][0] << "\t" << h_vectors[idx + 3][1] << "\t"
                                           << h_vectors[idx + 3][2] << "\tz\n";
                                        break;
                                    default :
                                        break;
                                }

                            } // loop points

                            // empty lines between components
                            os << "\n\n";

                            currentIdx += internal::nPoints[c];
                        } // loop components

                    }
                }

                HOST_PREFIX void getControlNeeds( bool & h_needsMeanVelocity, bool & h_needsTorque, bool & h_needsWindVane ) {

                    // KLAUSSY: this is not functionnal, use uint_t instead
                    bool *d_needsMeanVelocity, *d_needsTorque, *d_needsWindVane;

                    gpuMalloc((void**)&d_needsMeanVelocity, sizeof(bool));
                    gpuMalloc((void**)&d_needsTorque, sizeof(bool));
                    gpuMalloc((void**)&d_needsWindVane, sizeof(bool));
                    TURBINE_GPU_CHECK()

                    getControlNeedsOnGPU<<<1,1>>>(d_this, d_needsMeanVelocity, d_needsTorque, d_needsWindVane);
                    TURBINE_GPU_CHECK()

                    gpuMemcpy(&h_needsMeanVelocity, d_needsMeanVelocity, sizeof(bool), gpuMemcpyDeviceToHost);
                    gpuMemcpy(&h_needsTorque, d_needsTorque, sizeof(bool), gpuMemcpyDeviceToHost);
                    gpuMemcpy(&h_needsWindVane, d_needsWindVane, sizeof(bool), gpuMemcpyDeviceToHost);
                    TURBINE_GPU_CHECK()

                    gpuFree(d_needsMeanVelocity);
                    gpuFree(d_needsTorque);
                    gpuFree(d_needsWindVane);
                    TURBINE_GPU_CHECK()
                }

                HOST_PREFIX void applyControl( uint_t timeStep, real_t meanVelocity,
                                               real_t rotorTorque, real_t rotorOmega, Vector3<real_t> rotorWindVaneWind) {
                    applyControlOnGPU<T><<<1,1>>>(d_this, timeStep, meanVelocity, rotorTorque, rotorOmega, rotorWindVaneWind);
                    TURBINE_GPU_CHECK()
                }

                template< typename DensityField_T, typename VelocityField_T, typename Interpolator_T >
                HOST_PREFIX void evaluateDensityAndVelocity( walberla::IBlock * block, std::shared_ptr<walberla::StructuredBlockForest> & storage,
                                                             const BlockDataID &densityFieldID,
                                                             const BlockDataID &velocityFieldID,
                                                             const uint_t &nPointsComponents,
                                                             const uint_t &nComponents) {


                    int threadsPerBlock = nPointsComponents; //npoints
                    int numBlocks = nComponents + 1;  // number of components

                    blockforest::BlockInfo blockInfo{block, storage};

                    // get fields
                    field::Field<real_t> densityField(block->getData<walberla::gpu::GPUField<real_t>>(densityFieldID));
                    field::Field<real_t> velocityField(block->getData<walberla::gpu::GPUField<real_t>>(velocityFieldID));
                    evaluateDensityAndVelocityOnGPU<T, Interpolator_T><<<numBlocks, threadsPerBlock, 0, turbineStream_>>>(d_this, blockInfo, densityField, velocityField, nPointsComponents);
                    TURBINE_GPU_CHECK()

                    WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
                }

                HOST_PREFIX void syncNextNeighbour(
                        const domain::TurbineDomain & domain,
                        const mpi::SynchronisationMode mode,
                        const uint_t impactWidth,
                        const int turbineID,
                        const bool ) {

                    // NOTE: for the moment, using GPU direct for the turbine communication slows down the code
                    // significantly. Until the source of this issue is found and fixed, the gpuDirect option will
                    // be disabled.
                    constexpr bool gpuDirect = false;

                    // communication not needed for one MPI process
                    if (walberla::mpi::MPIManager::instance()->numProcesses() == 1)
                        return;

                    // current process does not contain turbine blocks
                    if (domain.getNumLocalAABBs() == 0)
                        return;

                    // get receiver information for all processes
                    const auto & neighbourProcesses = domain.getNeighborProcesses();

                    // TODO this is calculated every time even if turbine has not moved
                    const auto turbineAABB = this->getAABB();

                    bool needCommunication = false;
                    for (uint_t nbProcessRank: neighbourProcesses) {
                        if(domain.intersectsWithProcessSubdomain(nbProcessRank, turbineAABB))
                            needCommunication = true;
                    }
                    if(!needCommunication)
                        return;

                    if (!turbineCharacteristicsSet_) {
                        getTurbineCharacteristics();
                    }

                    walberla::gpu::ParallelStreams parallelSectionManager;
                    auto stream = walberla::gpu::StreamRAII::newPriorityStream(0);

                    // add two points for the rotor disk case
                    //TODO: instead, here we can use get_nPoints (discretisations) + nComponent * 1 (controllers) or directly have a "do_getNControlledComponents" ?
                    const auto send_recvPointMax = internal::nBlades * (internal::nBladePoints + 1) + 2;

                    ReceiverInfo_T receiverInfo;

                    if (!isCommunicationSetUp_) {

                        const auto &localAABBs = domain.localAABBs();
                        nLocalAABBs_ = localAABBs.size();

                        gpuMalloc((void **) &d_localAABBs_, nLocalAABBs_ * sizeof(math::AABB));
                        gpuMemcpy(d_localAABBs_, localAABBs.data(), nLocalAABBs_ * sizeof(math::AABB), gpuMemcpyHostToDevice);
                        TURBINE_GPU_CHECK()
                    }

                    CPUBufferSystem_T bufferSystemCPU (walberla::mpi::MPIManager::instance()->comm(), turbineID);
                    GPUBufferSystem_T bufferSystemGPU (walberla::mpi::MPIManager::instance()->comm(), turbineID);

                    const uint_t ownRank = walberla::mpi::MPIManager::instance()->rank();

                    for (uint_t nbProcessRank: neighbourProcesses) {

                        const auto intRank = static_cast<int>(nbProcessRank);

                        auto &sendBufferCPU = bufferSystemCPU.sendBuffer(nbProcessRank);
                        auto &sendBufferGPU = bufferSystemGPU.sendBuffer(nbProcessRank);

                        // fill empty buffer with dummy byte to force transmission
                        {
                            receiverInfo[intRank] = walberla::mpi::MPISize(sizeof(walberla::uint8_t));

                            auto bufferPtr = sendBufferGPU.advance(1);
                            WALBERLA_ASSERT_NOT_NULLPTR(bufferPtr)
                            gpuMemset(bufferPtr, ElementType_T(1), 1);

                            if (!gpuDirect) {
                                auto bufferPtr = sendBufferCPU.advance(1);
                                WALBERLA_ASSERT_NOT_NULLPTR(bufferPtr);
                                memset(bufferPtr, ElementType_T(1), 1);
                            }
                        }

                        const auto neighborAABB = domain.neighborAABBs(nbProcessRank);
                        const auto nNeighborAABBs = neighborAABB.size();

                        // first time to communicate something
                        if (!isCommunicationSetUp_) {

                            gpuMalloc((void **) &d_neighborAABBs_[intRank], nNeighborAABBs * sizeof(math::AABB));
                            gpuMemcpy(d_neighborAABBs_[intRank], neighborAABB.data(),
                                       nNeighborAABBs * sizeof(math::AABB), gpuMemcpyHostToDevice);
                            TURBINE_GPU_CHECK()

                            //NOTE for this allocation, send_recvPointMax of the first turbine is used, even if other turbines have a different number of points
                            gpuMalloc((void **) &d_sendPoints_[intRank], send_recvPointMax * sizeof(int));
                            gpuMalloc((void **) &d_recvPoints_[intRank], send_recvPointMax * sizeof(int));
                            TURBINE_GPU_CHECK()

                        }

                        // reset send and receive point buffers
                        gpuMemset(d_sendPoints_[intRank], -1, send_recvPointMax * sizeof(int));
                        gpuMemset(d_recvPoints_[intRank], -1, send_recvPointMax * sizeof(int));
                        TURBINE_GPU_CHECK()

                        uint_t h_nSendPoints;
                        uint_t *d_nSendPoints;
                        gpuMalloc(&d_nSendPoints, sizeof(uint_t));

                        walberla::mpi::MPISize *d_receiverSize;
                        gpuMalloc(&d_receiverSize, sizeof(walberla::mpi::MPISize));
                        TURBINE_GPU_CHECK()

                        setupCommunicationOnGPU<<<1, 1>>>(
                                d_this,
                                ownRank, nbProcessRank,                                 // information about ranks
                                d_sendPoints_[intRank],
                                d_recvPoints_[intRank],         // storage for points to send and receive
                                nLocalAABBs_,
                                nNeighborAABBs,                           // number of local and neighbor domains
                                d_localAABBs_,
                                d_neighborAABBs_[intRank],               // aabb's of local and neighbor domains
                                d_nSendPoints, d_receiverSize,                          // further information needed
                                mode, impactWidth
                        );
                        TURBINE_GPU_CHECK()

                        gpuMemcpy(&h_nSendPoints, d_nSendPoints, sizeof(uint_t), gpuMemcpyDeviceToHost);
                        gpuFree(d_nSendPoints);
                        TURBINE_GPU_CHECK()
                        gpuMemcpy(&receiverInfo[intRank], d_receiverSize, sizeof(walberla::mpi::MPISize),
                                   gpuMemcpyDeviceToHost);
                        gpuFree(d_receiverSize);
                        TURBINE_GPU_CHECK()

                        // account for dummy transfer in buffer that was overwritten
                        receiverInfo[intRank] += walberla::mpi::MPISize(sizeof(ElementType_T));

                        sendBufferCPU.resize(h_nSendPoints * uint_t(mpi::itemSize(mode)) +
                                             walberla::mpi::MPISize(sizeof(ElementType_T)));
                        sendBufferGPU.resize(h_nSendPoints * uint_t(mpi::itemSize(mode)) +
                                             walberla::mpi::MPISize(sizeof(ElementType_T)));

                    } // loop neighbourProcesses

                    if (!isCommunicationSetUp_) {
                        isCommunicationSetUp_ = true;
                    }

                    // set receiver info
                    if (gpuDirect) {
                        bufferSystemGPU.setReceiverInfo(receiverInfo);
                    } else {
                        bufferSystemCPU.setReceiverInfo(receiverInfo);
                    }

                    // pack buffers
                    {
                        auto parallelSection = parallelSectionManager.parallelSection(stream);

                        for (auto nbProcessRank: neighbourProcesses) {

                            const auto intRank = static_cast<int>(nbProcessRank);

                            parallelSection.run([&](auto s) {

                                // fill GPU buffer with GPU data
                                auto &sendBuffer = bufferSystemGPU.sendBuffer(nbProcessRank);
                                WALBERLA_ASSERT(!sendBuffer.isEmpty())

                                auto *pt = d_sendPoints_[intRank];

                                packTurbineDataOnGPU<<<1, 1, 0, s>>>(d_this, sendBuffer.cur(), pt, mode);
                                TURBINE_GPU_CHECK()

                                if (gpuDirect) {
                                    sendBuffer.advanceNoResize(sendBuffer.allocSize() - sendBuffer.size());
                                } else {

                                    auto &cpuBuffer = bufferSystemCPU.sendBuffer(nbProcessRank);
                                    WALBERLA_ASSERT(!cpuBuffer.isEmpty())

                                    auto *gpuPtr = sendBuffer.ptr();
                                    auto *cpuPtr = cpuBuffer.ptr();

                                    gpuMemcpyAsync(cpuPtr, gpuPtr, sendBuffer.allocSize(), gpuMemcpyDeviceToHost, s);
                                    TURBINE_GPU_CHECK()
                                    cpuBuffer.advanceNoResize(cpuBuffer.allocSize() - cpuBuffer.size());

                                }
                            });

                        } // loop neighborProcess
                    }

                    // wait for packing to finish
                    gpuStreamSynchronize( stream );

                    // send Buffers
                    if (gpuDirect) {
                        bufferSystemGPU.sendAll();
                    } else {
                        bufferSystemCPU.sendAll();
                    }

                    {
                        // unpack buffers
                        auto parallelSection = parallelSectionManager.parallelSection( stream );

                        if (gpuDirect) {

                            for (auto recv = bufferSystemGPU.begin(); recv != bufferSystemGPU.end(); ++recv) {

                                parallelSection.run([&](auto s) {

                                    recv.buffer().clear();
                                    const auto intRank = static_cast<int>(recv.rank());

                                    auto *tmp = recv.buffer().advanceNoResize(1);

                                    auto *pt = d_recvPoints_[intRank];
                                    unpackTurbineDataOnGPU<<<1, 1, 0, s>>>(d_this, recv.buffer().cur(), pt, mode);
                                    TURBINE_GPU_CHECK()

                                });

                            }

                        } else {

                            for (auto recv = bufferSystemCPU.begin(); recv != bufferSystemCPU.end(); ++recv) {

                                const auto intRank = static_cast<int>(recv.rank());

                                parallelSection.run([&](auto s) {

                                    auto cpuBuffer = recv.buffer();
                                    cpuBuffer.clear();

                                    // remove dummy buffer
                                    [[maybe_unused]] ElementType_T tmp;
                                    tmp = *(recv.buffer().advanceNoResize(1));

                                    // allocate GPU memory
                                    ElementType_T *d_buffer;
                                    const auto size = receiverInfo[intRank] - sizeof(ElementType_T);
                                    gpuMalloc((void **) &d_buffer, size);
                                    TURBINE_GPU_CHECK()

                                    // copy to GPU memory
                                    gpuMemcpyAsync(d_buffer, recv.buffer().cur(), size, gpuMemcpyHostToDevice, s);
                                    TURBINE_GPU_CHECK()

                                    auto *pt = d_recvPoints_[intRank];
                                    unpackTurbineDataOnGPU<<<1, 1, 0, s>>>(d_this, d_buffer, pt, mode);
                                    TURBINE_GPU_CHECK()

                                    gpuFree(d_buffer);
                                    TURBINE_GPU_CHECK()

                                });
                            }

                        }

                    }

                }

                template< typename ForceField_T, typename ForceDistributor_T >
                HOST_PREFIX void spreadForces(walberla::IBlock * block, std::shared_ptr<walberla::StructuredBlockForest> & storage,
                                              const BlockDataID &forceFieldID,
                                              const uint_t &nPointsComponentsForSpread) {

                    gpuDeviceProp prop;
                    gpuGetDeviceProperties(&prop, 0);
                    //int threadsPerBlock = prop.maxThreadsPerBlock;
                    int threadsPerBlock = 64;
                    int numBlocks = 1;

                    if (nPointsComponentsForSpread < threadsPerBlock)
                        numBlocks = 1;
                    else
                        numBlocks = (nPointsComponentsForSpread + threadsPerBlock - 1) / threadsPerBlock;

                    blockforest::BlockInfo blockInfo{block, storage};

                    /// GET INTERPOLATORS AND DISTRIBUTOR
                    field::Field<real_t> forceField(block->getData<walberla::gpu::GPUField<real_t>>(forceFieldID));

                    spreadForcesOnGPU<T, ForceDistributor_T><<<numBlocks, threadsPerBlock, 0, turbineStream_>>>(d_this, blockInfo, forceField, nPointsComponentsForSpread);
                    TURBINE_GPU_CHECK()
                }

                template<typename Discretisation_T>
                HOST_PREFIX void addDiscretisation(const std::string & parentID, const std::string & componentID,
                                                   const ComponentType & type,
                                                   const std::shared_ptr<Discretisation_T> & discretisation )
                {
                    /// copy component strings to device

                    char * d_parentID;      const size_t parentBytes    = (   parentID.size()+1) * sizeof(char);
                    char * d_componentID;   const size_t componentBytes = (componentID.size()+1) * sizeof(char);

                    gpuMalloc((void**)&d_parentID, parentBytes);
                    gpuMemcpy(d_parentID, parentID.c_str(), parentBytes, gpuMemcpyHostToDevice);

                    gpuMalloc((void**)&d_componentID, componentBytes);
                    gpuMemcpy(d_componentID, componentID.c_str(), componentBytes, gpuMemcpyHostToDevice);

                    /// copy discretisation

                    discretisation::gpu::CopyToGPU<Discretisation_T> copyDToGpu;
                    copy(copyDToGpu, discretisation);

                    TURBINE_GPU_CHECK()

                    /// call kernel

                    gpu::addGPUDiscretisation<Discretisation_T><<<1,1>>>(
                            d_this, d_parentID, d_componentID, type, copyDToGpu.devicePtr);
                    TURBINE_GPU_CHECK()

                    gpuFree(d_componentID);
                    gpuFree(d_parentID);
                    TURBINE_GPU_CHECK()

                }

                template<typename ForceModel_T, typename ControlModel_T>
                HOST_PREFIX void initialiseForceAndControlModel(const std::string & componentID,
                                                                const std::shared_ptr<ForceModel_T> & forceModel,
                                                                const std::shared_ptr<ControlModel_T> & controlModel)
                {
                    /// copy component strings to device
                    char * d_componentID;   const size_t componentBytes = (componentID.size()+1) * sizeof(char);

                    gpuMalloc((void**)&d_componentID, componentBytes);
                    gpuMemcpy(d_componentID, componentID.c_str(), componentBytes, gpuMemcpyHostToDevice);

                    /// copy force model

                    TURBINE_GPU_CHECK()

                    /// call kernel

                    //NOTE ugly but segfault for pointer in global function otherwise
                    if(forceModel){

                        force_model::gpu::CopyToGPU<ForceModel_T> copyFMToGpu;
                        copy(copyFMToGpu, forceModel);

                        if(controlModel) {
                            control_model::gpu::CopyToGPU<ControlModel_T> copyCMToGpu;
                            copy(copyCMToGpu, controlModel);
                            isControlled_ = true;
                            gpu::initialiseGPUForceAndControlModel<ForceModel_T, ControlModel_T><<<1,1>>>(
                                    d_this, d_componentID, true, copyFMToGpu.devicePtr, true, copyCMToGpu.devicePtr);
                        } else {
                            gpu::initialiseGPUForceAndControlModel<ForceModel_T, ControlModel_T><<<1,1>>>(
                                    d_this, d_componentID, true, copyFMToGpu.devicePtr, false, nullptr);
                        }
                    } else {
                        if(controlModel) {
                            control_model::gpu::CopyToGPU<ControlModel_T> copyCMToGpu;
                            copy(copyCMToGpu, controlModel);
                            isControlled_ = true;
                            gpu::initialiseGPUForceAndControlModel<ForceModel_T, ControlModel_T><<<1,1>>>(
                                    d_this, d_componentID, false, nullptr, true, copyCMToGpu.devicePtr);
                        } else {
                            gpu::initialiseGPUForceAndControlModel<ForceModel_T, ControlModel_T><<<1,1>>>(
                                    d_this, d_componentID, false, nullptr, false, nullptr);
                        }
                    }

                    TURBINE_GPU_CHECK()

                    gpuFree(d_componentID);
                    TURBINE_GPU_CHECK()

                }

                HOST_DEVICE_PREFIX void print() const {
                    static_cast<const T&>(*this).do_print();
                }

                HOST_PREFIX void setupEnvironment(const std::string & baseName, const Vector3<real_t> & origin) {

                    if(d_this != nullptr) {
                        printf("Cannot re-init device pointer to base point.");
                        exit(-1);
                    }

                    WALBERLA_GPU_CHECK(gpuDeviceGetStreamPriorityRange(&streamLowPriority_, &streamHighPriority_));
                    WALBERLA_GPU_CHECK(gpuStreamCreateWithPriority(&turbineStream_, gpuStreamDefault, streamHighPriority_));
//FIXME                    walberla::gpu::nameStream(turbineStream_, "turbine stream " + baseName);

                    gpuMalloc((void**) &d_this, sizeof(T*));

                    char * d_baseName;
                    gpuMalloc((void**)&d_baseName, (baseName.length()+1) * sizeof(char));
                    gpuMemcpy(d_baseName, baseName.c_str(), baseName.length()+1 * sizeof(char), gpuMemcpyHostToDevice);
                    TURBINE_GPU_CHECK()

                    setupOnGPU<<<1,1>>>(d_this, d_baseName, origin);
                    TURBINE_GPU_CHECK()

                    gpuFree(d_baseName);
                    TURBINE_GPU_CHECK()

                }

                HOST_PREFIX void synchroniseGpuStreams() {
                    WALBERLA_GPU_CHECK(gpuStreamSynchronize(turbineStream_))
//                    WALBERLA_GPU_CHECK(gpuStreamWaitEvent(turbineStream_, spreadForcesEvent_))
                }

                HOST_PREFIX void deleteEnvironment() const {
                    gpu::deleteOnGPU<<<1,1>>>(d_this);
                    WALBERLA_GPU_CHECK(gpuDeviceSynchronize())
                    gpuFree(d_this);

                    if(turbineCharacteristicsSet_) {
                        gpuFree(internal::nPoints);
                    }

                    if(isCommunicationSetUp_) {
                        gpuFree(d_localAABBs_);
                        for(auto it : d_neighborAABBs_) {
                            gpuFree(it.second);
                        }

                        for( auto it : d_sendPoints_ ) {
                            gpuFree(it.second);
                        }

                        for( auto it : d_recvPoints_ ) {
                            gpuFree(it.second);
                        }

                    }
                    WALBERLA_GPU_CHECK(gpuStreamDestroy(turbineStream_));
                }

                //! Drop the cached communication setup (process-local and neighbor AABBs, send/receive
                //! point buffers). Must be called whenever the domain decomposition changes, e.g. after
                //! a blockforest refresh during adaptive mesh refinement; the next syncNextNeighbour()
                //! rebuilds the buffers from the new decomposition.
                HOST_PREFIX void resetCommunication() {

                    if(!isCommunicationSetUp_)
                        return;

                    gpuFree(d_localAABBs_);
                    d_localAABBs_ = nullptr;
                    nLocalAABBs_ = uint_t(0);

                    for(auto & it : d_neighborAABBs_) { gpuFree(it.second); }
                    d_neighborAABBs_.clear();

                    for(auto & it : d_sendPoints_) { gpuFree(it.second); }
                    d_sendPoints_.clear();

                    for(auto & it : d_recvPoints_) { gpuFree(it.second); }
                    d_recvPoints_.clear();

                    TURBINE_GPU_CHECK()

                    isCommunicationSetUp_ = false;
                }

                HOST_PREFIX void writeForceOutput(walberla::IBlock * block, std::shared_ptr<walberla::StructuredBlockForest> & storage,
                                                  std::filesystem::path & baseFolder, const uint_t timestep) {

                    // host code below reads managed memory (internal::nBlades, ...); on pre-Pascal
                    // GPUs this requires an idle device
                    WALBERLA_GPU_CHECK(gpuDeviceSynchronize())

                    if(!turbineCharacteristicsSet_) {
                        getTurbineCharacteristics();
                    }

                    // block-local points, i.e., points with meaningful data
                    uint_t * d_localPoints{};
                    uint_t * d_nLocalPoints{};

                    // device data
                    real_t * d_azimuth;
                    real_t * d_relVelocity;
                    Vector2<real_t> * d_localForces;
                    Vector3<real_t> * d_localAero;

                    // malloc space on CPU and GPU
                    gpuMalloc((void**)&d_nLocalPoints, sizeof(uint_t));
                    TURBINE_GPU_CHECK()
                    gpuMalloc((void**)&d_localPoints, internal::nBlades * internal::nBladePoints * sizeof(uint_t));
                    gpuMalloc((void**)&d_azimuth, internal::nBlades * internal::nBladePoints * sizeof(real_t));
                    gpuMalloc((void**)&d_localForces, internal::nBlades * internal::nBladePoints * sizeof(Vector2<real_t>));
                    gpuMalloc((void**)&d_localAero, internal::nBlades * internal::nBladePoints * sizeof(Vector3<real_t>));
                    gpuMalloc((void**)&d_relVelocity, internal::nBlades * internal::nBladePoints * sizeof(real_t));
                    TURBINE_GPU_CHECK()

                    blockforest::BlockInfo blockInfo{block, storage};

                    // fill GPU data
                    gpu::getForceDataOnGPU<<<1,1>>>(d_this, blockInfo, d_nLocalPoints, d_localPoints, d_azimuth, d_localForces, d_localAero, d_relVelocity);
                    TURBINE_GPU_CHECK()

                    // get number of points to be written
                    uint_t h_nLocalPoints{};
                    gpuMemcpy( &h_nLocalPoints, d_nLocalPoints, sizeof(uint_t), gpuMemcpyDeviceToHost );
                    gpuFree(d_nLocalPoints);
                    TURBINE_GPU_CHECK()

                    WALBERLA_ASSERT_LESS_EQUAL(h_nLocalPoints, internal::nBlades * internal::nBladePoints,
                                               "Number of local points exceeds number of total points in turbine.")

                    // allocate necessary data
                    std::vector<uint_t> h_localPoints(h_nLocalPoints);
                    std::vector<real_t> h_azimuth(h_nLocalPoints);
                    std::vector<real_t> h_relVelocity(h_nLocalPoints);
                    std::vector<Vector2<real_t>> h_localForces(h_nLocalPoints);
                    std::vector<Vector3<real_t>> h_localAero(h_nLocalPoints);

                    // copy data to host
                    gpuMemcpy( h_localPoints.data(), d_localPoints, h_nLocalPoints * sizeof(uint_t), gpuMemcpyDeviceToHost );
                    gpuFree(d_localPoints);
                    TURBINE_GPU_CHECK()

                    gpuMemcpy( h_azimuth.data(), d_azimuth, h_nLocalPoints * sizeof(real_t), gpuMemcpyDeviceToHost );
                    gpuFree(d_azimuth);
                    TURBINE_GPU_CHECK()

                    gpuMemcpy( h_relVelocity.data(), d_relVelocity, h_nLocalPoints * sizeof(real_t), gpuMemcpyDeviceToHost );
                    gpuFree(d_relVelocity);
                    TURBINE_GPU_CHECK()

                    gpuMemcpy( h_localForces.data(), d_localForces, h_nLocalPoints * sizeof(Vector2<real_t>), gpuMemcpyDeviceToHost );
                    gpuFree(d_localForces);
                    TURBINE_GPU_CHECK()

                    gpuMemcpy( h_localAero.data(), d_localAero, h_nLocalPoints * sizeof(Vector3<real_t>), gpuMemcpyDeviceToHost );
                    gpuFree(d_localAero);
                    TURBINE_GPU_CHECK()

                    // conversion factors
                    const real_t Cf = Conversion::C_m() / ( Conversion::C_t() * Conversion::C_t() );
                    const real_t Cu = Conversion::C_l() / Conversion::C_t();

                    for (uint_t b = 0; b < internal::nBlades; ++b) {

                        auto bladeFolder =
                                baseFolder / std::filesystem::path(std::string("Blade_") + std::to_string(b));

                        for (uint_t i = 0; i < h_nLocalPoints; ++i) {

                            auto p = h_localPoints[i] - b * internal::nBladePoints;

                            // does not belong to current blade
                            if(p >= internal::nBladePoints)
                                continue;

                            const std::string pString{std::to_string(p)};

                            const std::string filename{
                                    "Point_" + std::string(3 - pString.length(), '0') + pString + ".txt"};

                            auto filepath = bladeFolder / filename;

                            std::ofstream os(filepath.string(), std::ios::app);

                            if (os.is_open()) {
                                os << timestep << "\t";
                                os << h_azimuth[i] << "\t";

                                auto & localForces = h_localForces[i];

                                os << localForces[0] * Cf << "\t" << localForces[1] * Cf << "\t";

                                auto & localAero = h_localAero[i];

                                os << localAero[0] / walberla::math::pi * real_t(180) << "\t" << localAero[1] << "\t" << localAero[2] << "\t";

                                os << h_relVelocity[i] * Cu;

                                os << "\n";

                                os.close();
                            } else {
                                WALBERLA_ABORT("Could not open file " + filepath.string())
                            }

                        } // loop points

                    } // loop blades

                }

                HOST_PREFIX void calculateMeanVelocity( walberla::IBlock * block, std::shared_ptr<walberla::StructuredBlockForest> & storage,
                                                        real_t & h_meanVelocity, uint_t & h_nPoints ) const {

                    uint_t *d_nPoints;
                    real_t *d_meanVelocity;

                    gpuMalloc((void**)&d_nPoints, sizeof(uint_t));
                    gpuMalloc((void**)&d_meanVelocity, sizeof(real_t));
                    TURBINE_GPU_CHECK()

                    blockforest::BlockInfo blockInfo{block, storage};

                    calculateMeanVelocityOnGPU<<<1,1>>>(d_this, blockInfo, d_meanVelocity, d_nPoints);
                    TURBINE_GPU_CHECK()

                    gpuMemcpy(&h_meanVelocity, d_meanVelocity, sizeof(real_t), gpuMemcpyDeviceToHost);
                    gpuMemcpy(&h_nPoints, d_nPoints, sizeof(uint_t), gpuMemcpyDeviceToHost);
                    TURBINE_GPU_CHECK()

                    gpuFree(d_nPoints);
                    gpuFree(d_meanVelocity);
                    TURBINE_GPU_CHECK()

                }

                HOST_PREFIX void calculateWindVaneVelocity( walberla::IBlock * block, std::shared_ptr<walberla::StructuredBlockForest> & storage,
                                                        Vector3<real_t> & h_vaneVelocity, uint_t & h_nPoints ) const {

                    uint_t *d_nPoints;
                    Vector3<real_t> *d_vaneVelocity;

                    gpuMalloc((void**)&d_nPoints, sizeof(uint_t));
                    gpuMalloc((void**)&d_vaneVelocity, sizeof(Vector3<real_t>));
                    TURBINE_GPU_CHECK()

                    blockforest::BlockInfo blockInfo{block, storage};

                    calculateWindVaneVelocityOnGPU<<<1,1>>>(d_this, blockInfo, d_vaneVelocity, d_nPoints);
                    TURBINE_GPU_CHECK()

                    gpuMemcpy(&h_vaneVelocity, d_vaneVelocity, sizeof(real_t), gpuMemcpyDeviceToHost);
                    gpuMemcpy(&h_nPoints, d_nPoints, sizeof(uint_t), gpuMemcpyDeviceToHost);
                    TURBINE_GPU_CHECK()

                    gpuFree(d_nPoints);
                    gpuFree(d_vaneVelocity);
                    TURBINE_GPU_CHECK()

                }

                HOST_PREFIX void calculatePowerAndThrust(walberla::IBlock * block, std::shared_ptr<walberla::StructuredBlockForest> & storage,
                                                         real_t & h_power, real_t & h_thrust) const {


                    real_t *d_power, *d_thrust;

                    gpuMalloc((void**)&d_power, sizeof(real_t));
                    gpuMalloc((void**)&d_thrust, sizeof(real_t));
                    TURBINE_GPU_CHECK()

                    blockforest::BlockInfo blockInfo{block, storage};

                    calculatePowerAndThrustOnGPU<<<1,1>>>(d_this, blockInfo, d_power, d_thrust);
                    TURBINE_GPU_CHECK()

                    gpuMemcpy(&h_power, d_power, sizeof(real_t), gpuMemcpyDeviceToHost);
                    gpuMemcpy(&h_thrust, d_thrust, sizeof(real_t), gpuMemcpyDeviceToHost);
                    TURBINE_GPU_CHECK()

                    gpuFree(d_power);
                    gpuFree(d_thrust);
                    TURBINE_GPU_CHECK()

                    // do unit conversion
                    const real_t Cf = Conversion::C_m() * Conversion::C_l() / ( Conversion::C_t() * Conversion::C_t() );
                    const real_t thrustFactor = -Cf;
                    const real_t powerFactor = +Cf * Conversion::C_l() / Conversion::C_t();

                    h_power *= powerFactor;
                    h_thrust *= thrustFactor;

                }

                HOST_PREFIX void calculateTorque(walberla::IBlock * block, std::shared_ptr<walberla::StructuredBlockForest> & storage,
                                                         real_t & h_torque) const {


                    real_t *d_torque;

                    gpuMalloc((void**)&d_torque, sizeof(real_t));
                    TURBINE_GPU_CHECK()

                    blockforest::BlockInfo blockInfo{block, storage};

                    calculateTorqueOnGPU<<<1,1>>>(d_this, blockInfo, d_torque);
                    TURBINE_GPU_CHECK()

                    gpuMemcpy(&h_torque, d_torque, sizeof(real_t), gpuMemcpyDeviceToHost);
                    TURBINE_GPU_CHECK()

                    gpuFree(d_torque);
                    TURBINE_GPU_CHECK()
                }

                HOST_PREFIX void getOmega(real_t & h_omega) const {


                    real_t *d_omega;

                    gpuMalloc((void**)&d_omega, sizeof(real_t));
                    TURBINE_GPU_CHECK()

                    getOmegaOnGPU<<<1,1>>>(d_this, d_omega);
                    TURBINE_GPU_CHECK()

                    gpuMemcpy(&h_omega, d_omega, sizeof(real_t), gpuMemcpyDeviceToHost);
                    TURBINE_GPU_CHECK()

                    gpuFree(d_omega);
                    TURBINE_GPU_CHECK()
                }

                HOST_PREFIX AABB getAABB() const {

                    const uint_t n = 6;

                    // stores bladeLength, hubRadius, towerHeight, hubPosition[0], hubPosition[1], hubPosition[2]
                    std::vector<real_t> h_info(n, 0);
                    real_t * d_info;

                    gpuMalloc((void**)&d_info, n * sizeof(real_t));
                    TURBINE_GPU_CHECK()

                    getAABBInformationOnGPU<<<1,1>>>( d_this, d_info );
                    TURBINE_GPU_CHECK()

                    gpuMemcpy(h_info.data(), d_info, n * sizeof(real_t), gpuMemcpyDeviceToHost);
                    gpuFree(d_info);
                    TURBINE_GPU_CHECK()

                    const real_t bladeLength{h_info[0]};
                    const real_t hubRadius{h_info[1]};
                    const real_t towerHeight{h_info[2]};
                    const Vector3<real_t> hubPosition{h_info[3], h_info[4], h_info[5]};

                    Vector3<real_t> halfWidth{bladeLength + hubRadius + towerHeight};

                    const auto min = hubPosition - halfWidth;
                    const auto max = hubPosition + halfWidth;

                    return {min, max};

                }

                HOST_PREFIX bool isControlled() const {return isControlled_;}
                HOST_PREFIX void setControlled(const bool isControlled) {isControlled_ = isControlled;}

                T ** d_this{nullptr};

            private:

                HOST_PREFIX void getTurbineCharacteristics() {
                    // on pre-Pascal GPUs (no concurrentManagedAccess) the host must not allocate or
                    // touch managed memory while kernels are in flight, so wait for the GPU first
                    WALBERLA_GPU_CHECK(gpuDeviceSynchronize())

                    //TODO what if more than 15 components?
                    gpuMallocManaged((void **) &internal::nPoints, 15 * sizeof(uint_t));
                    getTurbineCharacteristicsOnGPU<<<1, 1>>>(d_this, internal::nPoints);
                    TURBINE_GPU_CHECK()

                    gpu::getBladeDataOnGPU<<<1,1>>>(d_this);
                    gpuDeviceSynchronize(); // synchronise as kernels are asynchronous and we have to wait for the correct 'results'
                    TURBINE_GPU_CHECK()
                    turbineCharacteristicsSet_ = true;
                }

                int streamHighPriority_ = 0;
                int streamLowPriority_ = 0;

                gpuStream_t turbineStream_;

                bool turbineCharacteristicsSet_{false};
                bool isControlled_{false};

                bool isCommunicationSetUp_{false};

                uint_t nLocalAABBs_{};
                math::AABB * d_localAABBs_{nullptr};

                std::map<walberla::mpi::MPIRank, math::AABB*> d_neighborAABBs_{};

                std::map<walberla::mpi::MPIRank, int*> d_sendPoints_{};
                std::map<walberla::mpi::MPIRank, int*> d_recvPoints_{};

            };

        } // namespace gpu

    } // namespace data_structure

} // namespace turbine_core

#endif //TURBINECORE_GPUTURBINETOPOLOGY_H
