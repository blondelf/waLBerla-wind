#include <iostream>
#include <cmath>
#include <vector>
#include <functional>

#include <core/Environment.h>

#include "output/Logging.h"

//TODO
#include <blockforest/Initialization.h>
//ALERT
#include <blockforest/communication/NonUniformBufferedScheme.h>
#include <core/all.h>
#include <core/mpi/all.h>
#include <domain_decomposition/all.h>
#include <field/all.h>
#include <geometry/all.h>
#include <lbm_generated/field/AddToStorage.h>
//ALERT
#include <lbm_generated/communication/NonuniformGeneratedPdfPackInfo.h>

#include <lbm/all.h>
#include <lbm/vtk/all.h>
#include <timeloop/all.h>

#include "walberla_helper/field/all.h"

#include "domain/BoundarySetup.h"
#include "domain/BoundarySetter.h"
#include "domain/DomainSetup.h"
#include "domain/DomainInitialiser.h"

#include "farm_creator/WindFarm.h"
#include "farm_creator/ConfigTurbineCreator.h"

#include "wind_turbine_core/ProjectDefines.h"

#include "conversion/Conversion.h"

#include "turbine_topology/cpu/Tree.h"

#include "WindTurbineDefines.h"

#include "output/TurbineOutput.h"
#include "domain/BoundaryHandling.h"

#include "waLBerlaWind_KernelInfo.h"
#include "lbm_generated/refinement/BasicRecursiveTimeStep.h"

namespace turbine_core {

    using InterpolationKernel_T = projectors::SmoothedDiracDeltaKernel;
    using DistributionKernel_T = projectors::SmoothedDiracDeltaKernel;

    using WindFarm_T = WindFarm<topology::cpu::Tree, creator::ConfigTurbineCreator,
            ScalarField_T, VectorField_T, VectorField_T,
//            TrilinearMacroscopicFieldInterpolator,
            KernelFieldInterpolator_T<InterpolationKernel_T>,
            KernelDistributor_T<DistributionKernel_T>>;

    using TurbineOutput_T = output::TurbineOutput<
            FlagField_T, ScalarField_T, VectorField_T, SecondOrderTensorField_T, ThirdOrderTensorField_T,
            PdfField_T, Stencil_T, StorageSpecification_T::zeroCenteredPDFs, StorageSpecification_T::compressible
    >;


    int main(int argc, char** argv) {

        walberla::Environment walberlaEnv(argc, argv);

        walberla::logging::Logging::instance()->setLogLevel(walberla::logging::Logging::INFO);

        /// general simulation parameters
        auto globalConfig = walberlaEnv.config();
        auto parameters = globalConfig->getOneBlock("Parameters");

        uint_t timesteps = parameters.getParameter<uint_t>("timesteps", uint_t(10)); ++timesteps;
        const real_t remainingTimeLoggerFrequency = parameters.getParameter<real_t>("remainingTimeLoggerFrequency", real_t(3.0)); // in seconds

        const real_t length_SI = parameters.getParameter<real_t>("length_SI");
        const real_t length_LU = parameters.getParameter<real_t>("length_LU");
        const real_t velocity_SI = parameters.getParameter<real_t>("velocity_SI");
        const real_t velocity_LU = parameters.getParameter<real_t>("velocity_LU");
        const real_t viscosity_SI = parameters.getParameter<real_t>("viscosity_SI");
        const real_t density_SI = parameters.getParameter<real_t>("density_SI");

        Conversion::calculateConversionFactors(length_SI, length_LU, velocity_SI, velocity_LU, density_SI);

        Conversion::print();

        projectors::GaussianFunction5<>::setSigma(real_t(7.07106781));

        /// parameter conversion
        real_t viscosity_LU = (viscosity_SI / density_SI) * Conversion::C_t() / Conversion::C_l() / Conversion::C_l();

        WALBERLA_CHECK_GREATER(viscosity_LU, real_t(0.0), "Negative LU velocity - Did you calculate the conversion factors?")

        // calculate omega on coarsest level
        real_t omega = walberla::lbm::collision_model::omegaFromViscosity(viscosity_LU);
        WALBERLA_LOG_INFO_ON_ROOT("omega = " << omega)

        /// TURBINE

        auto windFarmConfig = globalConfig->getBlock("WindFarm");
        const std::string turbineOutputDirectory = windFarmConfig.getParameter<std::string>("outputFolder", "");
        const uint_t forceOutputStart = windFarmConfig.getParameter<uint_t>("startingTimeStepForces", 0);
        const uint_t forceOutputFrequency = windFarmConfig.getParameter<uint_t>("frequencyOutputForces", 1);

        WindFarm_T farm(turbineOutputDirectory, windFarmConfig);
        farm.createGeometry();
        farm.calculateTurbineAABBs();

        /// BLOCKFOREST

        WALBERLA_LOG_INFO_ON_ROOT("Creating blockforest...")

        auto boundariesConfig = globalConfig->getOneBlock( "Boundaries" );
        domain::BoundarySetup boundarySetup(boundariesConfig);

        if (boundarySetup.inflowType() == InflowSetup::ShiftedPeriodic && codegen::KernelInfo::inplace) {
            WALBERLA_ABORT("For the moment, shifted periodicity is not compatible with in-place streaming patterns. "
                     "Please change either inflow/outflow type or streaming pattern.")
        }

        domain::DomainSetup domainSetup ( globalConfig, boundarySetup.periodicity() );
        WALBERLA_ASSERT(!domainSetup.snapshotHandler_->loadSnapshot, "Restart is currently not supported for non-uniform grids.")

        refinement::StaticRefinementHandler<WindFarm_T> staticRefinementHandler{ globalConfig->getOneBlock("DomainSetup"), &farm, boundarySetup.environmentSetup() };
        auto blocks = domainSetup.createStructuredBlockForest(staticRefinementHandler, fieldGhostLayers);

        farm.setBlockForest(blocks);
        farm.initialiseForceAndControlModel();

        auto finestLevel = blocks->getDepth();

        omega = walberla::lbm::collision_model::levelDependentRelaxationParameter( uint_t(0), omega, finestLevel );


        /// FIELDS

        WALBERLA_LOG_INFO_ON_ROOT("Creating fields...")

        std::map<output::Fields::Types, BlockDataID> fieldMap {};

        //layout - DO NOT CHANGE
        auto layout = codegen::KernelInfo::layout;

        BlockDataID pdfFieldID;
        BlockDataID forceFieldID;
        BlockDataID meanVelocityFieldID;
        BlockDataID sumOfSquaresFieldID;

        BlockDataID nutFieldID = walberla::field::addToStorage<ScalarField_T>(blocks, "eddy viscosity field", real_t(0), layout, fieldGhostLayers);
        BlockDataID strainRateFieldID = walberla::field::addToStorage<SecondOrderTensorField_T>(blocks, "strain rate field", real_t(0), layout, fieldGhostLayers);
        BlockDataID densityFieldID = walberla::field::addToStorage<ScalarField_T>(blocks, "density field", real_t(1), layout, fieldGhostLayers);
        BlockDataID velocityFieldID = walberla::field::addToStorage<VectorField_T>(blocks, "velocity field", real_t(0), layout, fieldGhostLayers);

        BlockDataID flagFieldID = walberla::field::addFlagFieldToStorage<FlagField_T>(blocks, "flag field", fieldGhostLayers);

        const StorageSpecification_T storageSpec{};
        auto allocator = std::make_shared< walberla::field::AllocateAligned<real_t, 64> >();

        uint_t initialWelfordCounter = 0;
        if(domainSetup.snapshotHandler_->loadSnapshot) {
            auto & ssh = domainSetup.snapshotHandler_;

            auto vectorFieldDataHandling = std::make_shared<field::FlattenedFieldHandling<VectorField_T>> (blocks, fieldGhostLayers, layout, allocator);
            const auto forceFileName = std::string(std::filesystem::path(ssh->baseFolder) / std::filesystem::path(ssh->forceFile).c_str());

            forceFieldID = blocks->loadBlockData(forceFileName, vectorFieldDataHandling, "force field");

            auto pdfFieldDataHandling = std::make_shared< walberla::lbm_generated::internal::PdfFieldHandling< StorageSpecification_T > >( blocks, storageSpec, fieldGhostLayers, layout );

            const auto pdfFileName = std::string(std::filesystem::path(ssh->baseFolder) / std::filesystem::path(ssh->pdfFile).c_str());
            pdfFieldID = blocks->loadBlockData(pdfFileName, pdfFieldDataHandling, "pdf field");


            if(!ssh->meanVelFile.empty()) {
                initialWelfordCounter = ssh->welfordCounter;
                const auto meanVelFileName = std::string(std::filesystem::path(ssh->baseFolder) / std::filesystem::path(ssh->meanVelFile).c_str());

                meanVelocityFieldID = blocks->loadBlockData(meanVelFileName, vectorFieldDataHandling, "mean velocity field");

                if(!ssh->sosFile.empty()) {
                    auto tensorFieldDataHandling = std::make_shared<field::FlattenedFieldHandling<SecondOrderTensorField_T>> (blocks, fieldGhostLayers, layout, allocator);

                    const auto sosFileName = std::string(std::filesystem::path(ssh->baseFolder) / std::filesystem::path(ssh->sosFile).c_str());
                    sumOfSquaresFieldID = blocks->loadBlockData(sosFileName, tensorFieldDataHandling, "sum of squares field");
                }
            }

        } else {
            forceFieldID = walberla::field::addToStorage<VectorField_T>(blocks, "force field", real_t(0), layout, fieldGhostLayers);

            pdfFieldID = walberla::lbm_generated::addPdfFieldToStorage(blocks, "pdf field", storageSpec, fieldGhostLayers, layout);
            meanVelocityFieldID = walberla::field::addToStorage<VectorField_T>(blocks, "mean velocity field", real_t(0), layout, fieldGhostLayers);
            sumOfSquaresFieldID = walberla::field::addToStorage<SecondOrderTensorField_T>(blocks, "sum of squares field", real_t(0), layout, fieldGhostLayers);

        }

        fieldMap[output::Fields::PDF] = pdfFieldID;
        fieldMap[output::Fields::FORCE] = forceFieldID;
        fieldMap[output::Fields::DENSITY] = densityFieldID;
        fieldMap[output::Fields::VELOCITY] = velocityFieldID;
        fieldMap[output::Fields::MEAN_VELOCITY] = meanVelocityFieldID;
        fieldMap[output::Fields::STRAIN_RATE] = strainRateFieldID;
        fieldMap[output::Fields::EDDY_VISCOSITY] = nutFieldID;
        fieldMap[output::Fields::SUM_OF_SQUARES] = sumOfSquaresFieldID;

        fieldMap[output::Fields::FLAG] = flagFieldID;

        /// ABL calculations

        auto initParams = globalConfig->getOneBlock("Initialisation");
        const auto initType = domain::DomainInitialisation::toType(initParams.getParameter<std::string>("type", "uniform"));
        const Vector3<real_t> initialVelocity = initParams.getParameter<Vector3<real_t>>("initialVelocity", Vector3<real_t>(0.0));

        const real_t kappa = parameters.getParameter<real_t>("kappa", real_t(0.42));
        const real_t B = parameters.getParameter<real_t>("B", real_t(5.5));
        const real_t roughnessLengthRatio = parameters.getParameter<real_t>("roughnessLengthRatio", real_t(1e-4));
        const real_t referenceHeight = parameters.getParameter<real_t>("referenceHeight_LU", real_t(-1));
        const uint32_t samplingHeight = parameters.getParameter<uint32_t>("samplingHeight_LU", uint32_t(0));
        const real_t velocityFilterWidth = parameters.getParameter<real_t>("velocityFilterWidth");
        const real_t roughnessLength = roughnessLengthRatio * referenceHeight;

        const uint_t welfordIntervalOutput = walberlaEnv.config()->getOneBlock("Output").getParameter<uint_t>("welfordIntervalOutput", uint_t(0));

        // uTau from MOST
        const real_t uTau = kappa * initialVelocity[0] / std::log(real_t(1) / roughnessLengthRatio);
        const real_t pressure_gradient = uTau * uTau / real_t(domainSetup.domainSize_[2]);

        /// DOMAIN INITIALISATION

        WALBERLA_LOG_INFO_ON_ROOT("Initialise domain...")
        if(!domainSetup.snapshotHandler_->loadSnapshot) {

            const real_t flowDirection = atan(initialVelocity[1] / initialVelocity[0]);

            std::unique_ptr<domain::DomainInitialisation> initialiser{};
            if (initType == domain::DomainInitialisation::UNIFORM) {
                initialiser = std::make_unique<domain::UniformInitialisation>(blocks, initialVelocity);
            } else if (initType == domain::DomainInitialisation::ASMUTH) {
                initialiser = std::make_unique<domain::AsmuthInitialisation>(blocks, flowDirection, uTau, roughnessLength, kappa, B,
                                                                             viscosity_LU, domainSetup.domainSize_, true);
            } else if (initType == domain::DomainInitialisation::LOG_LAW) {
                // asmuth without perturbation velocities
                initialiser = std::make_unique<domain::AsmuthInitialisation>(blocks, uTau, flowDirection, roughnessLength, kappa, B,
                                                                             viscosity_LU, domainSetup.domainSize_, false);
            }

            using MacroSetter_T = walberla::pystencils::waLBerlaWind_MacroSetter;
            MacroSetter_T setter(densityFieldID, forceFieldID, pdfFieldID, velocityFieldID);
            initialiser->setViaVelocityField<VectorField_T, MacroSetter_T>(velocityFieldID, setter);
            initialiser->setViaVelocityField<VectorField_T, MacroSetter_T>(meanVelocityFieldID, setter);
//        initialiser->setPDFField<PdfField_T>(pdfFieldID);
        } else {
            if(domainSetup.snapshotHandler_->meanVelFile.empty()) {
                WALBERLA_ABORT("Mean velocity not loaded from snapshot but also not initialised.")
            }
            // initialise velocity field from pfd field if snapshot
            walberla::pystencils::waLBerlaWind_MacroGetter getter(densityFieldID, forceFieldID, pdfFieldID, velocityFieldID);
            for( auto & block : *blocks ) {
                getter(&block);
            }
        }

        SweepCollection_T sweepCollection( blocks, densityFieldID, forceFieldID, nutFieldID, pdfFieldID, velocityFieldID, omega );

        WALBERLA_MPI_BARRIER()
        WALBERLA_LOG_INFO_ON_ROOT("Initialisation done")

        /// BOUNDARIES

        WALBERLA_LOG_INFO_ON_ROOT("Setting up boundaries...")

        boundarySetup.fillFlagFieldFromConfig<FlagField_T>(blocks, flagFieldID, FluidFlagUID,
                                                           NoSlipFlagUID, WFBFlagUID, SymmetryFlagUID, UniformInflowFlagUID, LogLawInflowFlagUID, OutflowFlagUID);

        if(boundarySetup.wallType() == WallSetup::WFB){
            WALBERLA_CHECK(referenceHeight > real_t(0), "Reference height must be given in the parameter file for wall function boundary conditions.")
            WALBERLA_CHECK(boundarySetup.environmentSetup() == EnvironmentSetup::Open,
                           "Wall-function bounce is currently only supported for an open environment")
            // WALBERLA_CHECK(samplingHeight >= uint_t(0), "Sampling height for wall function boundary conditions must be a positive integer.")
            WALBERLA_CHECK(samplingHeight < domainSetup.cellsPerBlock_[2], "Sampling height must be smaller than block height.")
        }

        const auto& inflowVelocity = boundarySetup.inflowVelocity();
        auto velocityInit = boundary::velocityInit(roughnessLength, kappa, uTau);
        BoundaryCollection_T boundaryCollection(
                blocks, flagFieldID, pdfFieldID, FluidFlagUID, velocityFieldID,
                samplingHeight, velocityFilterWidth,  roughnessLengthRatio * referenceHeight,
                inflowVelocity[0], inflowVelocity[1], inflowVelocity[2], velocityInit
        );

        WALBERLA_CHECK(boundarySetup.inflowType() != InflowSetup::ShiftedPeriodic, "Shifted periodicity is currently not supported for non-uniform meshes.")

        /// set IDs for farm
        farm.setBlockForest(blocks);
        farm.setFieldIDs(densityFieldID, velocityFieldID, forceFieldID);

        /// WELFORD

        walberla::pystencils::waLBerlaWind_WelfordVelocity welfordSweepVelocity(meanVelocityFieldID, sumOfSquaresFieldID, velocityFieldID, 0.0);
        walberla::pystencils::waLBerlaWind_SoSResetter welfordSoSResetter(sumOfSquaresFieldID);

        /// TIMELOOP

        WALBERLA_LOG_INFO_ON_ROOT("Creating time loop...")

        // create time loop
        walberla::SweepTimeloop timeloop( blocks->getBlockStorage(), timesteps );

        farm.setTimeloop(&timeloop);

        /// flow driver
        WALBERLA_LOG_WARNING_ON_ROOT("Generated flow driver might be not adapted to mesh refinement. It is ignored for the moment.")
        walberla::wind::FlowDriverCollection flowDriver(blocks, &timeloop, globalConfig, domainSetup, forceFieldID, velocityFieldID, fieldGhostLayers);
        for( auto & block : *blocks) {
            flowDriver(&block);
        }

        //ALERT communication changed
        // create communication for PdfField
        auto communication = std::make_shared< walberla::blockforest::communication::NonUniformBufferedScheme< CommunicationStencil_T > >(blocks);
        auto packInfo = walberla::lbm_generated::setupNonuniformPdfCommunication< PdfField_T >(blocks, pdfFieldID);
        communication->addPackInfo(packInfo);

        communication->addPackInfo(std::make_shared<walberla::field::refinement::PackInfo<ScalarField_T, CommunicationStencil_T>>(densityFieldID));
        communication->addPackInfo(std::make_shared<walberla::field::refinement::PackInfo<VectorField_T, CommunicationStencil_T>>(velocityFieldID));

        uint_t maxKernelWidth = 1;
        if constexpr (std::is_same_v<WindFarm_T::Interpolator, KernelFieldInterpolator_T<InterpolationKernel_T>>) {
            maxKernelWidth = std::max({maxKernelWidth, InterpolationKernel_T::xWidth, InterpolationKernel_T::yWidth, InterpolationKernel_T::zWidth});
        }
        if constexpr (std::is_same_v<WindFarm_T::Distributor, KernelDistributor_T<DistributionKernel_T >>) {
            maxKernelWidth = std::max({maxKernelWidth, DistributionKernel_T::xWidth, DistributionKernel_T::yWidth, DistributionKernel_T::zWidth});
        }

        walberla::Config::Blocks turbineBlocks{};
        windFarmConfig.getBlocks(turbineBlocks);
        auto numTurbines = turbineBlocks.size();
        uint_t diameter = 0;
        //TODO will we ever have different models during the simulation ?
        for (auto & turbineConfig : turbineBlocks) {
            const std::string aeroModel = turbineConfig.getParameter<std::string>( "aeroModel" );
            if(aeroModel == "rotorDisk") {
                diameter = math::max(diameter, turbineConfig.getParameter<uint_t>("diameter_LU"));
            }
        }

        auto evaluateDensityAndVelocity = [&farm](walberla::IBlock * block){farm.evaluateDensityAndVelocity(block);};
        auto applyControl = [&farm](uint_t){farm.applyControl();};
        auto syncForces = [&](uint_t){farm.syncNextNeighbour(mpi::SYNC_FORCE, maxKernelWidth+uint_t(real_t(diameter)/real_t(2.0)), false);};
        auto syncMacros = [&](uint_t){farm.syncNextNeighbour(mpi::SYNC_MACRO, 1, false);};
        auto spreadForces = [&farm](walberla::IBlock * block){farm.spreadForces(block);};

        // initial communication
        communication->communicate();

        // add LBM sweep and communication to time loop
        // output
        TurbineOutput_T turbineOutput{
                blocks, &timeloop,
                globalConfig->getOneBlock("Output"), FluidFlagUID,
                fieldMap
        };

        //FIXME the current implementation expects the omega field, not the nu_t field. Strain rate writer must be called BEFORE eddy viscosity writer

        walberla::pystencils::waLBerlaWind_StrainRateWriter strainRateWriter(forceFieldID, nutFieldID, pdfFieldID, strainRateFieldID);
        turbineOutput.addBeforeFunction( [&]() {
            for( auto & block : *blocks ) {
                strainRateWriter(&block);
            }});

        walberla::pystencils::waLBerlaWind_EddyViscosityWriter eddyViscosityWriter(nutFieldID, omega);
        turbineOutput.addBeforeFunction( [&]() {
            for( auto & block : *blocks ) {
                eddyViscosityWriter(&block);
            }});

        //NOTE must convert sweeps that are altered to lambdas, otherwise copy and counter will stay 0
        welfordSweepVelocity.setCounter(real_t(initialWelfordCounter));
        auto welfordLambdaVelocity = [&welfordSweepVelocity](walberla::IBlock * block) {
            welfordSweepVelocity(block);
        };

        //ALERT
        // mesh refinement timestep
        walberla::lbm_generated::BasicRecursiveTimeStep<PdfField_T, SweepCollection_T, BoundaryCollection_T> LBMMeshRefinement (
            blocks, pdfFieldID, sweepCollection, boundaryCollection, communication, packInfo );

        //TODO add all functions here
        LBMMeshRefinement.addPostCollisionFunction([&](uint_t level) { if(level!=0) return; turbineOutput.write(); }, "VTK output");

        // snapshot
        LBMMeshRefinement.addPostCollisionFunction([&](uint_t) {
            domainSetup.snapshotHandler_->createSnapshot(timeloop.getCurrentTimeStep(), blocks, pdfFieldID,
                                                         forceFieldID,
                                                         uint_t(welfordSweepVelocity.getCounter()), meanVelocityFieldID,
                                                         sumOfSquaresFieldID);
        }, "Snapshot creation");

        // output welford handling
        LBMMeshRefinement.addPostCollisionFunction([&](uint_t) {
            if (welfordIntervalOutput // && int(welfordSweepOutput.getCounter())
                && (uint_t(welfordSweepVelocity.getCounter()) % welfordIntervalOutput == 0)) {
                welfordSweepVelocity.setCounter(real_t(1));
                // reset the field: swap pointers

                // Take care this is called only after the VTK output, otherwise we write the wrong data in the meanVelocity output
                // We live a dangerous life: we swap the velocity field counter
                for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt) {
                    auto velocityField = blockIt->getData<VectorField_T>(velocityFieldID);
                    auto meanVelocityField = blockIt->getData<VectorField_T>(meanVelocityFieldID);
                    velocityField->swapDataPointers(meanVelocityField);

                    welfordSoSResetter(blockIt.get());
                }
            } else {
                welfordSweepVelocity.setCounter(real_t(welfordSweepVelocity.getCounter() + 1));
            }
        }, "Welford Output Sweep");
        LBMMeshRefinement.addPostCollisionFunction([&](uint_t) { for (auto &block: *blocks) { welfordLambdaVelocity(&block); }}, "Welford Output Sweep");

        // turbine handling
        LBMMeshRefinement.addPostCollisionFunction([&](uint_t) { farm.callback(Component::Function::UPDATE_DISCRETISATION); }, "Rotation");
        LBMMeshRefinement.addPostCollisionFunction( [&](uint_t) {
            for (auto &block: *blocks) { evaluateDensityAndVelocity(&block); }
        }, "Evaluate density and velocity");
        LBMMeshRefinement.addPostCollisionFunction(syncMacros, "Communicate macroscopic turbine data");
        LBMMeshRefinement.addPostCollisionFunction(applyControl, "Apply Control");
        LBMMeshRefinement.addPostCollisionFunction([&](uint_t) {
            //FIXME this does not take into account force scaling during refinement ?
            for (auto &block: *blocks) {
                auto forceField = block.getData<VectorField_T>(forceFieldID);
                forceField->setWithGhostLayer(std::vector<real_t>({pressure_gradient, real_t(0.0), real_t(0.0)}));
            }
        }, "Setting driving force");
        LBMMeshRefinement.addPostCollisionFunction([&](uint_t) { farm.callback(Component::Function::CALCULATE_FORCES); }, "Calculate forces");
        LBMMeshRefinement.addPostCollisionFunction(syncForces, "Communicate force data");
        LBMMeshRefinement.addPostCollisionFunction([&](uint_t) { for (auto &block: *blocks) { spreadForces(&block); }}, "Spread forces");
        LBMMeshRefinement.addPostCollisionFunction([&](uint_t) {
            farm.callback(Component::Function::RESET_COMMUNICATION_DATA);
        }, "Reset communication data");

        //        ALERT
        LBMMeshRefinement.addRefinementToTimeLoop(timeloop);

        // LBM stability check
        timeloop.addFuncAfterTimeStep( walberla::makeSharedFunctor( walberla::field::makeStabilityChecker< PdfField_T, FlagField_T >(globalConfig, blocks, pdfFieldID, flagFieldID, FluidFlagUID ) ),
                                       "LBM stability check" );

        // log remaining time
        timeloop.addFuncAfterTimeStep( walberla::timing::RemainingTimeLogger( timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency ), "remaining time logger" );

        WALBERLA_LOG_INFO_ON_ROOT("Running timeloop...")

        WALBERLA_MPI_WORLD_BARRIER()

        walberla::WcTimingPool timing;
        timing.registerTimer("Force Output");

        walberla::WcTimer timer;
        timer.start();

        for( uint_t i = 0; i < timesteps; ++i ) {

            timeloop.singleStep(timing);

            timing["Force Output"].start();
            // farm.callback(Component::Output::ORIENTATIONS);
            for(auto & block : *blocks) {
                farm.writeForceOutput(&block, forceOutputStart, forceOutputFrequency);
            }
            farm.writeTurbinePowerAndThrust();
            timing["Force Output"].end();
        }

        timer.end();

        double time = timer.max();
        walberla::mpi::reduceInplace(time, walberla::mpi::MAX);

        const auto timeloopTiming = timing.getReduced();
        WALBERLA_LOG_INFO_ON_ROOT("Timeloop timing:\n" << *timeloopTiming)

        walberla::lbm::PerformanceEvaluation<FlagField_T> performance(blocks, flagFieldID, FluidFlagUID);
        performance.logResultOnRoot(timesteps, time);

        farm.calculatePowerAndThrust();

        logging::WindTurbineApplicationLogger<domain::DomainSetup, domain::BoundarySetup, codegen::KernelInfo, walberla::timing::WcPolicy> logger {
            domainSetup, boundarySetup, windFarmConfig, &timing, performance.loggingString(timesteps, time)
        };

        return EXIT_SUCCESS;

    }

}

int main(int argc, char** argv) {
    return turbine_core::main(argc, argv);
}
