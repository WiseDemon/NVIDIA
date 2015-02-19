/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


APEX_CUDA_STORAGE(migrationStorage)


APEX_CUDA_BOUND_KERNEL((), volumeMigrationKernel,
                       ((InplaceHandle<VolumeParamsArray>, volumeParamsArrayHandle))
                       ((InplaceHandle<ActorClassIDBitmapArray>, actorClassIDBitmapArrayHandle))
                       ((physx::PxU32, numActorClasses))((physx::PxU32, numVolumes))((physx::PxU32, numActorIDValues))
                       ((NiIofxActorID*, actorID))((physx::PxU32, maxInputID))
                       ((const float4*, positionMass))
                       ((physx::PxU32*, actorStart))((physx::PxU32*, actorEnd))((physx::PxU32*, actorVisibleEnd))
                      )
