/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


APEX_CUDA_STORAGE(remapStorage)


APEX_CUDA_TEXTURE_1D(texRefRemapPositions,      float4)
APEX_CUDA_TEXTURE_1D(texRefRemapActorIDs,       unsigned int)
APEX_CUDA_TEXTURE_1D(texRefRemapInStateToInput, unsigned int)


APEX_CUDA_BOUND_KERNEL((), makeSortKeys,
                       ((const physx::PxU32*, inStateToInput))((physx::PxU32, maxInputID))
                       ((physx::PxU32, numActorsPerVolume))((physx::PxU32, numActorIDs))
                       ((InplaceHandle<ActorIDRemapArray>, actorIDRemapArrayHandle))
                       ((const float4*, positionMass))((bool, outputDensityKeys))
                       ((physx::PxVec3, eyePos))((physx::PxVec3, eyeDir))((physx::PxF32, zNear))
                       ((physx::PxU32*, sortKey))((physx::PxU32*, sortValue))
                      )

APEX_CUDA_BOUND_KERNEL((), remapKernel,
                       ((const physx::PxU32*, inStateToInput))((physx::PxU32, maxInputID))
                       ((physx::PxU32, numActorsPerVolume))((physx::PxU32, numActorIDs))
                       ((InplaceHandle<ActorIDRemapArray>, actorIDRemapArrayHandle))
                       ((const unsigned int*, inSortedValue))((unsigned int*, outSortKey))
                      )
