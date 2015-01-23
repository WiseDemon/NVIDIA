/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "NxApexDefs.h"
#include "MinPhysxSdkVersion.h"
#include "NxRenderMeshActorDesc.h"
#include "NxRenderMeshActor.h"
#include "NxRenderMeshAsset.h"

#if NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED

#include "NxApex.h"

#include "BasicFSActor.h"
#include "BasicFSAsset.h"
#include "BasicFSScene.h"
#include "NiApexSDK.h"
#include "NiApexScene.h"
#include "NiApexRenderDebug.h"

#if NX_SDK_VERSION_MAJOR == 2
#include <NxScene.h>
#include "NxFromPx.h"
#elif NX_SDK_VERSION_MAJOR == 3
#include <PxScene.h>
#endif

#include <NiFieldSamplerManager.h>
#include "ApexResourceHelper.h"

namespace physx
{
namespace apex
{
namespace basicfs
{

#define NUM_DEBUG_POINTS 2048

BasicFSActor::BasicFSActor(BasicFSScene& scene)
	: mScene(&scene)
	, mPose(true)
	, mScale(1.0f)
	, mFieldSamplerChanged(true)
	, mFieldSamplerEnabled(true)
	, mFieldWeight(1.0f)
{
}

BasicFSActor::~BasicFSActor()
{
}

#if NX_SDK_VERSION_MAJOR == 2
void BasicFSActor::setPhysXScene(NxScene*) { }
NxScene* BasicFSActor::getPhysXScene() const
{
	return NULL;
}
#elif NX_SDK_VERSION_MAJOR == 3
void BasicFSActor::setPhysXScene(PxScene*) { }
PxScene* BasicFSActor::getPhysXScene() const
{
	return NULL;
}
#endif

}
}
} // end namespace physx::apex

#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
