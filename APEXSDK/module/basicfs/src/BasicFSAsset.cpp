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
#if NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED

#include "NxApex.h"

#include "BasicFSAsset.h"
#include "BasicFSActor.h"
#include "ModuleBasicFS.h"
//#include "ApexSharedSerialization.h"
#include "BasicFSScene.h"

namespace physx
{
namespace apex
{
namespace basicfs
{


BasicFSAsset::BasicFSAsset(ModuleBasicFS* module, const char* name)
			: mModule(module)
			, mName(name)
{
}

BasicFSAsset::~BasicFSAsset()
{
}

}
}
} // end namespace physx::apex


#endif // NX_SDK_VERSION_NUMBER >= MIN_PHYSX_SDK_VERSION_REQUIRED
