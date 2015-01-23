/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef NX_MODULE_BASIC_IOS_H
#define NX_MODULE_BASIC_IOS_H

#include "NxApex.h"
#include "NxTestBase.h"
#include <limits.h>

namespace physx
{
namespace apex
{

PX_PUSH_PACK_DEFAULT

class NxApexScene;
class NxBasicIosAsset;
class NxBasicIosActor;
class NxBasicIosAssetAuthoring;

/**
\brief BasicIOS Module
*/
class NxModuleBasicIos : public NxModule
{
protected:
	virtual											~NxModuleBasicIos() {}

public:
	/// Get BasicIOS authoring type name
	virtual const char*								getBasicIosTypeName() = 0;

	/// Get NxTestBase implementation of BasicIos scene
	virtual const NxTestBase*						getTestBase(NxApexScene* apexScene) const = 0;

};


PX_POP_PACK

}
} // namespace physx::apex

#endif // NX_MODULE_BASIC_IOS_H
