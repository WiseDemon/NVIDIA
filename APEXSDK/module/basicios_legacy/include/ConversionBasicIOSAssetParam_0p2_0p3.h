/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __CONVERSIONBASICIOSASSETPARAM_0P2_0P3H__
#define __CONVERSIONBASICIOSASSETPARAM_0P2_0P3H__

#include "ParamConversionTemplate.h"
#include "BasicIOSAssetParam_0p2.h"
#include "BasicIOSAssetParam_0p3.h"

namespace physx
{
namespace apex
{
namespace legacy
{

typedef ParamConversionTemplate<BasicIOSAssetParam_0p2, BasicIOSAssetParam_0p3, 2, 3> ConversionBasicIOSAssetParam_0p2_0p3Parent;

class ConversionBasicIOSAssetParam_0p2_0p3: ConversionBasicIOSAssetParam_0p2_0p3Parent
{
public:
	static NxParameterized::Conversion* Create(NxParameterized::Traits* t)
	{
		void* buf = t->alloc(sizeof(ConversionBasicIOSAssetParam_0p2_0p3));
		return buf ? PX_PLACEMENT_NEW(buf, ConversionBasicIOSAssetParam_0p2_0p3)(t) : 0;
	}

protected:
	ConversionBasicIOSAssetParam_0p2_0p3(NxParameterized::Traits* t) : ConversionBasicIOSAssetParam_0p2_0p3Parent(t) {}

	bool convert()
	{
		// Inherit median from legacy asset
		mNewData->particleMass.center = mLegacyData->particleMass;

		return true;
	}
};

}
}
} //end of physx::apex:: namespace

#endif
