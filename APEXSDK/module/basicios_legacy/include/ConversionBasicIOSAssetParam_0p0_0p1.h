/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __CONVERSIONBASICIOSASSETPARAMS_0P0_0P1H__
#define __CONVERSIONBASICIOSASSETPARAMS_0P0_0P1H__

#include "ParamConversionTemplate.h"
#include "BasicIOSAssetParam_0p0.h"
#include "BasicIOSAssetParam_0p1.h"

namespace physx
{
namespace apex
{
namespace legacy
{

typedef ParamConversionTemplate<BasicIOSAssetParam_0p0, BasicIOSAssetParam_0p1, 0, 1> ConversionBasicIOSAssetParam_0p0_0p1Parent;

class ConversionBasicIOSAssetParam_0p0_0p1: ConversionBasicIOSAssetParam_0p0_0p1Parent
{
public:
	static NxParameterized::Conversion* Create(NxParameterized::Traits* t)
	{
		void* buf = t->alloc(sizeof(ConversionBasicIOSAssetParam_0p0_0p1));
		return buf ? PX_PLACEMENT_NEW(buf, ConversionBasicIOSAssetParam_0p0_0p1)(t) : 0;
	}

protected:
	ConversionBasicIOSAssetParam_0p0_0p1(NxParameterized::Traits* t) : ConversionBasicIOSAssetParam_0p0_0p1Parent(t) {}

	bool convert()
	{
		// just take the default values (scaleSceneGravity = 1.0, externalAcceleration=0,0,0)
		return true;
	}
};

}
}
} //end of physx::apex:: namespace

#endif
