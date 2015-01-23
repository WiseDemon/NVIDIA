/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __CONVERSIONEXPLOSIONASSETPARAM_0P0_0P1H__
#define __CONVERSIONEXPLOSIONASSETPARAM_0P0_0P1H__

#include "ParamConversionTemplate.h"
#include "ExplosionAssetParam_0p0.h"
#include "ExplosionAssetParam_0p1.h"

namespace physx
{
namespace apex
{
namespace legacy
{


typedef ParamConversionTemplate<ExplosionAssetParam_0p0, ExplosionAssetParam_0p1, 0, 1> ConversionExplosionAssetParam_0p0_0p1Parent;

class ConversionExplosionAssetParam_0p0_0p1: ConversionExplosionAssetParam_0p0_0p1Parent
{
public:
	static NxParameterized::Conversion* Create(NxParameterized::Traits* t)
	{
		void* buf = t->alloc(sizeof(ConversionExplosionAssetParam_0p0_0p1));
		return buf ? PX_PLACEMENT_NEW(buf, ConversionExplosionAssetParam_0p0_0p1)(t) : 0;
	}

protected:
	ConversionExplosionAssetParam_0p0_0p1(NxParameterized::Traits* t) : ConversionExplosionAssetParam_0p0_0p1Parent(t) {}

	bool convert()
	{
		//Non-uniform stuff
		mNewData->nonUniformParams.fieldIntensity = mLegacyData->fieldIntensity;
		mNewData->nonUniformParams.distanceTarget = mLegacyData->distanceTarget;
		mNewData->nonUniformParams.distanceScale = mLegacyData->distanceScale;
		mNewData->nonUniformParams.velocityTarget = mLegacyData->velocityTarget;
		mNewData->nonUniformParams.velocityScale = mLegacyData->velocityScale;
		mNewData->nonUniformParams.disAttenuation = mLegacyData->disAttenuation;
		mNewData->nonUniformParams.degreeOfNoise = mLegacyData->degreeOfNoise;

		//Shockwave parameters
		mNewData->shockwaveParams.width = mLegacyData->shockwaveWidth;
		mNewData->shockwaveParams.travelVelocity = mLegacyData->travelVelocity;

		//mode is enum now

		static const physx::PxU32 NX_APEX_EPM_EXPLOSION = 1,
		                          NX_APEX_EPM_IMPLOSION = 2,
		                          NX_APEX_EPM_SHOCKWAVE = 3;

		switch (mLegacyData->mode)
		{
		case NX_APEX_EPM_EXPLOSION:
			mNewData->mode = "explosion";
			break;
		case NX_APEX_EPM_IMPLOSION:
			mNewData->mode = "implosion";
			break;
		case NX_APEX_EPM_SHOCKWAVE:
			mNewData->mode = "shockwave";
			break;
		default:
			return false;
		}

		return true;
	}
};

}
}
} //end of physx::apex:: namespace

#endif
