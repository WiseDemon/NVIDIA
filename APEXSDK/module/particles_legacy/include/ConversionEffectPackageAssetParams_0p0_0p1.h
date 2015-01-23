/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef CONVERSIONEFFECTPACKAGEASSETPARAMS_0P0_0P1H_H
#define CONVERSIONEFFECTPACKAGEASSETPARAMS_0P0_0P1H_H

#include "ParamConversionTemplate.h"
#include "EffectPackageAssetParams_0p0.h"
#include "EffectPackageAssetParams_0p1.h"

namespace physx
{
namespace apex
{

typedef ParamConversionTemplate<EffectPackageAssetParams_0p0, EffectPackageAssetParams_0p1, 0, 1> ConversionEffectPackageAssetParams_0p0_0p1Parent;

class ConversionEffectPackageAssetParams_0p0_0p1: ConversionEffectPackageAssetParams_0p0_0p1Parent
{
public:
	static NxParameterized::Conversion* Create(NxParameterized::Traits* t)
	{
		void* buf = t->alloc(sizeof(ConversionEffectPackageAssetParams_0p0_0p1));
		return buf ? PX_PLACEMENT_NEW(buf, ConversionEffectPackageAssetParams_0p0_0p1)(t) : 0;
	}

protected:
	ConversionEffectPackageAssetParams_0p0_0p1(NxParameterized::Traits* t) : ConversionEffectPackageAssetParams_0p0_0p1Parent(t) {}

	const NxParameterized::PrefVer* getPreferredVersions() const
	{
		static NxParameterized::PrefVer prefVers[] =
		{
			//TODO:
			//	Add your preferred versions for included references here.
			//	Entry format is
			//		{ (const char*)longName, (PxU32)preferredVersion }

			{ 0, 0 } // Terminator (do not remove!)
		};

		return prefVers;
	}

	bool convert()
	{
		//TODO:
		//	Write custom conversion code here using mNewData and mLegacyData members.
		//
		//	Note that
		//		- mNewData has already been initialized with default values
		//		- same-named/same-typed members have already been copied
		//			from mLegacyData to mNewData
		//		- included references were moved to mNewData
		//			(and updated to preferred versions according to getPreferredVersions)
		//
		//	For more info see the versioning wiki.
		if (mNewData->LODSettings.FadeDistanceBegin > mNewData->LODSettings.FadeDistanceEnd)
		{
			mNewData->LODSettings.FadeDistanceBegin = mNewData->LODSettings.FadeDistanceEnd;
		}
		return true;
	}
};

}
} // namespace physx::apex

#endif
