/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef CONVERSIONTURBULENCEFSASSETPARAMS_1P0_1P1H_H
#define CONVERSIONTURBULENCEFSASSETPARAMS_1P0_1P1H_H

#include "ParamConversionTemplate.h"
#include "TurbulenceFSAssetParams_1p0.h"
#include "TurbulenceFSAssetParams_1p1.h"

namespace physx
{
namespace apex
{

typedef ParamConversionTemplate<TurbulenceFSAssetParams_1p0, TurbulenceFSAssetParams_1p1, 65536, 65537> ConversionTurbulenceFSAssetParams_1p0_1p1Parent;

class ConversionTurbulenceFSAssetParams_1p0_1p1: ConversionTurbulenceFSAssetParams_1p0_1p1Parent
{
public:
	static NxParameterized::Conversion* Create(NxParameterized::Traits* t)
	{
		void* buf = t->alloc(sizeof(ConversionTurbulenceFSAssetParams_1p0_1p1));
		return buf ? PX_PLACEMENT_NEW(buf, ConversionTurbulenceFSAssetParams_1p0_1p1)(t) : 0;
	}

protected:
	ConversionTurbulenceFSAssetParams_1p0_1p1(NxParameterized::Traits* t) : ConversionTurbulenceFSAssetParams_1p0_1p1Parent(t) {}

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

		return true;
	}
};

}
} // namespace physx::apex

#endif
