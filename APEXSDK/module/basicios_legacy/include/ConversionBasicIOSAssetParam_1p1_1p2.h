/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef CONVERSIONBASICIOSASSETPARAM_1P1_1P2H_H
#define CONVERSIONBASICIOSASSETPARAM_1P1_1P2H_H

#include "ParamConversionTemplate.h"
#include "BasicIOSAssetParam_1p1.h"
#include "BasicIOSAssetParam_1p2.h"

namespace physx
{
namespace apex
{

typedef ParamConversionTemplate<BasicIOSAssetParam_1p1, BasicIOSAssetParam_1p2, 65537, 65538> ConversionBasicIOSAssetParam_1p1_1p2Parent;

class ConversionBasicIOSAssetParam_1p1_1p2: ConversionBasicIOSAssetParam_1p1_1p2Parent
{
public:
	static NxParameterized::Conversion* Create(NxParameterized::Traits* t)
	{
		void* buf = t->alloc(sizeof(ConversionBasicIOSAssetParam_1p1_1p2));
		return buf ? PX_PLACEMENT_NEW(buf, ConversionBasicIOSAssetParam_1p1_1p2)(t) : 0;
	}

protected:
	ConversionBasicIOSAssetParam_1p1_1p2(NxParameterized::Traits* t) : ConversionBasicIOSAssetParam_1p1_1p2Parent(t) {}

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
