/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef CONVERSIONBASICIOSASSETPARAM_1P0_1P1H_H
#define CONVERSIONBASICIOSASSETPARAM_1P0_1P1H_H

#include "ParamConversionTemplate.h"
#include "BasicIOSAssetParam_1p0.h"
#include "BasicIOSAssetParam_1p1.h"

namespace physx
{
namespace apex
{

typedef ParamConversionTemplate<BasicIOSAssetParam_1p0, BasicIOSAssetParam_1p1, 65536, 65537> ConversionBasicIOSAssetParam_1p0_1p1Parent;

class ConversionBasicIOSAssetParam_1p0_1p1: ConversionBasicIOSAssetParam_1p0_1p1Parent
{
public:
	static NxParameterized::Conversion* Create(NxParameterized::Traits* t)
	{
		void* buf = t->alloc(sizeof(ConversionBasicIOSAssetParam_1p0_1p1));
		return buf ? PX_PLACEMENT_NEW(buf, ConversionBasicIOSAssetParam_1p0_1p1)(t) : 0;
	}

protected:
	ConversionBasicIOSAssetParam_1p0_1p1(NxParameterized::Traits* t) : ConversionBasicIOSAssetParam_1p0_1p1Parent(t) {}

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
		mNewData->GridDensity.Enabled = mLegacyData->GridDensityGrid.Enabled;
		// mNewData->GridDensity.Resolution = mLegacyData->GridDensityGrid.GridResolution;
		mNewData->GridDensity.GridSize = mLegacyData->GridDensityGrid.FrustumParams.GridSize;
		mNewData->GridDensity.MaxCellCount = mLegacyData->GridDensityGrid.FrustumParams.GridMaxCellCount;

		// enums are strings, better do it the safe way
		NxParameterized::Handle hEnumNew(*mNewData, "GridDensity.Resolution");
		NxParameterized::Handle hEnumOld(*mLegacyData, "GridDensityGrid.GridResolution");
		PX_ASSERT(hEnumNew.isValid());
		PX_ASSERT(hEnumOld.isValid());

		const NxParameterized::Definition* paramDefOld;
		paramDefOld = hEnumOld.parameterDefinition();
		physx::PxI32 index = paramDefOld->enumValIndex(mLegacyData->GridDensityGrid.GridResolution);

		const NxParameterized::Definition* paramDefNew;
		paramDefNew = hEnumNew.parameterDefinition();
		hEnumNew.setParamEnum(paramDefNew->enumVal(index));

		return true;
	}
};

}
} // namespace physx::apex

#endif
