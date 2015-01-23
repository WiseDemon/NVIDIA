/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef CONVERSIONSIMPLEPARTICLESYSTEMPARAMS_0P2_0P3H_H
#define CONVERSIONSIMPLEPARTICLESYSTEMPARAMS_0P2_0P3H_H

#include "ParamConversionTemplate.h"
#include "SimpleParticleSystemParams_0p2.h"
#include "SimpleParticleSystemParams_0p3.h"

namespace physx
{
namespace apex
{

typedef ParamConversionTemplate<SimpleParticleSystemParams_0p2, SimpleParticleSystemParams_0p3, 2, 3> ConversionSimpleParticleSystemParams_0p2_0p3Parent;

class ConversionSimpleParticleSystemParams_0p2_0p3: ConversionSimpleParticleSystemParams_0p2_0p3Parent
{
public:
	static NxParameterized::Conversion* Create(NxParameterized::Traits* t)
	{
		void* buf = t->alloc(sizeof(ConversionSimpleParticleSystemParams_0p2_0p3));
		return buf ? PX_PLACEMENT_NEW(buf, ConversionSimpleParticleSystemParams_0p2_0p3)(t) : 0;
	}

protected:
	ConversionSimpleParticleSystemParams_0p2_0p3(NxParameterized::Traits* t) : ConversionSimpleParticleSystemParams_0p2_0p3Parent(t) {}

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
		//mNewData->GridDensity.Resolution = mLegacyData->GridDensityGrid.GridResolution;
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
