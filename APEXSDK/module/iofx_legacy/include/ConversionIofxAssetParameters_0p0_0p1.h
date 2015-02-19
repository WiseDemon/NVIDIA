/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __CONVERSIONIOFXASSETPARAMETERS_0P0_0P1H__
#define __CONVERSIONIOFXASSETPARAMETERS_0P0_0P1H__

#include "ParamConversionTemplate.h"
#include "IofxAssetParameters_0p0.h"
#include "IofxAssetParameters_0p1.h"

#include "NxParamUtils.h"

#define PARAM_RET(x) if( (x) != NxParameterized::ERROR_NONE ) \
		{ PX_ASSERT(0 && "INVALID Parameter"); return false; }

namespace physx
{
namespace apex
{
namespace legacy
{

typedef ParamConversionTemplate<IofxAssetParameters_0p0, IofxAssetParameters_0p1, 0, 1> ConversionIofxAssetParameters_0p0_0p1Parent;

class ConversionIofxAssetParameters_0p0_0p1 : public ConversionIofxAssetParameters_0p0_0p1Parent
{
public:

	static NxParameterized::Conversion* Create(NxParameterized::Traits* traits)
	{
		void* buf = traits->alloc(sizeof(ConversionIofxAssetParameters_0p0_0p1));
		return PX_PLACEMENT_NEW(buf, ConversionIofxAssetParameters_0p0_0p1)(traits);
	}

private:

	ConversionIofxAssetParameters_0p0_0p1(NxParameterized::Traits* traits) : ConversionIofxAssetParameters_0p0_0p1Parent(traits) {}

	// TODO: make it a generic "array copy" method
	bool deepCopyModifiers(NxParameterized::Interface* newModListParams, const char* modListString)
	{
		physx::PxI32 arraySize = 0, modIgnoreCount = 0;
		NxParameterized::Handle hOld(*mLegacyData, modListString);
		NxParameterized::Handle hNew(*newModListParams, modListString);

		PX_ASSERT(hOld.isValid() && hNew.isValid());

		// resize the new array
		NxParameterized::getParamArraySize(*mLegacyData, modListString, arraySize);
		PARAM_RET(hNew.resizeArray(arraySize));

		// copy the modifiers
		for (PxI32 i = 0; i < arraySize; i++)
		{
			NxParameterized::Interface* curOldMod = NULL;

			PARAM_RET(hNew.set(i - modIgnoreCount));
			PARAM_RET(hOld.set(i));

			PARAM_RET(hOld.getParamRef(curOldMod));
			PX_ASSERT(curOldMod);

			if (!strcmp(curOldMod->className(), "RotationModifierParams") &&
			        !strcmp(newModListParams->className(), "SpriteIofxParameters"))
			{
				modIgnoreCount++;

				hNew.popIndex();
				hOld.popIndex();
				continue;
			}

			PARAM_RET(hNew.setParamRef(curOldMod));
			PARAM_RET(hOld.setParamRef(0));

			hNew.popIndex();
			hOld.popIndex();
		}

		if (modIgnoreCount)
		{
			PARAM_RET(hNew.resizeArray(arraySize - modIgnoreCount));
		}

		return true;
	}

	bool copyRenderMeshList(NxParameterized::Interface* meshIofxParams)
	{
		char indexedWeightString[32];

		NxParameterized::Handle hOld(*mLegacyData, "renderMeshList");
		PX_ASSERT(hOld.isValid());

		NxParameterized::Handle hNew(*meshIofxParams, "renderMeshList");
		PX_ASSERT(hNew.isValid());

		physx::PxI32 arraySize = 0;
		PARAM_RET(hOld.getArraySize(arraySize));
		PARAM_RET(hNew.resizeArray(arraySize));

		for (PxI32 i = 0; i < arraySize; i++)
		{
			PARAM_RET(hNew.set(i));
			PARAM_RET(hOld.set(i));

			// first handle the weight
			physx::PxU32 oldWeight;
			physx::string::sprintf_s(indexedWeightString, sizeof(indexedWeightString), "renderMeshList[%i].weight", i);
			NxParameterized::getParamU32(*mLegacyData, indexedWeightString, oldWeight);
			NxParameterized::setParamU32(*meshIofxParams, indexedWeightString, oldWeight);

			// then get the meshAssetName (named ref)
			NxParameterized::Handle hChild(mLegacyData);
			PARAM_RET(hOld.getChildHandle(mLegacyData, "meshAssetName", hChild));
			PX_ASSERT(hOld.isValid());

			NxParameterized::Interface* oldMeshRef = NULL;

			PARAM_RET(hChild.getParamRef(oldMeshRef));
			PARAM_RET(hChild.setParamRef(0));
			PX_ASSERT(oldMeshRef);

			PARAM_RET(hNew.getChildHandle(meshIofxParams, "meshAssetName", hChild));
			PARAM_RET(hChild.setParamRef(oldMeshRef));

			hNew.popIndex();
			hOld.popIndex();
		}

		return true;
	}

protected:

	bool convert()
	{
		// if sprite name is present, then it's a sprite modifier, else, mesh
		NxParameterized::Interface* iofxType = 0;
		if (mLegacyData->spriteMaterialName && mLegacyData->spriteMaterialName->name())
		{
			// create new sprite type

			iofxType = mTraits->createNxParameterized("SpriteIofxParameters", 0);
			PX_ASSERT(iofxType);

			// copy the new spriteMaterialName named reference
			NxParameterized::Handle h(*iofxType, "spriteMaterialName");
			h.setParamRef(mLegacyData->spriteMaterialName);
			mLegacyData->spriteMaterialName = 0;
		}
		else
		{
			// create new mesh iofx type

			iofxType = mTraits->createNxParameterized("MeshIofxParameters", 0);
			PX_ASSERT(iofxType);

			// copy the renderMeshList array (renderMesh names and weights)
			copyRenderMeshList(iofxType);
		}

		// deep copy the spawn and continuous list
		deepCopyModifiers(iofxType, "spawnModifierList");
		deepCopyModifiers(iofxType, "continuousModifierList");

		mNewData->iofxType = iofxType;

		return true;
	}
};

}
}
} //end of physx::apex:: namespace

#endif
