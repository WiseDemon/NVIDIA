/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef IOFX_ASSET_H
#define IOFX_ASSET_H

#include "NxApex.h"
#include "PsShare.h"
#include "ApexSDKHelpers.h"
#include "ApexAssetAuthoring.h"
#include "ApexAssetTracker.h"
#include "ApexInterface.h"
#include "ApexContext.h"
#include "NxIofxAsset.h"
#include "NxModifier.h"
#include "ApexString.h"
#include "NiResourceProvider.h"
#include "ApexAuthorableObject.h"
#include "IofxAssetParameters.h"
#include "MeshIofxParameters.h"
#include "SpriteIofxParameters.h"
#include "NxParamArray.h"
#include "Modifier.h"
#include "ApexRWLockable.h"
#include "ReadCheck.h"
#include "WriteCheck.h"
#include "ApexAuthorableObject.h"

// instead of having an initial color modifier, just drop
// in 4 color vs life modifiers instead
#define IOFX_SLOW_COMPOSITE_MODIFIERS 1

namespace physx
{
namespace apex
{
namespace iofx
{

typedef physx::Array<NxModifier*> ModifierStack;

class ModuleIofx;

class IofxAsset : public NxIofxAsset,
	public NxResourceList,
	public NxApexResource,
	public ApexResource,
	public NxParameterized::SerializationCallback,
	public ApexContext,
	public ApexRWLockable
{
	friend class IofxAssetDummyAuthoring;
protected:
	IofxAsset(ModuleIofx* module, NxResourceList& list, const char* name);
	IofxAsset(ModuleIofx* module,
	          NxResourceList& list,
	          NxParameterized::Interface* params,
	          const char* name);

public:
	APEX_RW_LOCKABLE_BOILERPLATE

	~IofxAsset();

	// NxApexAsset
	virtual void								release();
	virtual const char* 						getName(void) const
	{
		NX_READ_ZONE();
		return mName.c_str();
	}
	virtual NxAuthObjTypeID						getObjTypeID() const
	{
		NX_READ_ZONE();
		return mAssetTypeID;
	}
	virtual const char* 						getObjTypeName() const
	{
		NX_READ_ZONE();
		return getClassName();
	}
	virtual physx::PxU32						forceLoadAssets();
	// NxApexResource
	virtual void								setListIndex(NxResourceList& list, physx::PxU32 index)
	{
		m_listIndex = index;
		m_list = &list;
	}
	virtual physx::PxU32						getListIndex() const
	{
		return m_listIndex;
	}

	// NxApexContext
	virtual void								removeAllActors();
	NxApexRenderableIterator*					createRenderableIterator()
	{
		return ApexContext::createRenderableIterator();
	}
	void										releaseRenderableIterator(NxApexRenderableIterator& iter)
	{
		ApexContext::releaseRenderableIterator(iter);
	}

	void										addDependentActor(ApexActor* actor);

	bool isOpaqueMesh(physx::PxU32 index) const;

	virtual physx::PxU32						getMeshAssetCount() const
	{
		NX_READ_ZONE();
		return mRenderMeshList ? mRenderMeshList->size() : 0;
	}
	virtual const char*							getMeshAssetName(physx::PxU32 index) const;
	virtual physx::PxU32   						getMeshAssetWeight(physx::PxU32 index) const;
	virtual const char*							getSpriteMaterialName() const;
	virtual physx::PxU32						getContinuousModifierCount() const
	{
		return mContinuousModifierStack.size();
	}
	virtual const NxModifier*					getSpawnModifiers(physx::PxU32& outCount) const
	{
		NX_READ_ZONE();
		outCount = mSpawnModifierStack.size();
		return mSpawnModifierStack.front();
	}
	virtual const NxModifier*					getContinuousModifiers(physx::PxU32& outCount) const
	{
		NX_READ_ZONE();
		outCount = mContinuousModifierStack.size();
		return mContinuousModifierStack.front();
	}

	template<class ModifierType>
	PX_INLINE physx::PxF32								getMaxYFromCurveModifier(NxModifier* modifier) const
	{
		PxF32 maxScale = 0.0f;
		ModifierType* m = DYNAMIC_CAST(ModifierType*)(modifier);
		const NxCurve* curve = m->getFunction();
		PxU32 numControlPoints = 0;
		const NxVec2R* controlPoints = curve->getControlPoints(numControlPoints);
		for (PxU32 j = 0; j < numControlPoints; ++j)
		{
			maxScale = physx::PxMax(maxScale, controlPoints->y);
			++controlPoints;
		}
		return maxScale;
	}

	virtual physx::PxF32						getScaleUpperBound(physx::PxF32 maxVelocity) const
	{
		// check all modifiers and return the biggest scale they can produce
		NX_READ_ZONE();
		PxF32 scale = 1.0f;

		for (PxU32 i = 0; i < mSpawnModifierStack.size(); ++i)
		{
			NxModifier* modifier = mSpawnModifierStack[i];

			switch (modifier->getModifierType())
			{
			case ModifierType_SimpleScale:
			{
				SimpleScaleModifier* m = DYNAMIC_CAST(SimpleScaleModifier*)(modifier);
				scale *= m->getScaleFactor().maxElement();
				break;
			}
			case ModifierType_RandomScale:
			{
				RandomScaleModifier* m = DYNAMIC_CAST(RandomScaleModifier*)(modifier);
				scale *= m->getScaleFactor().maximum;
				break;
			}
			default:
				break;
			}

		}
		for (PxU32 i = 0; i < mContinuousModifierStack.size(); ++i)
		{
			NxModifier* modifier = mContinuousModifierStack[i];
			switch (mContinuousModifierStack[i]->getModifierType())
			{
			case ModifierType_ScaleAlongVelocity:
			{
				ScaleAlongVelocityModifier* m = DYNAMIC_CAST(ScaleAlongVelocityModifier*)(modifier);
				scale *= m->getScaleFactor() * maxVelocity;
				break;
			}
			case ModifierType_ScaleVsLife:
			{
				PxF32 maxScale = getMaxYFromCurveModifier<ScaleVsLifeModifier>(modifier);
				if (maxScale != 0.0f)
				{
					scale *= maxScale;
				}
				break;
			}
			case ModifierType_ScaleVsDensity:
			{
				PxF32 maxScale = getMaxYFromCurveModifier<ScaleVsDensityModifier>(modifier);
				if (maxScale != 0.0f)
				{
					scale *= maxScale;
				}
				break;
			}
			case ModifierType_ScaleVsCameraDistance:
			{
				PxF32 maxScale = getMaxYFromCurveModifier<ScaleVsCameraDistanceModifier>(modifier);
				if (maxScale != 0.0f)
				{
					scale *= maxScale;
				}
				break;
			}
			case ModifierType_OrientScaleAlongScreenVelocity:
			{
				OrientScaleAlongScreenVelocityModifier* m = DYNAMIC_CAST(OrientScaleAlongScreenVelocityModifier*)(modifier);
				scale *= m->getScalePerVelocity() * maxVelocity;
				break;
			}
			default:
				break;
			}
		}

		return scale;
	}

	physx::PxU32								getSpriteSemanticsBitmap() const
	{
		return mSpriteSemanticBitmap;
	}
	PX_INLINE bool								isSpriteSemanticUsed(NxRenderSpriteSemantic::Enum semantic)
	{
		return (((1 << semantic) & mSpriteSemanticBitmap) ? true : false);
	}
	void										setSpriteSemanticsUsed(physx::PxU32 spriteSemanticsBitmap);

	physx::PxU32								getMeshSemanticsBitmap() const
	{
		return mMeshSemanticBitmap;
	}
	PX_INLINE bool								isMeshSemanticUsed(NxRenderInstanceSemantic::Enum semantic)
	{
		return (((1 << semantic) & mMeshSemanticBitmap) ? true : false);
	}
	void										setMeshSemanticsUsed(physx::PxU32 meshSemanticsBitmap);

	/* objects that assist in force loading and proper "assets own assets" behavior */
	ApexAssetTracker							mRenderMeshAssetTracker;
	ApexAssetTracker							mSpriteMaterialAssetTracker;

	const NxParameterized::Interface* getAssetNxParameterized() const
	{
		return mParams;
	}
	/**
	 * \brief Releases the ApexAsset but returns the NxParameterized::Interface and *ownership* to the caller.
	 */
	virtual NxParameterized::Interface* releaseAndReturnNxParameterizedInterface(void)
	{
		NxParameterized::Interface* ret = mParams;
		mParams = NULL;
		release();
		return ret;
	}
	NxParameterized::Interface* getDefaultActorDesc()
	{
		NX_READ_ZONE();
		APEX_INVALID_OPERATION("Not yet implemented!");
		return NULL;
	};

	NxParameterized::Interface* getDefaultAssetPreviewDesc()
	{
		NX_READ_ZONE();
		APEX_INVALID_OPERATION("Not yet implemented!");
		return NULL;
	}

	virtual NxApexActor* createApexActor(const NxParameterized::Interface& /*parms*/, NxApexScene& /*apexScene*/)
	{
		NX_READ_ZONE();
		APEX_INVALID_OPERATION("Not yet implemented!");
		return NULL;
	}

	virtual NxApexAssetPreview* createApexAssetPreview(const NxParameterized::Interface& /*params*/, NxApexAssetPreviewScene* /*previewScene*/)
	{
		NX_READ_ZONE();
		APEX_INVALID_OPERATION("Not yet implemented!");
		return NULL;
	}

	virtual bool isValidForActorCreation(const ::NxParameterized::Interface& /*parms*/, NxApexScene& /*apexScene*/) const
	{
		NX_READ_ZONE();
		return true; // todo, implement!
	}

	virtual bool isDirty() const
	{
		NX_READ_ZONE();
		return false;
	}

	void										destroy();
	static const char* 							getClassName()
	{
		return NX_IOFX_AUTHORING_TYPE_NAME;
	}
	static NxAuthObjTypeID						mAssetTypeID;

	struct RenderMesh
	{
		physx::PxU32        mWeight;
		ApexSimpleString	mMeshAssetName;
	};

	// authorable data
	IofxAssetParameters*					   	mParams;
	physx::Array<RenderMesh>					mRenderMeshes;
	ModifierStack								mSpawnModifierStack;
	ModifierStack								mContinuousModifierStack;
	NxParamArray<MeshIofxParametersNS::meshProperties_Type> *mRenderMeshList;
	SpriteIofxParameters*				   		mSpriteParams;

	// runtime data
	ModuleIofx*							        mModule;
	ApexSimpleString							mName;
	physx::PxU32								mSpriteSemanticBitmap;
	physx::PxU32								mMeshSemanticBitmap;

#if IOFX_SLOW_COMPOSITE_MODIFIERS
	physx::Array<NxParameterized::Interface*>	mCompositeParams;
#endif

	PX_INLINE ModifierStack&					getModifierStack(physx::PxU32 modStage)
	{
		switch (modStage)
		{
		case ModifierStage_Spawn:
			return mSpawnModifierStack;
		case ModifierStage_Continuous:
			return mContinuousModifierStack;
		default:
			PX_ALWAYS_ASSERT();
		};
		return mSpawnModifierStack; // should never get here.
	}

	PX_INLINE const ModifierStack&				getModifierStack(physx::PxU32 modStage) const
	{
		switch (modStage)
		{
		case ModifierStage_Spawn:
			return mSpawnModifierStack;
		case ModifierStage_Continuous:
			return mContinuousModifierStack;
		default:
			PX_ALWAYS_ASSERT();
		};
		return mSpawnModifierStack; // should never get here.
	}

	PX_INLINE NxParameterized::ErrorType getModifierStack(physx::PxU32 modStage, NxParameterized::Handle& h)
	{
		switch (modStage)
		{
		case ModifierStage_Spawn:
			return mParams->getParameterHandle("spawnModifierList", h);
		case ModifierStage_Continuous:
			return mParams->getParameterHandle("continuousModifierList", h);
		default:
			PX_ALWAYS_ASSERT();
		};
		return NxParameterized::ERROR_INDEX_OUT_OF_RANGE; // should never get here.
	}

	PX_INLINE NxParameterized::ErrorType getModifierStack(physx::PxU32 modStage, NxParameterized::Handle& h) const
	{
		switch (modStage)
		{
		case ModifierStage_Spawn:
			return mParams->getParameterHandle("spawnModifierList", h);
		case ModifierStage_Continuous:
			return mParams->getParameterHandle("continuousModifierList", h);
		default:
			PX_ALWAYS_ASSERT();
		};
		return NxParameterized::ERROR_INDEX_OUT_OF_RANGE; // should never get here.
	}

	bool isSortingEnabled() const;

	/* NxParameterized Serialization callbacks */
	void						preSerialize(void* userData = NULL);
	void						postDeserialize(void* userData = NULL);

	// initialize a table of assets and resource IDs for resource tracking
	void						initializeAssetNameTable();

	PxU32						getPubStateSize() const;
	PxU32						getPrivStateSize() const;

	friend class ModuleIofx;
	template <class T_Module, class T_Asset, class T_AssetAuthoring> friend class physx::apex::ApexAuthorableObject;
};

#ifndef WITHOUT_APEX_AUTHORING
class IofxAssetAuthoring : public NxIofxAssetAuthoring, public IofxAsset, public ApexAssetAuthoring
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	IofxAssetAuthoring(ModuleIofx* module, NxResourceList& list) :
		IofxAsset(module, list, "IofxAssetAuthoring") {}

	IofxAssetAuthoring(ModuleIofx* module, NxResourceList& list, const char* name) :
		IofxAsset(module, list, name) {}

	IofxAssetAuthoring(ModuleIofx* module, NxResourceList& list, NxParameterized::Interface* params, const char* name) :
		IofxAsset(module, list, params, name) {}

	virtual void	        release();

	virtual const char* 			getName(void) const
	{
		NX_READ_ZONE();
		return IofxAsset::getName();
	}
	virtual const char* 			getObjTypeName() const;
	virtual bool					prepareForPlatform(physx::apex::NxPlatformTag)
	{
		APEX_INVALID_OPERATION("Not Implemented.");
		return false;
	}

	void setToolString(const char* toolName, const char* toolVersion, PxU32 toolChangelist)
	{
		ApexAssetAuthoring::setToolString(toolName, toolVersion, toolChangelist);
	}

#if IOFX_AUTHORING_API_ENABLED
	virtual void	        setMeshAssetName(const char* meshAssetName, physx::PxU32 meshIndex = 0);
	virtual void	        setMeshAssetWeight(const physx::PxU32 weight, physx::PxU32 meshIndex = 0);
	virtual void            setSpriteMaterialName(const char* spriteMaterialName);

	virtual physx::PxU32	getMeshAssetCount()	const
	{
		return IofxAsset::getMeshAssetCount();
	}
	virtual const char*		getMeshAssetName(physx::PxU32 index) const
	{
		return IofxAsset::getMeshAssetName(index);
	}
	virtual physx::PxU32   	getMeshAssetWeight(physx::PxU32 index) const
	{
		return IofxAsset::getMeshAssetWeight(index);
	}
	virtual const char*		getSpriteMaterialName() const
	{
		return IofxAsset::getSpriteMaterialName();
	}

	virtual NxModifier*		createModifier(physx::PxU32 modStage, physx::PxU32 modType);

	virtual void			removeModifier(physx::PxU32 modStage, physx::PxU32 position);
	virtual physx::PxU32	findModifier(physx::PxU32 modStage, NxModifier* modifier);
	virtual NxModifier* 	getModifier(physx::PxU32 modStage, physx::PxU32 position) const;
	virtual physx::PxU32	getModifierCount(physx::PxU32 modStage) const;
#endif

	NxParameterized::Interface* getNxParameterized() const
	{
		NX_READ_ZONE();
		return (NxParameterized::Interface*)getAssetNxParameterized();
	}
	/**
	 * \brief Releases the ApexAsset but returns the NxParameterized::Interface and *ownership* to the caller.
	 */
	virtual NxParameterized::Interface* releaseAndReturnNxParameterizedInterface(void)
	{
		NX_READ_ZONE();
		NxParameterized::Interface* ret = mParams;
		mParams = NULL;
		release();
		return ret;
	}
	physx::PxU32			getAssetTarget() const;
};
#endif

}
}
} // namespace apex

#endif // IOFX_ASSET_H
