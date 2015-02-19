/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FORCEFIELD_ACTOR_H__
#define __FORCEFIELD_ACTOR_H__

#include "NxApex.h"

#include "NxForceFieldAsset.h"
#include "NxForceFieldActor.h"
#include "ForceFieldAsset.h"
#include "ApexActor.h"
#include "ApexInterface.h"
#include "ApexString.h"
#include "ApexRWLockable.h"
#include "ReadCheck.h"
#include "WriteCheck.h"
#include "NiFieldSampler.h"

#if defined(APEX_CUDA_SUPPORT)
#include "ApexCudaWrapper.h"
#endif

#include "ForceFieldFSCommon.h"

class ForceFieldAssetParams;

namespace physx
{
namespace apex
{

/*
PX_INLINE bool operator != (const NxGroupsMask64& d1, const NxGroupsMask64& d2)
{
	return d1.bits0 != d2.bits0 || d1.bits1 != d2.bits1;
}*/
PX_INLINE bool operator != (const physx::PxFilterData& d1, const physx::PxFilterData& d2)
{
	//if (d1.word3 != d2.word3) return d1.word3 < d2.word3;
	//if (d1.word2 != d2.word2) return d1.word2 < d2.word2;
	//if (d1.word1 != d2.word1) return d1.word1 < d2.word1;
	return d1.word0 != d2.word0 || d1.word1 != d2.word1 || d1.word2 != d2.word2 || d1.word3 != d2.word3;
}

namespace forcefield
{

class ForceFieldAsset;
class ForceFieldScene;

/**
Union class to hold all kernel parameter types. Avoided the use of templates 
for the getters, as that resulting code using traits for type safty 
was about the same amount as the non-templated one.
*/
class ForceFieldFSKernelParamsUnion
{
public:
	ForceFieldFSKernelParams& getForceFieldFSKernelParams()
	{
		return reinterpret_cast<ForceFieldFSKernelParams&>(params);
	}

	const ForceFieldFSKernelParams& getForceFieldFSKernelParams() const
	{
		return reinterpret_cast<const ForceFieldFSKernelParams&>(params);
	}

	const RadialForceFieldFSKernelParams& getRadialForceFieldFSKernelParams() const
	{
		PX_ASSERT(kernelType == ForceFieldKernelType::RADIAL);
		return reinterpret_cast<const RadialForceFieldFSKernelParams&>(params);
	}

	RadialForceFieldFSKernelParams& getRadialForceFieldFSKernelParams()
	{
		PX_ASSERT(kernelType == ForceFieldKernelType::RADIAL);
		return reinterpret_cast<RadialForceFieldFSKernelParams&>(params);
	}

	GenericForceFieldFSKernelParams& getGenericForceFieldFSKernelParams()
	{
		PX_ASSERT(kernelType == ForceFieldKernelType::GENERIC);
		return reinterpret_cast<GenericForceFieldFSKernelParams&>(params);
	}

	const GenericForceFieldFSKernelParams& getGenericForceFieldFSKernelParams() const
	{
		PX_ASSERT(kernelType == ForceFieldKernelType::GENERIC);
		return reinterpret_cast<const GenericForceFieldFSKernelParams&>(params);
	}

	ForceFieldKernelType::Enum kernelType;

private:

	union
	{
		void* alignment; //makes data aligned to pointer size
		PxU8 radial[sizeof(RadialForceFieldFSKernelParams)];
		PxU8 generic[sizeof(GenericForceFieldFSKernelParams)];
	} params;
};

class ForceFieldActor : public NxForceFieldActor, public ApexRWLockable, public ApexActor, public ApexActorSource, public NxApexResource, public ApexResource, public NiFieldSampler
{
public:
	APEX_RW_LOCKABLE_BOILERPLATE

	/* ForceFieldActor methods */
	ForceFieldActor(const NxForceFieldActorDesc&, ForceFieldAsset&, NxResourceList&, ForceFieldScene&);
	~ForceFieldActor() {}
	NxForceFieldAsset* 	getForceFieldAsset() const;

	bool				disable();
	bool				enable();
	bool				isEnable()
	{
		NX_READ_ZONE();
		return mEnable;
	}
	physx::PxMat44		getPose() const;
	void				setPose(const physx::PxMat44& pose);

	physx::PxF32	getCurrentScale(void) const
	{
		NX_READ_ZONE();
		return getScale();
	}

	void setCurrentScale(PxF32 scale)
	{
		NX_WRITE_ZONE();
		setScale(scale);
	}

	PX_DEPRECATED physx::PxF32		getScale() const
	{
		NX_READ_ZONE();
		return 0.0f;
	}

	PX_DEPRECATED void				setScale(physx::PxF32 scale);

	const char*			getName() const
	{
		NX_READ_ZONE();
		return mName.c_str();
	}
	void				setName(const char* name)
	{
		NX_WRITE_ZONE();
		mName = name;
	}

	void				setStrength(const physx::PxF32 strength);
	void				setLifetime(const physx::PxF32 lifetime);

	//kernel specific functionality
	void				setRadialFalloffType(const char* type);
	void				setRadialFalloffMultiplier(const physx::PxF32 multiplier);

	// deprecated
	void				setFalloffType(const char* type);
	void				setFalloffMultiplier(const physx::PxF32 multiplier);

	void                updatePoseAndBounds();  // Called by ExampleScene::fetchResults()

	/* NxApexResource, ApexResource */
	void				release();
	physx::PxU32		getListIndex() const
	{
		return m_listIndex;
	}
	void				setListIndex(class NxResourceList& list, physx::PxU32 index)
	{
		m_list = &list;
		m_listIndex = index;
	}

	/* NxApexActor, ApexActor */
	void                destroy();
	NxApexAsset*		getOwner() const;

	/* PhysX scene management */
	void				setPhysXScene(PxScene*);
	PxScene*			getPhysXScene() const;

	void				getPhysicalLodRange(physx::PxF32& min, physx::PxF32& max, bool& intOnly) const;
	physx::PxF32		getActivePhysicalLod() const;
	void				forcePhysicalLod(physx::PxF32 lod);
	/**
	\brief Selectively enables/disables debug visualization of a specific APEX actor.  Default value it true.
	*/
	virtual void setEnableDebugVisualization(bool state)
	{
		NX_WRITE_ZONE();
		ApexActor::setEnableDebugVisualization(state);
	}

	/* NiFieldSampler */
	virtual bool		updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled);

	virtual physx::PxVec3 queryFieldSamplerVelocity() const
	{
		return physx::PxVec3(0.0f);
	}

protected:
	void				updateForceField(physx::PxF32 dt);
	void				releaseNxForceField();

protected:
	ForceFieldScene*		mForceFieldScene;
	
	//not used, setters and getters deprecated
	//physx::PxF32			mScale;

	physx::PxU32			mFlags;

	ApexSimpleString		mName;

	ForceFieldAsset*		mAsset;

	bool					mEnable;
	physx::PxF32			mElapsedTime;

	/* Force field actor parameters */
	physx::PxF32			mLifetime;
	void					initActorParams(const PxMat44& initialPose);

	/* Field Sampler Stuff */
	bool					mFieldSamplerChanged;
	void					initFieldSampler(const NxForceFieldActorDesc& desc);
	void					releaseFieldSampler();

	/* Debug Rendering Stuff */
	void					visualize();
	void					visualizeIncludeShape();
	void					visualizeForces();

	ForceFieldFSKernelParamsUnion mKernelParams;
	ForceFieldFSKernelParamsUnion mKernelExecutionParams; //buffered data

	friend class ForceFieldScene;
};

class ForceFieldActorCPU : public ForceFieldActor
{
public:
	ForceFieldActorCPU(const NxForceFieldActorDesc& desc, ForceFieldAsset& asset, NxResourceList& list, ForceFieldScene& scene);
	~ForceFieldActorCPU();

	/* NiFieldSampler */
	virtual void executeFieldSampler(const ExecuteData& data);

	/**
	\brief Selectively enables/disables debug visualization of a specific APEX actor.  Default value it true.
	*/
	virtual void setEnableDebugVisualization(bool state)
	{
		ApexActor::setEnableDebugVisualization(state);
	}


private:
};

#if defined(APEX_CUDA_SUPPORT)

class ForceFieldActorGPU : public ForceFieldActorCPU
{
public:
	ForceFieldActorGPU(const NxForceFieldActorDesc& desc, ForceFieldAsset& asset, NxResourceList& list, ForceFieldScene& scene);
	~ForceFieldActorGPU();

	/* NiFieldSampler */
	virtual bool updateFieldSampler(NiFieldShapeDesc& shapeDesc, bool& isEnabled);

	virtual void getFieldSamplerCudaExecuteInfo(CudaExecuteInfo& info) const;

private:
	ApexCudaConstMemGroup				mConstMemGroup;
	InplaceHandle<RadialForceFieldFSKernelParams>	mParamsHandle;
};

#endif

}
}
} // end namespace physx::apex

#endif
