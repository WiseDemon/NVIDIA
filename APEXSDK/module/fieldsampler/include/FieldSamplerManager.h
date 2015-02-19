/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __FIELD_SAMPLER_MANAGER_H__
#define __FIELD_SAMPLER_MANAGER_H__

#include "NiFieldSamplerManager.h"
#include "FieldSamplerScene.h"
#include "ApexGroupsFiltering.h"

#if defined(APEX_CUDA_SUPPORT)
#define CUDA_OBJ(name) SCENE_CUDA_OBJ(mManager->getScene(), name)
#endif

namespace physx
{
namespace apex
{

#if NX_SDK_VERSION_MAJOR == 3
PX_INLINE bool operator != (const PxFilterData& d1, const PxFilterData& d2)
{
	return d1.word0 != d2.word0 || d1.word1 != d2.word1 || d1.word2 != d2.word2 || d1.word3 != d2.word3;
}
#else
PX_INLINE void setZero(NxGroupsMask64& mask)
{
	mask.bits0 = mask.bits1 = 0;
}
PX_INLINE bool hasBits(const NxGroupsMask64& mask)
{
	return (mask.bits0 | mask.bits1) != 0;
}

PX_INLINE NxGroupsMask64 operator ~(const NxGroupsMask64& arg)
{
	NxGroupsMask64 res;
	res.bits0 = ~arg.bits0;
	res.bits1 = ~arg.bits1;
	return res;
}
PX_INLINE NxGroupsMask64 operator & (const NxGroupsMask64& lhs, const NxGroupsMask64& rhs)
{
	NxGroupsMask64 res;
	res.bits0 = lhs.bits0 & rhs.bits0;
	res.bits1 = lhs.bits1 & rhs.bits1;
	return res;
}
PX_INLINE NxGroupsMask64 operator | (const NxGroupsMask64& lhs, const NxGroupsMask64& rhs)
{
	NxGroupsMask64 res;
	res.bits0 = lhs.bits0 | rhs.bits0;
	res.bits1 = lhs.bits1 | rhs.bits1;
	return res;
}
PX_INLINE NxGroupsMask64 operator ^(const NxGroupsMask64& lhs, const NxGroupsMask64& rhs)
{
	NxGroupsMask64 res;
	res.bits0 = lhs.bits0 ^ rhs.bits0;
	res.bits1 = lhs.bits1 ^ rhs.bits1;
	return res;
}
PX_INLINE NxGroupsMask64 SWAP_AND(const NxGroupsMask64& lhs, const NxGroupsMask64& rhs)
{
	NxGroupsMask64 res;
	res.bits0 = lhs.bits0 & rhs.bits1;
	res.bits1 = lhs.bits1 & rhs.bits0;
	return res;
}
PX_INLINE bool operator != (const NxGroupsMask64& lhs, const NxGroupsMask64& rhs)
{
	return (lhs.bits0 != rhs.bits0) || (lhs.bits1 != rhs.bits1);
}
#endif

namespace fieldsampler
{

class FieldSamplerScene;

class FieldSamplerQuery;
class FieldSamplerSceneWrapper;
class FieldSamplerWrapper;
class FieldBoundaryWrapper;

class FieldSamplerManager : public NiFieldSamplerManager,  public physx::UserAllocated
{
public:
	FieldSamplerManager(FieldSamplerScene* scene);
	virtual ~FieldSamplerManager() {}

	void		submitTasks();
	void		setTaskDependencies();
	void		fetchResults();

	/* NxFieldSamplerManager */
	NiFieldSamplerQuery* createFieldSamplerQuery(const NiFieldSamplerQueryDesc&);

	void registerFieldSampler(NiFieldSampler* , const NiFieldSamplerDesc& , NiFieldSamplerScene*);
	void unregisterFieldSampler(NiFieldSampler*);

	void registerFieldBoundary(NiFieldBoundary* , const NiFieldBoundaryDesc&);
	void unregisterFieldBoundary(NiFieldBoundary*);

#if NX_SDK_VERSION_MAJOR == 3
	void registerUnhandledParticleSystem(physx::PxActor*);
	void unregisterUnhandledParticleSystem(physx::PxActor*);
	bool isUnhandledParticleSystem(physx::PxActor*);
#endif

	FieldSamplerScene*	getScene() const
	{
		return mScene;
	}
	NiApexScene&		getApexScene() const
	{
		return *mScene->mApexScene;
	}

#if NX_SDK_VERSION_MAJOR == 2
	void setFieldBoundaryGroupsFilteringParams(const NxGroupsFilteringParams64& params);
	void getFieldBoundaryGroupsFilteringParams(NxGroupsFilteringParams64& params) const;

	void setFieldSamplerGroupsFilteringParams(const NxGroupsFilteringParams64& params);
	void getFieldSamplerGroupsFilteringParams(NxGroupsFilteringParams64& params) const;


	const ApexGroupsFiltering<NxGroupsMask64>& getFieldBoundaryGroupsFiltering() const
	{
		return mFieldBoundaryGroupsFiltering;
	}

	const ApexGroupsFiltering<NxGroupsMask64>& getFieldSamplerGroupsFiltering() const
	{
		return mFieldSamplerGroupsFiltering;
	}

	virtual bool doFieldSamplerFiltering(const NxGroupsMask64 &o1, const NxGroupsMask64 &o2, PxF32 &weight) const
	{
		weight = 1.0f;
		return mFieldSamplerGroupsFiltering(o1, o2);
	}
#else
	bool getFieldSamplerGroupsFiltering(const PxFilterData &o1,const PxFilterData &o2,PxF32 &weight) const;
	void setFieldSamplerWeightedCollisionFilterCallback(NxFieldSamplerWeightedCollisionFilterCallback *callback)
	{
		mWeightedCallback = callback;
	}

	virtual bool doFieldSamplerFiltering(const PxFilterData &o1, const PxFilterData &o2, PxF32 &weight) const
	{
		return getFieldSamplerGroupsFiltering(o1, o2, weight);
	}
#endif

protected:
	virtual FieldSamplerQuery* allocateFieldSamplerQuery(const NiFieldSamplerQueryDesc&) = 0;
	virtual FieldSamplerSceneWrapper* allocateFieldSamplerSceneWrapper(NiFieldSamplerScene*) = 0;
	virtual FieldSamplerWrapper* allocateFieldSamplerWrapper(NiFieldSampler* , const NiFieldSamplerDesc& , FieldSamplerSceneWrapper*) = 0;

	static PX_INLINE void addFieldSamplerToQuery(FieldSamplerWrapper* fieldSamplerWrapper, FieldSamplerQuery* query);
	void addAllFieldSamplersToQuery(FieldSamplerQuery*) const;

	FieldSamplerScene*	mScene;

	NxResourceList		mFieldSamplerQueryList;
	NxResourceList		mFieldSamplerSceneWrapperList;
	NxResourceList		mFieldSamplerWrapperList;
	NxResourceList		mFieldBoundaryWrapperList;

#if NX_SDK_VERSION_MAJOR == 3
	physx::Array<physx::PxActor*>		mUnhandledParticleSystems;
	NxFieldSamplerWeightedCollisionFilterCallback *mWeightedCallback;
#endif

#if NX_SDK_VERSION_MAJOR == 2
	ApexGroupsFiltering<NxGroupsMask64>	mFieldBoundaryGroupsFiltering;
	ApexGroupsFiltering<NxGroupsMask64>	mFieldSamplerGroupsFiltering;
#endif

	bool								mFieldSamplerGroupsFilteringChanged;

	friend class FieldSamplerSceneWrapperGPU;
};

class FieldSamplerManagerCPU : public FieldSamplerManager
{
public:
	FieldSamplerManagerCPU(FieldSamplerScene* scene) : FieldSamplerManager(scene) {}

protected:
	FieldSamplerQuery* allocateFieldSamplerQuery(const NiFieldSamplerQueryDesc&);
	FieldSamplerSceneWrapper* allocateFieldSamplerSceneWrapper(NiFieldSamplerScene*);
	FieldSamplerWrapper* allocateFieldSamplerWrapper(NiFieldSampler* , const NiFieldSamplerDesc& , FieldSamplerSceneWrapper*);
};

#if defined(APEX_CUDA_SUPPORT)
class FieldSamplerManagerGPU : public FieldSamplerManager
{
public:
	FieldSamplerManagerGPU(FieldSamplerScene* scene) : FieldSamplerManager(scene) {}

protected:
	FieldSamplerQuery* allocateFieldSamplerQuery(const NiFieldSamplerQueryDesc&);
	FieldSamplerSceneWrapper* allocateFieldSamplerSceneWrapper(NiFieldSamplerScene*);
	FieldSamplerWrapper* allocateFieldSamplerWrapper(NiFieldSampler* , const NiFieldSamplerDesc& , FieldSamplerSceneWrapper*);
};
#endif // defined(APEX_CUDA_SUPPORT)

}
} // end namespace physx::apex
}
#endif // __FIELD_SAMPLER_MANAGER_H__
