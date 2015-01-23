/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __IOFX_RENDER_DATA_H__
#define __IOFX_RENDER_DATA_H__

#include "PsShare.h"
#include "PsUserAllocated.h"

#if defined(APEX_CUDA_SUPPORT)
#include "ApexCuda.h"
#endif

namespace physx
{
namespace apex
{
namespace iofx
{

class IofxActor;
class IosObjectBaseData;

class IofxSharedRenderData : public physx::UserAllocated
{
public:
	virtual ~IofxSharedRenderData() = 0 {}

	virtual void alloc(IosObjectBaseData*, PxCudaContextManager*) = 0;

	virtual void release() = 0;

	virtual bool update(IosObjectBaseData* objData) = 0;

#if defined(APEX_CUDA_SUPPORT)
	static const PxU32 RESOURCE_LIST_MAX_COUNT = 8;

	virtual bool getResourceList(PxU32& count, CUgraphicsResource* list) = 0;
	virtual bool resolveResourceList(CUdeviceptr& ptr, PxU32& arrayCount, CUarray* arrayList) = 0;


	bool addResourcesToArray(physx::Array<CUgraphicsResource> &resourceArray)
	{
		PxU32 resCount;
		CUgraphicsResource resList[RESOURCE_LIST_MAX_COUNT];

		if (getResourceList(resCount, resList))
		{
			for( PxU32 i = 0; i < resCount; ++i )
			{
				resourceArray.pushBack( resList[i] );
			}
			return true;
		}
		return false;
	}

#endif

	virtual bool checkSemantics(PxU32 semantics) const = 0;

	PX_INLINE bool getUseInterop() const
	{
		return useInterop;
	}
	PX_INLINE void setUseInterop(bool value)
	{
		useInterop = value;
	}
	PX_INLINE void setBufferIsMapped(bool value)
	{
		bufferIsMapped = value;
	}

	PX_INLINE PxU32 getAllocSemantics() const
	{
		return allocSemantics;
	}

	PX_INLINE bool getBufferIsMapped() const
	{
		return bufferIsMapped;
	}

	PX_INLINE PxU32 getInstanceID() const
	{
		return instanceID;
	}

protected:
	IofxSharedRenderData(PxU32 instance)
		: instanceID(instance)
	{
		allocSemantics = 0;

		useInterop = false;
		bufferIsMapped = false;
	}

	const PxU32	instanceID;

	PxU32	allocSemantics;

	bool	useInterop;
	bool	bufferIsMapped;

	template<typename SemaTy>
	static bool checkSemantics(PxU32 semantics, PxU32 allocSemantics)
	{
		return (semantics & allocSemantics) == semantics;
	}

private:
	IofxSharedRenderData& operator=(const IofxSharedRenderData&);
};

class IofxSharedRenderDataMesh : public IofxSharedRenderData
{
public:
	IofxSharedRenderDataMesh(PxU32 instance);
	virtual ~IofxSharedRenderDataMesh() { IofxSharedRenderDataMesh::release(); }

	virtual void alloc(IosObjectBaseData*, PxCudaContextManager*);

	virtual void release();

	virtual bool update(IosObjectBaseData* objData);

#if defined(APEX_CUDA_SUPPORT)
	virtual bool getResourceList(PxU32& count, CUgraphicsResource* list);
	virtual bool resolveResourceList(CUdeviceptr& ptr, PxU32& arrayCount, CUarray* arrayList);
#endif

	virtual bool checkSemantics(PxU32 semantics) const
	{
		return IofxSharedRenderData::checkSemantics<NxRenderInstanceSemantic>(semantics, allocSemantics);
	}

	PX_INLINE NxUserRenderInstanceBuffer* getInstanceBuffer() const
	{
		return instanceBuffer;
	}
	PX_INLINE const NxUserRenderInstanceBufferDesc& getInstanceBufferDesc() const
	{
		return instanceBufferDesc;
	}

private:
	IofxSharedRenderDataMesh& operator=(const IofxSharedRenderDataMesh&);

	NxUserRenderInstanceBuffer*		instanceBuffer;
	NxUserRenderInstanceBufferDesc	instanceBufferDesc;
};

class IofxSharedRenderDataSprite : public IofxSharedRenderData
{
public:
	IofxSharedRenderDataSprite(PxU32 instance);
	virtual ~IofxSharedRenderDataSprite() { IofxSharedRenderDataSprite::release(); }

	virtual void alloc(IosObjectBaseData*, PxCudaContextManager*);

	virtual void release();

	virtual bool update(IosObjectBaseData* objData);

#if defined(APEX_CUDA_SUPPORT)
	virtual bool getResourceList(PxU32& count, CUgraphicsResource* list);
	virtual bool resolveResourceList(CUdeviceptr& ptr, PxU32& arrayCount, CUarray* arrayList);
#endif

	virtual bool checkSemantics(PxU32 semantics) const
	{
		return IofxSharedRenderData::checkSemantics<NxRenderSpriteSemantic>(semantics, allocSemantics);
	}

	PX_INLINE NxUserRenderSpriteBuffer* getSpriteBuffer() const
	{
		return spriteBuffer;
	}
	PX_INLINE const NxUserRenderSpriteBufferDesc& getSpriteBufferDesc() const
	{
		return spriteBufferDesc;
	}

	//TODO: maybe remove these methods and use getSpriteBufferDesc() instead
	PX_INLINE PxU32 getMaxSprites() const
	{
		return spriteBufferDesc.maxSprites;
	}
	PX_INLINE PxU32 getTextureCount() const
	{
		return spriteBufferDesc.textureCount;
	}
	PX_INLINE const NxUserRenderSpriteTextureDesc* getTextureDescArray() const
	{
		return spriteBufferDesc.textureDescs;
	}
	PX_INLINE const NxUserRenderSpriteTextureDesc& getTextureDesc(PxU32 i) const
	{
		PX_ASSERT(i < spriteBufferDesc.textureCount);
		return spriteBufferDesc.textureDescs[i];
	}

private:
	IofxSharedRenderDataSprite& operator=(const IofxSharedRenderDataSprite&);

	NxUserRenderSpriteBuffer*		spriteBuffer;
	NxUserRenderSpriteBufferDesc	spriteBufferDesc;
};


class IofxActorRenderData : public physx::UserAllocated
{
public:
	virtual void updateRenderResources(bool rewriteBuffers, void* userRenderData) = 0;
	virtual void dispatchRenderResources(NxUserRenderer& renderer) = 0;

	virtual ~IofxActorRenderData() {}

	void setSharedRenderData(IofxSharedRenderData* sharedRenderData)
	{
		mSharedRenderData = sharedRenderData;
	}

protected:
	IofxActorRenderData(IofxActor* iofxActor) : mIofxActor(iofxActor), mSharedRenderData(NULL) {}

	IofxActor* mIofxActor;
	IofxSharedRenderData* mSharedRenderData;
};

class IofxActorRenderDataMesh : public IofxActorRenderData
{
public:
	IofxActorRenderDataMesh(IofxActor* iofxActor, NxRenderMeshActor* renderMeshActor)
		: IofxActorRenderData(iofxActor), mRenderMeshActor(renderMeshActor)
	{
	}
	virtual ~IofxActorRenderDataMesh()
	{
		if (mRenderMeshActor != NULL)
		{
			mRenderMeshActor->release();
		}
	}

	virtual void updateRenderResources(bool rewriteBuffers, void* userRenderData);
	virtual void dispatchRenderResources(NxUserRenderer& renderer);

private:
	NxRenderMeshActor*			mRenderMeshActor;
};

class IofxActorRenderDataSprite : public IofxActorRenderData
{
public:
	IofxActorRenderDataSprite(IofxActor* iofxActor, void* spriteMaterial)
		: IofxActorRenderData(iofxActor), mSpriteMaterial(spriteMaterial), mRenderResource(NULL)
	{
	}
	virtual ~IofxActorRenderDataSprite()
	{
		if (mRenderResource != NULL)
		{
			NiGetApexSDK()->getUserRenderResourceManager()->releaseResource(*mRenderResource);
		}
	}

	virtual void updateRenderResources(bool rewriteBuffers, void* userRenderData);
	virtual void dispatchRenderResources(NxUserRenderer& renderer);

private:
	void*						mSpriteMaterial;
	NxUserRenderResource*		mRenderResource;
};

}
}
} // namespace apex

#endif /* __IOFX_RENDER_DATA_H__ */
