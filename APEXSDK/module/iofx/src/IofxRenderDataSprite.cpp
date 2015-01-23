/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "NxApex.h"
#include "IofxSceneCPU.h"
#include "IofxSceneGPU.h"
#include "IosObjectData.h"
#include "IofxRenderData.h"
#include "IofxActor.h"

namespace physx
{
namespace apex
{
namespace iofx
{

void IofxActorRenderDataSprite::updateRenderResources(bool rewriteBuffers, void* userRenderData)
{
	PX_UNUSED(rewriteBuffers);

	IofxSharedRenderDataSprite* renderDataSprite = DYNAMIC_CAST(IofxSharedRenderDataSprite*)(mSharedRenderData);
	NxUserRenderSpriteBuffer* spriteBuffer = renderDataSprite->getSpriteBuffer();

	if (mRenderResource == NULL || mRenderResource->getSpriteBuffer() != spriteBuffer)
	{
		NxUserRenderResourceManager* rrm = NiGetApexSDK()->getUserRenderResourceManager();

		if (mRenderResource != NULL)
		{
			/* The sprite buffer was re-allocated for more semantics.  We
			 * must release our old render resource and allocate a new one.
			 */
			rrm->releaseResource(*mRenderResource);
			mRenderResource = NULL;
		}

		if (spriteBuffer != NULL)
		{
			NxUserRenderResourceDesc rDesc;
			rDesc.spriteBuffer = spriteBuffer;
			rDesc.firstSprite = 0;
			rDesc.numSprites = renderDataSprite->getMaxSprites();
			rDesc.userRenderData = userRenderData;
			rDesc.primitives = NxRenderPrimitiveType::POINT_SPRITES;
			rDesc.material = mSpriteMaterial;

			mRenderResource = rrm->createResource(rDesc);
			//mSemantics = obj->allocSemantics;
		}
	}

	if (mRenderResource != NULL)
	{
		PX_ASSERT( mRenderResource->getSpriteBuffer() == spriteBuffer );

		const ObjectRange& range = mIofxActor->mRenderRange;
		mRenderResource->setSpriteBufferRange(range.startIndex, range.objectCount);
		mRenderResource->setSpriteVisibleCount(mIofxActor->mRenderVisibleCount);
	}
}

void IofxActorRenderDataSprite::dispatchRenderResources(NxUserRenderer& renderer)
{
	if (mRenderResource != NULL)
	{
		NxApexRenderContext context;
		context.world2local = physx::PxMat44::createIdentity();
		context.local2world = physx::PxMat44::createIdentity();
		context.renderResource = mRenderResource;
		renderer.renderResource(context);
	}
}


IofxSharedRenderDataSprite::IofxSharedRenderDataSprite(PxU32 instance)
	: IofxSharedRenderData(instance)
{
	spriteBuffer = NULL;
}

void IofxSharedRenderDataSprite::release()
{
	NxUserRenderResourceManager* rrm = NiGetApexSDK()->getUserRenderResourceManager();
	if (spriteBuffer != NULL)
	{
		rrm->releaseSpriteBuffer(*spriteBuffer);
		spriteBuffer = NULL;
		bufferIsMapped = false;
	}
}

void IofxSharedRenderDataSprite::alloc(IosObjectBaseData* objData, PxCudaContextManager* ctxman)
{
	if (useInterop ^ (ctxman != NULL))
	{
		//ignore this call
		return;
	}

	const IofxOutputDataSprite* outputData = static_cast<const IofxOutputDataSprite*>(objData->outputData);

	NxUserRenderResourceManager* rrm = NiGetApexSDK()->getUserRenderResourceManager();
	if (objData->outputSemantics != allocSemantics)
	{
		if (outputData->getVertexDesc().isTheSameAs(spriteBufferDesc) == false)
		{
			if (spriteBuffer != NULL)
			{
				rrm->releaseSpriteBuffer(*spriteBuffer);
				spriteBuffer = NULL;
				spriteBufferDesc.setDefaults();
			}

			if (objData->outputSemantics != 0)
			{
				NxUserRenderSpriteBufferDesc desc = outputData->getVertexDesc();
				desc.registerInCUDA = useInterop;
				desc.interopContext = useInterop ? ctxman : 0;

				PX_ASSERT(objData->outputDWords <= SPRITE_MAX_DWORDS_PER_OUTPUT);
				spriteBuffer = rrm->createSpriteBuffer(desc);
				if (spriteBuffer != NULL)
				{
					spriteBufferDesc = outputData->getVertexDesc();
				}
			}
		}
		allocSemantics = (spriteBuffer != NULL) ? objData->outputSemantics : 0;

		bufferIsMapped = false;
		objData->writeBufferCalled = false;
	}
}

bool IofxSharedRenderDataSprite::update(IosObjectBaseData* objData)
{
	if (useInterop)
	{
		//ignore this call in case of interop
		return false;
	}

	// IOFX manager will set writeBufferCalled = true when it writes directly to the mapped buffer
	if (objData->writeBufferCalled == false)
	{
		const IofxOutputDataSprite* outputData = static_cast<const IofxOutputDataSprite*>(objData->outputData);
		if (outputData->getDefaultBuffer().getCapacity() == 0 && outputData->getTextureCount() > 0)
		{
			PX_ASSERT(outputData->getTextureCount() == spriteBufferDesc.textureCount);
			for (PxU32 i = 0; i < outputData->getTextureCount(); ++i)
			{
				const IofxOutputBuffer& textureBuffer = outputData->getTextureBuffer(i);

				spriteBuffer->writeTexture(i, objData->numParticles, textureBuffer.getPtr(), textureBuffer.getSize());
			}
		}
		else
		{
			if (outputData->getDefaultBuffer().getSize() == 0)
			{
				return false;
			}

			PX_ASSERT(objData->outputSemantics == allocSemantics);

			PxU32* vertexBuffer = static_cast<PxU32*>( outputData->getDefaultBuffer().getPtr() );

			spriteBuffer->writeBuffer(vertexBuffer, 0, objData->numParticles);
		}
		objData->writeBufferCalled = true;
	}
	return true;
}

#if defined(APEX_CUDA_SUPPORT)
bool IofxSharedRenderDataSprite::getResourceList(PxU32& count, CUgraphicsResource* list)
{
	if (spriteBuffer)
	{
		if (spriteBufferDesc.textureCount > 0)
		{
			count = spriteBufferDesc.textureCount;
			return spriteBuffer->getInteropTextureHandleList(list);
		}
		count = 1;
		return spriteBuffer->getInteropResourceHandle(list[0]);
	}
	else
	{
		count = 0;
		return false;
	}
}

bool IofxSharedRenderDataSprite::resolveResourceList(CUdeviceptr& ptr, PxU32& arrayCount, CUarray* arrayList)
{
	if (spriteBuffer)
	{
		if (spriteBufferDesc.textureCount > 0)
		{
			CUgraphicsResource handleList[NxUserRenderSpriteBufferDesc::MAX_SPRITE_TEXTURES];
			if (spriteBuffer->getInteropTextureHandleList(handleList))
			{
				for (PxU32 i = 0; i < spriteBufferDesc.textureCount; ++i)
				{
					CUresult ret = cuGraphicsSubResourceGetMappedArray(&arrayList[i], handleList[i], spriteBufferDesc.textureDescs[i].arrayIndex, spriteBufferDesc.textureDescs[i].mipLevel);
					if (ret != CUDA_SUCCESS)
					{
						return false;
					}
				}
				ptr = 0;
				arrayCount = spriteBufferDesc.textureCount;
				return true;
			}
		}
		else
		{
			CUgraphicsResource handle;
			if (spriteBuffer->getInteropResourceHandle(handle))
			{
				size_t size = 0;
				CUresult ret = cuGraphicsResourceGetMappedPointer(&ptr, &size, handle);
				if (ret == CUDA_SUCCESS && size > 0)
				{
					arrayCount = 0;
					return true;
				}
			}
		}
	}
	return false;
}
#endif

}
}
} // namespace physx::apex
