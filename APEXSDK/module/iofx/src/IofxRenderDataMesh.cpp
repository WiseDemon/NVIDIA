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

void IofxActorRenderDataMesh::updateRenderResources(bool rewriteBuffers, void* userRenderData)
{
	PX_ASSERT(mRenderMeshActor != NULL);
	if (mRenderMeshActor == NULL)
	{
		return;
	}

	NxUserRenderInstanceBuffer* instanceBuffer = DYNAMIC_CAST(IofxSharedRenderDataMesh*)(mSharedRenderData)->getInstanceBuffer();
	if (mRenderMeshActor->getInstanceBuffer() != instanceBuffer)
	{
		mRenderMeshActor->setInstanceBuffer(instanceBuffer);
		//mSemantics = obj->allocSemantics;
	}

	PX_ASSERT( mRenderMeshActor->getInstanceBuffer() == instanceBuffer );

	const ObjectRange& range = mIofxActor->mRenderRange;
	mRenderMeshActor->setInstanceBufferRange(range.startIndex, range.objectCount);
	mRenderMeshActor->updateRenderResources(rewriteBuffers, userRenderData);
}

void IofxActorRenderDataMesh::dispatchRenderResources(NxUserRenderer& renderer)
{
	PX_ASSERT(mRenderMeshActor != NULL);
	if (mRenderMeshActor == NULL)
	{
		return;
	}

	mRenderMeshActor->dispatchRenderResources(renderer);
}


IofxSharedRenderDataMesh::IofxSharedRenderDataMesh(PxU32 instance)
	: IofxSharedRenderData(instance)
{
	instanceBuffer = NULL;
}

void IofxSharedRenderDataMesh::release()
{
	NxUserRenderResourceManager* rrm = NiGetApexSDK()->getUserRenderResourceManager();
	if (instanceBuffer != NULL)
	{
		rrm->releaseInstanceBuffer(*instanceBuffer);
		instanceBuffer = NULL;
		bufferIsMapped = false;
	}
}

void IofxSharedRenderDataMesh::alloc(IosObjectBaseData* objData, PxCudaContextManager* ctxman)
{
	if (useInterop ^ (ctxman != NULL))
	{
		//ignore this call
		return;
	}

	NxUserRenderResourceManager* rrm = NiGetApexSDK()->getUserRenderResourceManager();
	const IofxOutputDataMesh* outputData = static_cast<const IofxOutputDataMesh*>(objData->outputData);
	if (objData->outputSemantics != allocSemantics)
	{
		if (outputData->getVertexDesc().isTheSameAs(instanceBufferDesc) == false)
		{
			if (instanceBuffer != NULL)
			{
				rrm->releaseInstanceBuffer(*instanceBuffer);
				instanceBuffer = NULL;
				instanceBufferDesc.setDefaults();
			}

			if (objData->outputSemantics != 0)
			{
				NxUserRenderInstanceBufferDesc desc = outputData->getVertexDesc();
				desc.registerInCUDA = useInterop;
				desc.interopContext = useInterop ? ctxman : NULL;

				PX_ASSERT(objData->outputDWords <= MESH_MAX_DWORDS_PER_OUTPUT);
				instanceBuffer = rrm->createInstanceBuffer(desc);
				if (instanceBuffer != NULL)
				{
					instanceBufferDesc = outputData->getVertexDesc();
				}
			}
		}
		allocSemantics = (instanceBuffer != NULL) ? objData->outputSemantics : 0;

		bufferIsMapped = false;
		objData->writeBufferCalled = false;
	}
}

bool IofxSharedRenderDataMesh::update(IosObjectBaseData* objData)
{
	if (useInterop)
	{
		//ignore this call in case of interop
		return false;
	}

	// IOFX manager will set writeBufferCalled = true when it writes directly to the mapped buffer
	if (objData->writeBufferCalled == false)
	{
		if (objData->outputData->getDefaultBuffer().getSize() == 0)
		{
			return false;
		}
		const IofxOutputDataMesh* outputData = static_cast<const IofxOutputDataMesh*>(objData->outputData);

		PX_ASSERT(objData->outputSemantics == allocSemantics);

		PxU32* outputBuffer = static_cast<PxU32*>( outputData->getDefaultBuffer().getPtr() );
		instanceBuffer->writeBuffer(outputBuffer, 0, objData->numParticles);
		objData->writeBufferCalled = true;
	}
	return true;
}


#if defined(APEX_CUDA_SUPPORT)
bool IofxSharedRenderDataMesh::getResourceList(PxU32& count, CUgraphicsResource* list)
{
	if (instanceBuffer)
	{
		count = 1;
		return instanceBuffer->getInteropResourceHandle(list[0]);
	}
	else
	{
		count = 0;
		return false;
	}
}

bool IofxSharedRenderDataMesh::resolveResourceList(CUdeviceptr& ptr, PxU32& arrayCount, CUarray* arrayList)
{
	PX_UNUSED(arrayList);
	if (instanceBuffer)
	{
		CUgraphicsResource resourceHandle;
		if (instanceBuffer->getInteropResourceHandle(resourceHandle))
		{
			size_t size = 0;
			CUresult ret = cuGraphicsResourceGetMappedPointer(&ptr, &size, resourceHandle);
			if (ret == CUDA_SUCCESS && size > 0)
			{
				arrayCount = 0;
				return true;
			}
		}
	}
	return false;
}
#endif

}
}
} // namespace physx::apex
