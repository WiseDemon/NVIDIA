/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __IOFX_MANAGER_GPU_H__
#define __IOFX_MANAGER_GPU_H__

#include "NxApex.h"
#include "IofxManager.h"
#include "IofxSceneGPU.h"

#if defined(APEX_CUDA_SUPPORT)
#include "ApexCudaWrapper.h"
#include "PxCudaContextManager.h"
#endif

#include "ModifierData.h"
#include "IosObjectData.h"

namespace physx
{
namespace apex
{

class NiApexScene;
class NxModifier;

namespace iofx
{

class IofxAsset;
class IofxScene;
class ModifierParamsMapperGPU;

/* Class which manages a per-IOS CUDA IOFX pipeline */
class IofxManagerGPU : public CudaPipeline, public physx::UserAllocated
{
public:
	IofxManagerGPU(NiApexScene& scene, const NiIofxManagerDesc& desc, IofxManager&, const ApexMirroredPlace::Enum defaultPlace = ApexMirroredPlace::GPU);
	~IofxManagerGPU();

	void release();
	virtual void submitTasks();
	virtual void fetchResults();
	PxTaskID IofxManagerGPU::launchGpuTasks();
	void launchPrep();

	IofxManagerClient* createClient(IofxAssetSceneInst* assetSceneInst, PxU32 actorClassID, const NiIofxManagerClient::Params& params);
	IofxAssetSceneInst* createAssetSceneInst(IofxAsset* asset, PxU32 semantics);

	bool swapObjectData();


	IofxManager&        mManager;
	IofxScene&          mIofxScene;

	bool				cudaLaunch(CUstream stream, int kernelIndex);
	void				cudaLaunchRadixSort(CUstream stream, unsigned int numElements, unsigned int keyBits, unsigned int startBit, bool useSyncKernels);

	PxU32				mCurSeed;
	PxU32*				mTargetBufDevPtr;
	PxU32               mCountActorIDs;
	PxU32               mNumberVolumes;
	PxU32				mNumberActorClasses;
	PxU32				mOutputDWords;
	bool                mEmptySimulation;
	PxTask*		mTaskLaunch;

	ApexCudaConstMemGroup					mVolumeConstMemGroup;
	InplaceHandle<VolumeParamsArray>		mVolumeParamsArrayHandle;
	InplaceHandle<ActorClassIDBitmapArray>	mActorClassIDBitmapArrayHandle;

	ApexCudaConstMemGroup					mRemapConstMemGroup;
	InplaceHandle<ActorIDRemapArray>		mActorIDRemapArrayHandle;

	ApexCudaConstMemGroup					mModifierConstMemGroup;
	InplaceHandle<ClientParamsHandleArray>	mClientParamsHandleArrayHandle;
	InplaceHandle<SpriteOutputLayout>		mSpriteOutputLayoutHandle;
	InplaceHandle<MeshOutputLayout>			mMeshOutputLayoutHandle;

	physx::PxGpuCopyDescQueue mCopyQueue;

	const ApexMirroredPlace::Enum mDefaultPlace;

	ApexMirroredArray<physx::PxF32>	mCuSpawnScale;
	ApexMirroredArray<physx::PxU32>	mCuSpawnSeed;

	ApexMirroredArray<LCG_PRNG>		mCuBlockPRNGs;

	// sprite sorting, then actor ID sorting
	ApexMirroredArray<physx::PxU32>	mCuSortedActorIDs;
	ApexMirroredArray<physx::PxU32>	mCuSortedStateIDs;
	ApexMirroredArray<physx::PxU32>	mCuSortTempKeys;
	ApexMirroredArray<physx::PxU32>	mCuSortTempValues;
	ApexMirroredArray<physx::PxU32>	mCuSortTemp;

	ApexMirroredArray<physx::PxU32>	mCuActorStart;
	ApexMirroredArray<physx::PxU32>	mCuActorEnd;
	ApexMirroredArray<physx::PxU32>	mCuActorVisibleEnd;

	ApexMirroredArray<PxVec4>		mCuMinBounds;
	ApexMirroredArray<PxVec4>		mCuMaxBounds;
	ApexMirroredArray<PxVec4>		mCuTempMinBounds;
	ApexMirroredArray<PxVec4>		mCuTempMaxBounds;
	ApexMirroredArray<PxU32>		mCuTempActorIDs;
	LCG_PRNG						mRandThreadLeap;
	LCG_PRNG						mRandGridLeap;

	bool							mOutputToBuffer;

	IofxManager::InteropState		mInteropStateCopy;


	class OutputBuffer
	{
		physx::PxCudaBuffer* mGpuBuffer;

	public:
		OutputBuffer()
		{
			mGpuBuffer = 0;
		}
		~OutputBuffer()
		{
			release();
		}

		PX_INLINE bool isValid() const
		{
			return (mGpuBuffer != 0);
		}

		PX_INLINE void* getGpuPtr() const
		{
			return (mGpuBuffer != 0) ? reinterpret_cast<void*>(mGpuBuffer->getPtr()) : 0;
		}

		void release()
		{
			if (mGpuBuffer != 0)
			{
				mGpuBuffer->free();
				mGpuBuffer = 0;
			}
		}

		bool realloc(size_t capacity, physx::PxCudaContextManager* ctx)
		{
			if (mGpuBuffer != 0 && mGpuBuffer->getSize() >= capacity) 
			{
				return false;
			}
			release();

			mGpuBuffer = ctx->getMemoryManager()->alloc(
			                physx::PxCudaBufferType(physx::PxCudaBufferMemorySpace::T_GPU, physx::PxCudaBufferFlags::F_READ_WRITE),
			                capacity);
			PX_ASSERT(mGpuBuffer != 0);
			return true;
		}

		void copyDeviceToHostQ(const IofxOutputBuffer& outputBuffer, physx::PxGpuCopyDescQueue& copyQueue)
		{
			PxGpuCopyDesc desc;
			desc.type = PxGpuCopyDesc::DeviceToHost;
			desc.bytes = outputBuffer.getSize();
			desc.source = reinterpret_cast<size_t>( getGpuPtr() );
			desc.dest = reinterpret_cast<size_t>( outputBuffer.getPtr() );

			copyQueue.enqueue(desc);
		}

	};

	class TextureBuffer
	{
		OutputBuffer	mOutputBuffer;

		NxRenderSpriteTextureLayout::Enum	mLayout;
		physx::PxU32						mWidth;
		physx::PxU32						mHeight;
		physx::PxU32						mElemSize;
		physx::PxU32						mPitch;

	public:
		TextureBuffer()
		{
		}
		~TextureBuffer()
		{
		}

		PX_INLINE NxRenderSpriteTextureLayout::Enum getLayout() const
		{
			return mLayout;
		}
		PX_INLINE physx::PxU32 getPitch() const
		{
			PX_ASSERT(mOutputBuffer.isValid());
			return mPitch;
		}
		PX_INLINE physx::PxU32 getWidth() const
		{
			PX_ASSERT(mOutputBuffer.isValid());
			return mWidth;
		}
		PX_INLINE physx::PxU32 getHeight() const
		{
			PX_ASSERT(mOutputBuffer.isValid());
			return mHeight;
		}
		PX_INLINE physx::PxU8* getPtr() const
		{
			PX_ASSERT(mOutputBuffer.isValid());
			return reinterpret_cast<physx::PxU8*>(mOutputBuffer.getGpuPtr());
		}

		void release()
		{
			mOutputBuffer.release();
		}

		void alloc(physx::PxCudaContextManager* ctx, const NxUserRenderSpriteTextureDesc& textureDesc)
		{
			mLayout = textureDesc.layout;
			mWidth = textureDesc.width;
			mHeight = textureDesc.height;
			mElemSize = NxRenderDataFormat::getFormatDataSize( NxRenderSpriteTextureLayout::getLayoutFormat(textureDesc.layout) );
			mPitch = textureDesc.pitchBytes;
			PX_ASSERT(mPitch >= mWidth * mElemSize);

			const size_t size = mPitch * mHeight;
			mOutputBuffer.realloc(size, ctx);
		}
		void alloc(physx::PxCudaContextManager* ctx, const NxUserRenderSpriteTextureDesc& textureDesc, CUarray cuArray)
		{
			PX_ASSERT(cuArray != 0);
			CUDA_ARRAY3D_DESCRIPTOR cuArrayDesc;
			CUT_SAFE_CALL(cuArray3DGetDescriptor(&cuArrayDesc, cuArray));

			physx::PxU32 formatSize = 0;
			switch (cuArrayDesc.Format)
			{
			case CU_AD_FORMAT_UNSIGNED_INT8:
			case CU_AD_FORMAT_SIGNED_INT8:
				formatSize = 1;
				break;
			case CU_AD_FORMAT_UNSIGNED_INT16:
			case CU_AD_FORMAT_SIGNED_INT16:
			case CU_AD_FORMAT_HALF:
				formatSize = 2;
				break;
			case CU_AD_FORMAT_UNSIGNED_INT32:
			case CU_AD_FORMAT_SIGNED_INT32:
			case CU_AD_FORMAT_FLOAT:
				formatSize = 4;
				break;
			default:
				PX_ALWAYS_ASSERT();
			}

			mLayout = textureDesc.layout;
			mWidth = (physx::PxU32)cuArrayDesc.Width;
			mHeight = (physx::PxU32)cuArrayDesc.Height;
			mElemSize = formatSize * cuArrayDesc.NumChannels;
			mPitch = APEX_CUDA_ALIGN_UP(mWidth * mElemSize, APEX_CUDA_MEM_ALIGNMENT);

			const size_t size = mPitch * mHeight;
			mOutputBuffer.realloc(size, ctx);
		}

		void copyToArray(CUarray cuArray, CUstream stream, PxU32 numParticles)
		{
			PX_ASSERT(mOutputBuffer.isValid());
			PX_ASSERT(cuArray != 0);

			CUDA_MEMCPY2D desc;
			desc.srcXInBytes = desc.srcY = 0;
			desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
			desc.srcDevice = reinterpret_cast<CUdeviceptr>(mOutputBuffer.getGpuPtr());
			desc.srcPitch = mPitch;

			desc.dstXInBytes = desc.dstY = 0;
			desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
			desc.dstArray = cuArray;

			desc.WidthInBytes = mWidth * mElemSize;
			desc.Height = PxMin((numParticles + mWidth-1) / mWidth, mHeight);

			CUT_SAFE_CALL(cuMemcpy2DAsync(&desc, stream));
		}
		void copyDeviceToHostQ(const IofxOutputBuffer& buff, physx::PxGpuCopyDescQueue& copyQueue)
		{
			mOutputBuffer.copyDeviceToHostQ(buff, copyQueue);
		}
	};

	OutputBuffer					mTargetOutputBuffer;
	physx::PxU32					mTargetTextureCount;
	TextureBuffer					mTargetTextureBufferList[NxUserRenderSpriteBufferDesc::MAX_SPRITE_TEXTURES];
	CUarray							mTargetCuArrayList[NxUserRenderSpriteBufferDesc::MAX_SPRITE_TEXTURES];
};

}
}
} // namespace apex

#endif // __IOFX_ACTOR_GPU_H__
