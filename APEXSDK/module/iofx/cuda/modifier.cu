/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#define APEX_CUDA_STORAGE_NAME modifierStorage
#include "include/common.h"
#include "common.cuh"

#include "../public/NxModifierDefs.h"

using namespace physx::apex;
using namespace physx::apex::iofx;
#include "include/modifier.h"


// Modifiers

#define MODIFIER_DECL __device__
#define CURVE_TYPE Curve
#define EVAL_CURVE(curve, value) curve.evaluate(KERNEL_CONST_STORAGE, value)
#define PARAMS_NAME(name) name ## ParamsGPU

#include "../include/ModifierSrc.h"

#undef MODIFIER_DECL
#undef CURVE_TYPE
#undef EVAL_CURVE
#undef PARAMS_NAME


#define MODIFIER_LIST_BEG(usage) \
INPLACE_TEMPL_VA_ARGS_DEF(bool spawn, typename Input, typename PublicState, typename PrivateState) \
__device__ void runModifiers(Usage2Type< usage > , const ModifierList& list, const ModifierCommonParams& commonParams, const Input& input, PublicState& pubState, PrivateState& privState, physx::RandState& randState) \
{ \
	typedef Usage2Type< usage > UsageType; \
	const physx::PxU32 listCount = list.getSize(); \
	for (unsigned int i = 0; i < listCount; ++i) \
	{ \
		ModifierListElem listElem; \
		list.fetchElem(KERNEL_CONST_STORAGE, listElem, i); \

#define MODIFIER_LIST_ELEM(name) \
		if (listElem.type == physx::apex::ModifierType_##name) \
		{ \
			name##ParamsGPU params; \
			listElem.paramsHandle.fetch(KERNEL_CONST_STORAGE, params); \
			modifier##name INPLACE_TEMPL_VA_ARGS_VAL(spawn, UsageType::usage) (params, input, pubState, privState, commonParams, randState); \
		} \
		else \

#define MODIFIER_LIST_END() \
		{ \
		} \
	} \
} \

template <int U>
struct Usage2Type
{
	static const int usage = U;
};

//Sprite modifiers
MODIFIER_LIST_BEG(physx::apex::ModifierUsage_Sprite)
#define _MODIFIER_SPRITE(name) MODIFIER_LIST_ELEM(name)
#include "../include/ModifierList.h"
MODIFIER_LIST_END()

//Mesh modifiers
MODIFIER_LIST_BEG(physx::apex::ModifierUsage_Mesh)
#define _MODIFIER_MESH(name) MODIFIER_LIST_ELEM(name)
#include "../include/ModifierList.h"
MODIFIER_LIST_END()


INPLACE_TEMPL_VA_ARGS_DEF(
	int Usage, typename Input, typename PublicState, typename PrivateState,
	typename InputArgs, typename PrivateStateArgs, typename OutputLayout
)
__device__ void modifiersKernel(
	unsigned int outputCount, unsigned int OutputDWords,
	unsigned int inStateOffset, unsigned int outStateOffset,
	InplaceHandle<ClientParamsHandleArray> clientParamsHandleArrayHandle,
	ModifierCommonParams commonParams,
	unsigned int* g_sortedActorIDs, unsigned int* g_sortedStateIDs, unsigned int* g_outStateToInput,
	InputArgs inputArgs, PrivateStateArgs privStateArgs, physx::PxF32* g_stateSpawnScale,
	PRNGInfo rand, unsigned int* g_outputBuffer, OutputLayout outputLayout
)
{
	unsigned int idx = threadIdx.x;

	const unsigned int BlockSize = blockDim.x;
	const unsigned int Pitch = BlockSize + (NUM_BANKS + OutputDWords-1) / OutputDWords;
	extern __shared__ volatile unsigned int sdata[]; //size = (BlockSize + NUM_BANKS) * outputDWords;

	__shared__ physx::apex::LCG_PRNG randBlock;
	if (idx == 0) {
		randBlock = rand.g_randBlock[blockIdx.x];
	}

	for (unsigned int outputBeg = BlockSize * blockIdx.x; outputBeg < outputCount; outputBeg += BlockSize*gridDim.x)
	{
		physx::apex::LCG_PRNG randVal = (idx == 0 ? randBlock : rand.randThread);
		randVal = randScanBlock(randVal, sdata, sdata + BlockSize*2);

		unsigned int currSeed = randVal(rand.seed);
		if (idx == 0) {
			randBlock *= rand.randGrid;
		}
		__syncthreads();

		const unsigned int outputEnd = min(outputBeg + BlockSize, outputCount);
		const unsigned int outputID = outputBeg + idx;
		if (outputID < outputEnd)
		{
			unsigned int stateID = (g_sortedStateIDs[ outputID ] & STATE_ID_MASK);
			// stateID should be < maxStateID
			unsigned int inputID = tex1Dfetch( KERNEL_TEX_REF(InStateToInput), stateID );
			// inputID should be < maxInputID
			bool isNewParticle = ((inputID & NiIosBufferDesc::NEW_PARTICLE_FLAG) != 0);
			inputID &= ~NiIosBufferDesc::NEW_PARTICLE_FLAG;

			NiIofxActorID actorID;
			actorID.value = tex1Dfetch( KERNEL_TEX_REF(ActorIDs), inputID );

			if (actorID.getVolumeID() != NiIofxActorID::NO_VOLUME)
			{
				ClientParamsHandleArray clientParamsHandleArray;
				clientParamsHandleArrayHandle.fetch(KERNEL_CONST_STORAGE, clientParamsHandleArray);

				InplaceHandle<ClientParams> clientParamsHanlde;
				clientParamsHandleArray.fetchElem(KERNEL_CONST_STORAGE, clientParamsHanlde, actorID.getActorClassID());

				ClientParams clientParams;
				clientParamsHanlde.fetch(KERNEL_CONST_STORAGE, clientParams);

				AssetParams assetParams;
				clientParams.assetParamsHandle.fetch(KERNEL_CONST_STORAGE, assetParams);


				//prepare input
				Input		input;
				InputArgs::read(inputArgs, input, inputID, commonParams);

				//prepare state
				PublicState  pubState;
				PrivateState privState;

				physx::PxF32 spawnScale = isNewParticle ? clientParams.objectScale : tex1Dfetch( KERNEL_TEX_REF(StateSpawnScale), inStateOffset + stateID );

				//always run spawn modifiers
				PublicState::initDefault(pubState, spawnScale);
				PrivateState::initDefault(privState);

				unsigned int spawnSeed = isNewParticle ? currSeed : tex1Dfetch( KERNEL_TEX_REF(StateSpawnSeed), inStateOffset + stateID );
				RandState spawnRandState( spawnSeed );
				runModifiers INPLACE_TEMPL_VA_ARGS_VAL(true) (Usage2Type<Usage>(), assetParams.spawnModifierList, commonParams, input, pubState, privState, spawnRandState);

				if (!isNewParticle)
				{
					//read private state
					PrivateStateArgs::read(privStateArgs, privState, inStateOffset + stateID);
				}

				//run continuous modifiers
				RandState currRandState( currSeed );
				runModifiers INPLACE_TEMPL_VA_ARGS_VAL(false) (Usage2Type<Usage>(), assetParams.continuousModifierList, commonParams, input, pubState, privState, currRandState);

				//write state
				g_stateSpawnScale[ outStateOffset + outputID ] = spawnScale;
				rand.g_stateSpawnSeed[ outStateOffset + outputID ] = spawnSeed;
				PrivateStateArgs::write(privStateArgs, privState, outStateOffset + outputID);

				//write output to Output
				outputLayout.write(sdata, idx, Pitch, input, pubState, outputID);
			}
			g_outStateToInput[ outputID ] = inputID;
		}
		__syncthreads();

		if (g_outputBuffer != 0)
		{
			const unsigned int OutputBufferDwords = OutputDWords * (outputEnd - outputBeg);
			for (unsigned int pos = threadIdx.x; pos < OutputBufferDwords; pos += BlockSize)
			{
				g_outputBuffer[(outputBeg * OutputDWords) + pos] = sdata[(pos / OutputDWords) + (pos % OutputDWords)*Pitch];
			}
		}
		__syncthreads();
	}
}

// Sprite
namespace physx {
namespace apex {

struct SpriteInputArgs
{
	static __device__ void read(const SpriteInputArgs& args, SpriteInput& input, unsigned int pos, const ModifierCommonParams& commonParams)
	{
		float4 positionMass = tex1Dfetch(KERNEL_TEX_REF(PositionMass), pos);
		float4 velocityLife = tex1Dfetch(KERNEL_TEX_REF(VelocityLife), pos);

		input.position.x = positionMass.x;
		input.position.y = positionMass.y;
		input.position.z = positionMass.z;
		input.mass       = positionMass.w;

		input.velocity.x = velocityLife.x;
		input.velocity.y = velocityLife.y;
		input.velocity.z = velocityLife.z;
		input.liferemain = velocityLife.w;

		input.density    = commonParams.inputHasDensity ? tex1Dfetch(KERNEL_TEX_REF(Density), pos) : 0;

		input.userData   = commonParams.inputHasUserData ? tex1Dfetch(KERNEL_TEX_REF(UserData), pos) : 0;
	}
};

__device__ unsigned int floatFlip(float f)
{
    unsigned int i = __float_as_int(f);
	unsigned int mask = -int(i >> 31) | 0x80000000;
	return i ^ mask;
}


__device__ void SpritePrivateStateArgs::read(const SpritePrivateStateArgs& args, SpritePrivateState& state, unsigned int pos)
{
	IofxSlice slice0 = uint4_to_IofxSlice(tex1Dfetch(KERNEL_TEX_REF(SpritePrivState0), pos));

	// Slice 0 (underused)
	state.rotation = __int_as_float(slice0.x);
	state.scale    = __int_as_float(slice0.y);
}
__device__ void SpritePrivateStateArgs::write(SpritePrivateStateArgs& args, const SpritePrivateState& state, unsigned int pos)
{
	IofxSlice slice0;

	// Slice 0 (underused)
	slice0.x = __float_as_int(state.rotation);
	slice0.y = __float_as_int(state.scale);

	args.g_state[0][pos] = slice0;
}

// Mesh

struct MeshInputArgs
{
	static __device__ void read(const MeshInputArgs& args, MeshInput& input, unsigned int pos, const ModifierCommonParams& commonParams)
	{
		float4 positionMass         = tex1Dfetch(KERNEL_TEX_REF(PositionMass), pos);
		float4 velocityLife         = tex1Dfetch(KERNEL_TEX_REF(VelocityLife), pos);
		float4 collisionNormalFlags = commonParams.inputHasCollision ? tex1Dfetch(KERNEL_TEX_REF(CollisionNormalFlags), pos) : make_float4(0, 0, 0, 0);

		input.position.x = positionMass.x;
		input.position.y = positionMass.y;
		input.position.z = positionMass.z;
		input.mass       = positionMass.w;

		input.velocity.x = velocityLife.x;
		input.velocity.y = velocityLife.y;
		input.velocity.z = velocityLife.z;
		input.liferemain = velocityLife.w;

		input.density    = commonParams.inputHasDensity ? tex1Dfetch(KERNEL_TEX_REF(Density), pos) : 0;

		input.collisionNormal.x = collisionNormalFlags.x;
		input.collisionNormal.y = collisionNormalFlags.y;
		input.collisionNormal.z = collisionNormalFlags.z;
		input.collisionFlags    = __float_as_int(collisionNormalFlags.w);

		input.userData   = commonParams.inputHasUserData ? tex1Dfetch(KERNEL_TEX_REF(UserData), pos) : 0;
	}
};


__device__ void MeshPrivateStateArgs::read(const MeshPrivateStateArgs& args, MeshPrivateState& state, unsigned int pos)
{
	IofxSlice slice0 = uint4_to_IofxSlice(tex1Dfetch(KERNEL_TEX_REF(MeshPrivState0), pos)),
		slice1 = uint4_to_IofxSlice(tex1Dfetch(KERNEL_TEX_REF(MeshPrivState1), pos)),
		slice2 = uint4_to_IofxSlice(tex1Dfetch(KERNEL_TEX_REF(MeshPrivState2), pos));

	// Slice 0
	state.rotation(0,0) = __int_as_float(slice0.x);
	state.rotation(0,1) = __int_as_float(slice0.y);
	state.rotation(0,2) = __int_as_float(slice0.z);
	state.rotation(1,0) = __int_as_float(slice0.w);

	// Slice 1
	state.rotation(1,1) = __int_as_float(slice1.x);
	state.rotation(1,2) = __int_as_float(slice1.y);
	state.rotation(2,0) = __int_as_float(slice1.z);
	state.rotation(2,1) = __int_as_float(slice1.w);

	// Slice 2 (underused)
	state.rotation(2,2) = __int_as_float(slice2.x);
}
__device__ void MeshPrivateStateArgs::write(MeshPrivateStateArgs& args, const MeshPrivateState& state, unsigned int pos)
{
	IofxSlice slice0, slice1, slice2;
	
	// Slice 0
	slice0.x = __float_as_int(state.rotation(0,0));
	slice0.y = __float_as_int(state.rotation(0,1));
	slice0.z = __float_as_int(state.rotation(0,2));
	slice0.w = __float_as_int(state.rotation(1,0));

	// Slice 1
	slice1.x = __float_as_int(state.rotation(1,1));
	slice1.y = __float_as_int(state.rotation(1,2));
	slice1.z = __float_as_int(state.rotation(2,0));
	slice1.w = __float_as_int(state.rotation(2,1));

	// Slice 2 (underused)
	slice2.x = __float_as_int(state.rotation(2,2));

	args.g_state[0][pos] = slice0;
	args.g_state[1][pos] = slice1;
	args.g_state[2][pos] = slice2;
}


}} // namespace apex

BOUND_S2_KERNEL_BEG(spriteModifiersKernel,
	((unsigned int, inStateOffset))((unsigned int, outStateOffset))
	((InplaceHandle<ClientParamsHandleArray>, clientParamsHandleArrayHandle))
	((ModifierCommonParams, commonParams))
	((unsigned int*, g_sortedActorIDs))((unsigned int*, g_sortedStateIDs))((unsigned int*, g_outStateToInput))
	((SpritePrivateStateArgs, privStateArgs))((physx::PxF32*, g_stateSpawnScale))
	((PRNGInfo, rand))((unsigned int*, g_outputBuffer))
	((InplaceHandle<SpriteOutputLayout>, outputLayoutHandle))
)
	SpriteInputArgs inputArgs;

	SpriteOutputLayout outputLayout;
	outputLayoutHandle.fetch(KERNEL_CONST_STORAGE, outputLayout);
	unsigned int OutputDWords = (outputLayout.stride >> 2);

	modifiersKernel INPLACE_TEMPL_VA_ARGS_VAL(ModifierUsage_Sprite, 
		SpriteInput, SpritePublicState, SpritePrivateState, 
		SpriteInputArgs, SpritePrivateStateArgs, SpriteOutputLayout)
	(
		_threadCount, OutputDWords,
		inStateOffset, outStateOffset,
		clientParamsHandleArrayHandle,
		commonParams,
		g_sortedActorIDs, g_sortedStateIDs, g_outStateToInput,
		inputArgs, privStateArgs, g_stateSpawnScale,
		rand, g_outputBuffer, outputLayout
	);
BOUND_S2_KERNEL_END()


BOUND_S2_KERNEL_BEG(spriteTextureModifiersKernel,
	((unsigned int, inStateOffset))((unsigned int, outStateOffset))
	((InplaceHandle<ClientParamsHandleArray>, clientParamsHandleArrayHandle))
	((ModifierCommonParams, commonParams))
	((unsigned int*, g_sortedActorIDs))((unsigned int*, g_sortedStateIDs))((unsigned int*, g_outStateToInput))
	((SpritePrivateStateArgs, privStateArgs))((physx::PxF32*, g_stateSpawnScale))
	((PRNGInfo, rand))((SpriteTextureOutputLayout, outputLayout))
)
	SpriteInputArgs inputArgs;

	modifiersKernel INPLACE_TEMPL_VA_ARGS_VAL(ModifierUsage_Sprite, 
		SpriteInput, SpritePublicState, SpritePrivateState, 
		SpriteInputArgs, SpritePrivateStateArgs, SpriteTextureOutputLayout)
	(
		_threadCount, 1,
		inStateOffset, outStateOffset,
		clientParamsHandleArrayHandle,
		commonParams,
		g_sortedActorIDs, g_sortedStateIDs, g_outStateToInput,
		inputArgs, privStateArgs, g_stateSpawnScale,
		rand, 0, outputLayout
	);
BOUND_S2_KERNEL_END()

BOUND_S2_KERNEL_BEG(meshModifiersKernel,
	((unsigned int, inStateOffset))((unsigned int, outStateOffset))
	((InplaceHandle<ClientParamsHandleArray>, clientParamsHandleArrayHandle))
	((ModifierCommonParams, commonParams))
	((unsigned int*, g_sortedActorIDs))((unsigned int*, g_sortedStateIDs))((unsigned int*, g_outStateToInput))
	((MeshPrivateStateArgs, privStateArgs))((physx::PxF32*, g_stateSpawnScale))
	((PRNGInfo, rand))((unsigned int*, g_outputBuffer))
	((InplaceHandle<MeshOutputLayout>, outputLayoutHandle))
)
	MeshInputArgs inputArgs;

	MeshOutputLayout outputLayout;
	outputLayoutHandle.fetch(KERNEL_CONST_STORAGE, outputLayout);
	unsigned int OutputDWords = (outputLayout.stride >> 2);

	modifiersKernel INPLACE_TEMPL_VA_ARGS_VAL(ModifierUsage_Mesh,
		MeshInput, MeshPublicState, MeshPrivateState,
		MeshInputArgs, MeshPrivateStateArgs, MeshOutputLayout)
	(
		_threadCount, OutputDWords,
		inStateOffset, outStateOffset,
		clientParamsHandleArrayHandle,
		commonParams,
		g_sortedActorIDs, g_sortedStateIDs, g_outStateToInput,
		inputArgs, privStateArgs, g_stateSpawnScale,
		rand, g_outputBuffer, outputLayout
	);
BOUND_S2_KERNEL_END()
