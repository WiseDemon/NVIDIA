/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#ifndef __MODIFIER_DATA_H__
#define __MODIFIER_DATA_H__

#include "PsShare.h"
#include "foundation/PxVec3.h"
#include <PxMat33Legacy.h>
#include "InplaceTypes.h"
#include "InplaceStorage.h"
#include "RandState.h"

#include "NxUserRenderInstanceBufferDesc.h"
#include "NxUserRenderSpriteBufferDesc.h"

namespace physx
{
namespace apex
{
namespace iofx
{

#ifndef __CUDACC__
PX_INLINE float saturate(float x)
{
	return (x < 0.0f) ? 0.0f : (1.0f < x) ? 1.0f : x;
}
#endif

// output color is NxRenderDataFormat::B8G8R8A8
#define FLT_TO_BYTE(x) ( (unsigned int)(saturate(physx::PxAbs(x)) * 255) )
#define MAKE_COLOR_UBYTE4(r, g, b, a) ( ((r) << 16) | ((g) << 8) | ((b) << 0) | ((a) << 24) )


class IosObjectBaseData;

//struct ModifierCommonParams
#define INPLACE_TYPE_STRUCT_NAME ModifierCommonParams
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(InplaceBool,		inputHasCollision) \
	INPLACE_TYPE_FIELD(InplaceBool,		inputHasDensity) \
	INPLACE_TYPE_FIELD(InplaceBool,		inputHasUserData) \
	INPLACE_TYPE_FIELD(physx::PxVec3,	upVector) \
	INPLACE_TYPE_FIELD(physx::PxVec3,	eyePosition) \
	INPLACE_TYPE_FIELD(physx::PxVec3,	eyeDirection) \
	INPLACE_TYPE_FIELD(physx::PxVec3,	eyeAxisX) \
	INPLACE_TYPE_FIELD(physx::PxVec3,	eyeAxisY) \
	INPLACE_TYPE_FIELD(physx::PxF32,	zNear) \
	INPLACE_TYPE_FIELD(physx::PxF32,	deltaTime)
#include INPLACE_TYPE_BUILD()


// Mesh structs
struct MeshInput
{
	physx::PxVec3	position;
	physx::PxF32	mass;
	physx::PxVec3	velocity;
	physx::PxF32	liferemain;
	physx::PxF32	density;
	physx::PxVec3	collisionNormal;
	unsigned int	collisionFlags;
	physx::PxU32	userData;

	PX_INLINE void load(const IosObjectBaseData& objData, physx::PxU32 pos);
};

struct MeshPublicState
{
	physx::PxMat33Legacy	rotation;
	physx::PxVec3			scale;

	float	color[4];

	static APEX_CUDA_CALLABLE PX_INLINE void initDefault(MeshPublicState& state, PxF32 objectScale)
	{
		state.rotation.setIdentity();
		state.scale = physx::PxVec3(objectScale);

		state.color[0] = 1.0f;
		state.color[1] = 1.0f;
		state.color[2] = 1.0f;
		state.color[3] = 1.0f;
	}
};

/* TODO: Private state size should be declared by each IOFX asset, so the IOS can allocate
 * the private buffer dynamically based on the IOFX assets used with the IOS.  Each asset would
 * in turn be given an offset for their private data in this buffer.
 */
struct MeshPrivateState
{
	physx::PxMat33Legacy			rotation;

	static APEX_CUDA_CALLABLE PX_INLINE void initDefault(MeshPrivateState& state)
	{
		state.rotation.setIdentity();
	}
};


//struct MeshOutputLayout
#define INPLACE_TYPE_STRUCT_NAME MeshOutputLayout
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxU32,	stride) \
	INPLACE_TYPE_FIELD_N(physx::PxU32,	offsets, NxRenderInstanceLayoutElement::NUM_SEMANTICS)
#define INPLACE_TYPE_STRUCT_LEAVE_OPEN 1
#include INPLACE_TYPE_BUILD()

#ifdef __CUDACC__
#define WRITE_TO_FLOAT(data) { *((volatile float*)(sdata + ((offset) >> 2) * pitch) + idx) = (data); offset += 4; }
#define WRITE_TO_UINT(data) { *((volatile unsigned int*)(sdata + ((offset) >> 2) * pitch + idx)) = (data); offset += 4; }

	__device__ PX_INLINE void write(volatile unsigned int* sdata, unsigned int idx, unsigned int pitch, const MeshInput& input, const MeshPublicState& state, unsigned int outputID) const
#else
#define WRITE_TO_FLOAT(data) { *(float*)(outputPtr + outputID * stride + offset) = (data); offset += 4; }
#define WRITE_TO_UINT(data) { *(unsigned int*)(outputPtr + outputID * stride + offset) = (data); offset += 4; }

	PX_INLINE void write(physx::PxU32 outputID, const MeshInput& input, const MeshPublicState& state, const physx::PxU8* outputPtr) const
#endif
	{
		physx::PxU32 offset;
		if ((offset = offsets[NxRenderInstanceLayoutElement::POSITION_FLOAT3]) != static_cast<physx::PxU32>(-1)) //POSITION: 3 dwords
		{
			WRITE_TO_FLOAT( input.position.x )
			WRITE_TO_FLOAT( input.position.y )
			WRITE_TO_FLOAT( input.position.z )
		}
		if ((offset = offsets[NxRenderInstanceLayoutElement::ROTATION_SCALE_FLOAT3x3]) != static_cast<physx::PxU32>(-1)) //ROTATION_SCALE: 9 dwords
		{
			physx::PxVec3 axis0 = state.rotation.getColumn(0) * state.scale.x;
			physx::PxVec3 axis1 = state.rotation.getColumn(1) * state.scale.y;
			physx::PxVec3 axis2 = state.rotation.getColumn(2) * state.scale.z;

			WRITE_TO_FLOAT( axis0.x )
			WRITE_TO_FLOAT( axis0.y )
			WRITE_TO_FLOAT( axis0.z )

			WRITE_TO_FLOAT( axis1.x )
			WRITE_TO_FLOAT( axis1.y )
			WRITE_TO_FLOAT( axis1.z )

			WRITE_TO_FLOAT( axis2.x )
			WRITE_TO_FLOAT( axis2.y )
			WRITE_TO_FLOAT( axis2.z )
		}
		if ((offset = offsets[NxRenderInstanceLayoutElement::VELOCITY_LIFE_FLOAT4]) != static_cast<physx::PxU32>(-1)) //VELOCITY: 3 dwords
		{
			WRITE_TO_FLOAT( input.velocity.x )
			WRITE_TO_FLOAT( input.velocity.y )
			WRITE_TO_FLOAT( input.velocity.z )
			WRITE_TO_FLOAT( input.liferemain )
		}
		if ((offset = offsets[NxRenderInstanceLayoutElement::DENSITY_FLOAT1]) != static_cast<physx::PxU32>(-1)) //DENSITY: 1 dword
		{
			WRITE_TO_FLOAT( input.density )
		}
		if ((offset = offsets[NxRenderInstanceLayoutElement::COLOR_BGRA8]) != static_cast<physx::PxU32>(-1)) //COLOR: 1 dword
		{
			WRITE_TO_UINT( MAKE_COLOR_UBYTE4( FLT_TO_BYTE(state.color[0]), 
											FLT_TO_BYTE(state.color[1]), 
											FLT_TO_BYTE(state.color[2]), 
											FLT_TO_BYTE(state.color[3]) ) )
		}
		if ((offset = offsets[NxRenderInstanceLayoutElement::COLOR_RGBA8]) != static_cast<physx::PxU32>(-1)) //COLOR: 1 dword
		{
			WRITE_TO_UINT( MAKE_COLOR_UBYTE4( FLT_TO_BYTE(state.color[2]), 
											FLT_TO_BYTE(state.color[1]), 
											FLT_TO_BYTE(state.color[0]), 
											FLT_TO_BYTE(state.color[3]) ) )
											
		}
		if ((offset = offsets[NxRenderInstanceLayoutElement::COLOR_FLOAT4]) != static_cast<physx::PxU32>(-1)) //COLOR_FLOAT4: 4 dword
		{
			WRITE_TO_FLOAT( state.color[0] )
			WRITE_TO_FLOAT( state.color[1] )
			WRITE_TO_FLOAT( state.color[2] )
			WRITE_TO_FLOAT( state.color[3] )
		}
		if ((offset = offsets[NxRenderInstanceLayoutElement::USER_DATA_UINT1]) != static_cast<physx::PxU32>(-1)) //USER_DATA: 1 dword
		{
			WRITE_TO_UINT( input.userData )
		}
		if ((offset = offsets[NxRenderInstanceLayoutElement::POSE_FLOAT3x4]) != static_cast<physx::PxU32>(-1)) //POSE: 12 dwords
		{
			physx::PxVec3 axis0 = state.rotation.getColumn(0) * state.scale.x;
			physx::PxVec3 axis1 = state.rotation.getColumn(1) * state.scale.y;
			physx::PxVec3 axis2 = state.rotation.getColumn(2) * state.scale.z;

			WRITE_TO_FLOAT( axis0.x )
			WRITE_TO_FLOAT( axis1.x )
			WRITE_TO_FLOAT( axis2.x )
			WRITE_TO_FLOAT( input.position.x )

			WRITE_TO_FLOAT( axis0.y )
			WRITE_TO_FLOAT( axis1.y )
			WRITE_TO_FLOAT( axis2.y )
			WRITE_TO_FLOAT( input.position.y )

			WRITE_TO_FLOAT( axis0.z )
			WRITE_TO_FLOAT( axis1.z )
			WRITE_TO_FLOAT( axis2.z )
			WRITE_TO_FLOAT( input.position.z )
		}
	}
#undef WRITE_TO_UINT
#undef WRITE_TO_FLOAT
};


// Sprite structs
struct SpriteInput
{
	physx::PxVec3	position;
	physx::PxF32	mass;
	physx::PxVec3	velocity;
	physx::PxF32	liferemain;
	physx::PxF32	density;
	physx::PxU32	userData;

	PX_INLINE void load(const IosObjectBaseData& objData, physx::PxU32 pos);
};


struct SpritePublicState
{
	physx::PxVec3	scale;
	float			subTextureId;
	float			rotation;

	float			color[4];

	static APEX_CUDA_CALLABLE PX_INLINE void initDefault(SpritePublicState& state, PxF32 objectScale)
	{
		state.scale = physx::PxVec3(objectScale);

		state.subTextureId = 0;
		state.rotation = 0;

		state.color[0] = 1.0f;
		state.color[1] = 1.0f;
		state.color[2] = 1.0f;
		state.color[3] = 1.0f;
	}
};

/* TODO: Private state size should be declared by each IOFX asset, so the IOS can allocate
 * the private buffer dynamically based on the IOFX assets used with the IOS.  Each asset would
 * in turn be given an offset for their private data in this buffer.
 */
struct SpritePrivateState
{
	float	rotation;
	float	scale;

	static APEX_CUDA_CALLABLE PX_INLINE void initDefault(SpritePrivateState& state)
	{
		state.rotation = 0.0f;
		state.scale = 1.0f;
	}
};

//struct SpriteOutputLayout
#define INPLACE_TYPE_STRUCT_NAME SpriteOutputLayout
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxU32,	stride) \
	INPLACE_TYPE_FIELD_N(physx::PxU32,	offsets, NxRenderSpriteLayoutElement::NUM_SEMANTICS)
#define INPLACE_TYPE_STRUCT_LEAVE_OPEN 1
#include INPLACE_TYPE_BUILD()

#ifdef __CUDACC__
#define WRITE_TO_FLOAT(data) { *((volatile float*)(sdata + ((offset) >> 2) * pitch) + idx) = (data); offset += 4; }
#define WRITE_TO_UINT(data) { *((volatile unsigned int*)(sdata + ((offset) >> 2) * pitch + idx)) = (data); offset += 4; }

	__device__ PX_INLINE void write(volatile unsigned int* sdata, unsigned int idx, unsigned int pitch, const SpriteInput& input, const SpritePublicState& state, unsigned int outputID) const
#else
#define WRITE_TO_FLOAT(data) { *(float*)(outputPtr + outputID * stride + offset) = (data); offset += 4; }
#define WRITE_TO_UINT(data) { *(unsigned int*)(outputPtr + outputID * stride + offset) = (data); offset += 4; }

	PX_INLINE void write(physx::PxU32 outputID, const SpriteInput& input, const SpritePublicState& state, const physx::PxU8* outputPtr) const
#endif
	{
		physx::PxU32 offset;
		if((offset = offsets[NxRenderSpriteLayoutElement::POSITION_FLOAT3]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_FLOAT( input.position.x )
			WRITE_TO_FLOAT( input.position.y )
			WRITE_TO_FLOAT( input.position.z )
		}
		if((offset = offsets[NxRenderSpriteLayoutElement::COLOR_BGRA8]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_UINT( MAKE_COLOR_UBYTE4(FLT_TO_BYTE(state.color[0]), 
											FLT_TO_BYTE(state.color[1]), 
											FLT_TO_BYTE(state.color[2]), 
											FLT_TO_BYTE(state.color[3])) )
		}
		if((offset = offsets[NxRenderSpriteLayoutElement::COLOR_RGBA8]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_UINT( MAKE_COLOR_UBYTE4(FLT_TO_BYTE(state.color[2]), 
											FLT_TO_BYTE(state.color[1]), 
											FLT_TO_BYTE(state.color[0]), 
											FLT_TO_BYTE(state.color[3])) )
		}
		if((offset = offsets[NxRenderSpriteLayoutElement::COLOR_FLOAT4]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_FLOAT( state.color[0] )
			WRITE_TO_FLOAT( state.color[1] )
			WRITE_TO_FLOAT( state.color[2] )
			WRITE_TO_FLOAT( state.color[3] )
		}
		if((offset = offsets[NxRenderSpriteLayoutElement::VELOCITY_FLOAT3]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_FLOAT( input.velocity.x )
			WRITE_TO_FLOAT( input.velocity.y )
			WRITE_TO_FLOAT( input.velocity.z )
		}
		if((offset = offsets[NxRenderSpriteLayoutElement::SCALE_FLOAT2]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_FLOAT( state.scale.x )
			WRITE_TO_FLOAT( state.scale.y )
		}
		if((offset = offsets[NxRenderSpriteLayoutElement::LIFE_REMAIN_FLOAT1]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_FLOAT( input.liferemain )
		}
		if((offset = offsets[NxRenderSpriteLayoutElement::DENSITY_FLOAT1]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_FLOAT( input.density )
		}
		if((offset = offsets[NxRenderSpriteLayoutElement::SUBTEXTURE_FLOAT1]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_FLOAT( state.subTextureId )
		}
		if((offset = offsets[NxRenderSpriteLayoutElement::ORIENTATION_FLOAT1]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_FLOAT( state.rotation )
		}
		if((offset = offsets[NxRenderSpriteLayoutElement::USER_DATA_UINT1]) != static_cast<physx::PxU32>(-1))
		{
			WRITE_TO_UINT( input.userData )
		}
	}
#undef WRITE_TO_UINT
#undef WRITE_TO_FLOAT
};

struct TextureOutputData
{
	physx::PxU16    layout;
	physx::PxU8     widthShift;
	physx::PxU8     pitchShift;
};

struct SpriteTextureOutputLayout
{
	physx::PxU32      textureCount;
	TextureOutputData textureData[4];
	physx::PxU8*      texturePtr[4];

#ifdef __CUDACC__
#define WRITE_TO_FLOAT4(e0, e1, e2, e3) { *(float4*)(ptr + (y << pitchShift) + (x << 4)) = make_float4(e0, e1, e2, e3); }
#define WRITE_TO_UINT(data) { *(unsigned int*)(ptr + (y << pitchShift) + (x << 2)) = data; }

	__device__ PX_INLINE void write(volatile unsigned int* sdata, unsigned int idx, unsigned int pitch, const SpriteInput& input, const SpritePublicState& state, unsigned int outputID) const
#else
#define WRITE_TO_FLOAT4(e0, e1, e2, e3) { *(physx::PxVec4*)(ptr + (y << pitchShift) + (x << 4)) = physx::PxVec4(e0, e1, e2, e3); }
#define WRITE_TO_UINT(data) { *(unsigned int*)(ptr + (y << pitchShift) + (x << 2)) = data; }

	PX_INLINE void write(unsigned int outputID, const SpriteInput& input, const SpritePublicState& state, const physx::PxU8*) const
#endif
	{
#define WRITE_TO_TEXTURE(N) \
		if (N < textureCount) \
		{ \
			physx::PxU32 y = (outputID >> textureData[N].widthShift); \
			physx::PxU32 x = outputID - (y << textureData[N].widthShift); \
			physx::PxU8  pitchShift = textureData[N].pitchShift; \
			physx::PxU8* ptr = texturePtr[N]; \
			switch (textureData[N].layout) \
			{ \
			case NxRenderSpriteTextureLayout::POSITION_FLOAT4: \
				WRITE_TO_FLOAT4( input.position.x, input.position.y, input.position.z, 1.0f ) \
				break; \
			case NxRenderSpriteTextureLayout::SCALE_ORIENT_SUBTEX_FLOAT4: \
				WRITE_TO_FLOAT4( state.scale.x, state.scale.y, state.rotation, state.subTextureId ) \
				break; \
			case NxRenderSpriteTextureLayout::COLOR_BGRA8: \
				WRITE_TO_UINT( MAKE_COLOR_UBYTE4( FLT_TO_BYTE(state.color[0]), FLT_TO_BYTE(state.color[1]), FLT_TO_BYTE(state.color[2]), FLT_TO_BYTE(state.color[3]) ) ) \
				break; \
			case NxRenderSpriteTextureLayout::COLOR_RGBA8: \
				WRITE_TO_UINT( MAKE_COLOR_UBYTE4( FLT_TO_BYTE(state.color[2]), FLT_TO_BYTE(state.color[1]), FLT_TO_BYTE(state.color[0]), FLT_TO_BYTE(state.color[3]) ) ) \
				break; \
			case NxRenderSpriteTextureLayout::COLOR_FLOAT4: \
				WRITE_TO_FLOAT4( state.color[0], state.color[1], state.color[2], state.color[3] ) \
				break; \
			} \
		}

		WRITE_TO_TEXTURE(0)
		WRITE_TO_TEXTURE(1)
		WRITE_TO_TEXTURE(2)
		WRITE_TO_TEXTURE(3)
	}
#undef WRITE_TO_TEXTURE
#undef WRITE_TO_UINT
#undef WRITE_TO_FLOAT4
};

//struct CurvePoint
#define INPLACE_TYPE_STRUCT_NAME CurvePoint
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(physx::PxF32,	x) \
	INPLACE_TYPE_FIELD(physx::PxF32,	y)
#define INPLACE_TYPE_STRUCT_LEAVE_OPEN 1
#include INPLACE_TYPE_BUILD()

	APEX_CUDA_CALLABLE PX_INLINE CurvePoint() : x(0.0f), y(0.0f) {}
	APEX_CUDA_CALLABLE PX_INLINE CurvePoint(physx::PxF32 _x, physx::PxF32 _y) : x(_x), y(_y) {}
};

APEX_CUDA_CALLABLE PX_INLINE physx::PxF32 lerpPoints(physx::PxF32 x, const CurvePoint& p0, const CurvePoint& p1)
{
	return ((x - p0.x) / (p1.x - p0.x)) * (p1.y - p0.y) + p0.y;
}


//struct Curve
#define INPLACE_TYPE_STRUCT_NAME Curve
#define INPLACE_TYPE_STRUCT_FIELDS \
	INPLACE_TYPE_FIELD(InplaceArray<CurvePoint>,	_pointArray)
#define INPLACE_TYPE_STRUCT_LEAVE_OPEN 1
#include INPLACE_TYPE_BUILD()

#ifndef __CUDACC__
	PX_INLINE void resize(InplaceStorage& storage, physx::PxU32 numPoints)
	{
		_pointArray.resize(storage, numPoints);
	}

	PX_INLINE void setPoint(InplaceStorage& storage, const CurvePoint& point, physx::PxU32 index)
	{
		_pointArray.updateElem(storage, point, index);
	}
#endif

	INPLACE_TEMPL_ARGS_DEF
	APEX_CUDA_CALLABLE PX_INLINE physx::PxF32 evaluate(INPLACE_STORAGE_ARGS_DEF, physx::PxF32 x) const
	{
		physx::PxU32 count = _pointArray.getSize();
		if (count == 0)
		{
			return 0.0f;
		}

		CurvePoint begPoint;
		_pointArray.fetchElem(INPLACE_STORAGE_ARGS_VAL, begPoint, 0);
		if (x <= begPoint.x)
		{
			return begPoint.y;
		}

		CurvePoint endPoint;
		_pointArray.fetchElem(INPLACE_STORAGE_ARGS_VAL, endPoint, count - 1);
		if (x >= endPoint.x)
		{
			return endPoint.y;
		}

		//do binary search
		unsigned int beg = 0;
		unsigned int end = count;
		while (beg < end)
		{
			unsigned int mid = beg + ((end - beg) >> 1);
			CurvePoint midPoint;
			_pointArray.fetchElem(INPLACE_STORAGE_ARGS_VAL, midPoint, mid);
			if (x < midPoint.x)
			{
				end = mid;
			}
			else
			{
				beg = mid + 1;
			}
		}
		beg = physx::PxMin<physx::PxU32>(beg, count - 1);
		CurvePoint point0, point1;
		_pointArray.fetchElem(INPLACE_STORAGE_ARGS_VAL, point0, beg - 1);
		_pointArray.fetchElem(INPLACE_STORAGE_ARGS_VAL, point1, beg);
		return lerpPoints(x, point0, point1);
	}
};

}
}
} // namespace apex

#endif /* __MODIFIER_DATA_H__ */
