/*
 * Copyright (c) 2008-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

#ifndef GU_EDGE_LIST_DATA_H
#define GU_EDGE_LIST_DATA_H

#include "PxSimpleTypes.h"
#include "CmPhysXCommon.h"

namespace physx
{
namespace Gu
{

/*!
NOTICE!

This is a data-code separated version of PxPhysics::EdgeList.

It is to be shared between high and low level code, so make sure both are recompiled
if any change is done here.
*/

// Flags
enum EdgeType
{
	PX_EDGE_UNDEFINED,

	PX_EDGE_BOUNDARY,		//!< Edge belongs to a single triangle
	PX_EDGE_INTERNAL,		//!< Edge belongs to exactly two triangles
	PX_EDGE_SINGULAR,		//!< Edge belongs to three or more triangles

	PX_EDGE_FORCE_DWORD	= 0x7fffffff
};

enum EdgeFlag
{
	PX_EDGE_ACTIVE	= (1<<0)
};


// Data


//! Basic edge-data
struct EdgeData
{
	PxU32	Ref0;	//!< First vertex reference	
	PxU32	Ref1;	//!< Second vertex reference
};
PX_COMPILE_TIME_ASSERT(sizeof(Gu::EdgeData) == 8);


//! Basic edge-data using 8-bit references
struct Edge8Data
{
	PxU8	Ref0;	//!< First vertex reference
	PxU8	Ref1;	//!< Second vertex reference
};
PX_COMPILE_TIME_ASSERT(sizeof(Gu::Edge8Data) == 2);


//! A count/offset pair = an edge descriptor
struct EdgeDescData
{
	PxU16	Flags;
	PxU16	Count;
	PxU32	Offset;
};
PX_COMPILE_TIME_ASSERT(sizeof(Gu::EdgeDescData) == 8);


//! Edge<->triangle mapping
struct EdgeTriangleData
{
	PxU32	mLink[3];
};
PX_COMPILE_TIME_ASSERT(sizeof(Gu::EdgeTriangleData) == 12);


struct EdgeListData
{
	// The edge list
	PxU32					mNbEdges;			//!< Number of edges in the list
	Gu::EdgeData*			mEdges;				//!< List of edges
	// Faces to edges
	PxU32					mNbFaces;			//!< Number of faces for which we have data
	Gu::EdgeTriangleData*	mEdgeFaces;			//!< Array of edge-triangles referencing mEdges
	// Edges to faces
	Gu::EdgeDescData*		mEdgeToTriangles;	//!< An EdgeDesc structure for each edge
	PxU32*					mFacesByEdges;		//!< A pool of face indices
};
#if defined(PX_X64)
PX_COMPILE_TIME_ASSERT(sizeof(Gu::EdgeListData) == 48);
#else
PX_COMPILE_TIME_ASSERT(sizeof(Gu::EdgeListData) == 24);
#endif


// Accessors

enum
{
	MSH_EDGE_LINK_MASK		= 0x0fffffff,
	MSH_ACTIVE_EDGE_MASK	= 0x80000000,
	MSH_ACTIVE_VERTEX_MASK	= 0x40000000
};

class EdgeTriangleAC
{
public:
	PX_INLINE static PxU32			GetEdge01(const Gu::EdgeTriangleData& data)					{ return data.mLink[0] & MSH_EDGE_LINK_MASK;	}
	PX_INLINE static PxU32			GetEdge12(const Gu::EdgeTriangleData& data)					{ return data.mLink[1] & MSH_EDGE_LINK_MASK;	}
	PX_INLINE static PxU32			GetEdge20(const Gu::EdgeTriangleData& data)					{ return data.mLink[2] & MSH_EDGE_LINK_MASK;	}
	PX_INLINE static PxU32			GetEdge(const Gu::EdgeTriangleData& data, PxU32 i)			{ return data.mLink[i] & MSH_EDGE_LINK_MASK;	}

	PX_INLINE static Ps::IntBool	HasActiveEdge01(const Gu::EdgeTriangleData& data)			{ return Ps::IntBool(data.mLink[0] & MSH_ACTIVE_EDGE_MASK);	}
	PX_INLINE static Ps::IntBool	HasActiveEdge12(const Gu::EdgeTriangleData& data)			{ return Ps::IntBool(data.mLink[1] & MSH_ACTIVE_EDGE_MASK);	}
	PX_INLINE static Ps::IntBool	HasActiveEdge20(const Gu::EdgeTriangleData& data)			{ return Ps::IntBool(data.mLink[2] & MSH_ACTIVE_EDGE_MASK);	}
	PX_INLINE static Ps::IntBool	HasActiveEdge(const Gu::EdgeTriangleData& data, PxU32 i)	{ return Ps::IntBool(data.mLink[i] & MSH_ACTIVE_EDGE_MASK);	}
};

} // namespace Gu

}

#endif
