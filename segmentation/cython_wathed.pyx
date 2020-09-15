cimport cython

from collections import OrderedDict
import numpy as np
cimport numpy as np


cpdef steepestascent(np.ndarray[np.float32_t, ndim=4] aff, float low, float high):
    cdef int xdim,ydim,zdim,x,y,z
    cdef np.float32_t negx,negy,negz,posx,posy,posz,m
    cdef np.ndarray[np.uint32_t,ndim=3] sag 
    
    #(_, zdim, ydim, xdim) = aff.shape  # extract image size
    zdim = aff.shape[1]
    ydim = aff.shape[2]
    xdim = aff.shape[3]
    sag = np.zeros([zdim,ydim,xdim], dtype = "uint32")  # initialize steepest ascent graph
    
    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                #weights of all six edges incident to (x,y,z)
                if(x > 0):
                    negx = aff[0,z,y,x]
                else:
                    negx = low
                if(y > 0):
                    negy = aff[1,z,y,x]
                else:
                    negy = low
                if(z > 0):
                    negz = aff[2,z,y,x]
                else:
                    negz = low
                if(x < xdim-1):
                    posx = aff[0,z,y,x+1]
                else:
                    posx = low
                if(y < ydim-1):
                    posy = aff[1,z,y+1,x]
                else:
                    posy = low
                if(z < zdim-1):
                    posz = aff[2,z+1,y,x]
                else:
                    posz = low
                # aff=low for edges directed outside boundaries of image
                
                m = max(negx,negy)
                m = max(m,negz)
                m = max(m,posx)
                m = max(m,posy)
                m = max(m,posz)
                #m = maximum((negx,negy,negz,posx,posy,posz))
                
                # keep edges with maximal affinity
                if(m > low):
                    if ( negx == m or negx >= high ):
                        sag[z,y,x] = sag[z,y,x] | 0x01
                    if ( negy == m or negy >= high ):
                        sag[z,y,x] = sag[z,y,x] | 0x02
                    if ( negz == m or negz >= high ):
                        sag[z,y,x] = sag[z,y,x] | 0x04
                    if ( posx == m or posx >= high ):
                        sag[z,y,x] = sag[z,y,x] | 0x08
                    if ( posy == m or posy >= high ):
                        sag[z,y,x] = sag[z,y,x] | 0x10
                    if ( posz == m or posz >= high ):
                        sag[z,y,x] = sag[z,y,x] | 0x20
                    
    return sag



cpdef divideplateaus(np.ndarray[np.uint32_t,ndim=3] sag_):
    
    cdef int zdim,ydim,xdim,idx,d,bfs_index
    cdef np.ndarray[np.uint32_t,ndim=1] sag
    cdef np.ndarray[np.uint32_t,ndim=3] sag__
    cdef np.ndarray[np.int64_t,ndim=1] dir
    cdef list dirmask,idirmask,bfs
    cdef np.uint32_t to_set
    
    #(zdim,ydim,xdim) = sag.shape
    zdim = sag_.shape[0]
    ydim = sag_.shape[1]
    xdim = sag_.shape[2]
    
    dir = np.array([-1, -xdim, -xdim*ydim, 1, xdim, xdim*ydim], dtype="int64")
    dirmask  = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20]
    idirmask = [0x08, 0x10, 0x20, 0x01, 0x02, 0x04]
    
    # queue all vertices for which a purely outgoing edge exists
    bfs = []
    #capacity is no
       #sizehint!(bfs,length(sag))
    
    sag = sag_.flatten()
    for idx in range(sag.size):
        for d in range(6):
            if((sag[idx] & dirmask[d]) != 0):
                if((sag[idx+dir[d]] & idirmask[d]) == 0):
                    sag[idx] = sag[idx] | 0x40
                    bfs.append(idx)
                    break
    
    #divide pateaus
    bfs_index = 0
    while(bfs_index <= len(bfs)-1):
        idx = bfs[bfs_index]
        to_set = 0
        for d in range(6):
            if((sag[idx] & dirmask[d]) != 0):
                if (sag[idx+dir[d]] & idirmask[d]) != 0:
                    if ( sag[idx+dir[d]] & 0x40 ) == 0:
                        bfs.append(idx+dir[d])
                        sag[idx+dir[d]] = sag[idx+dir[d]] | 0x40
                else:
                    to_set = dirmask[d]
        sag[idx] = to_set
        bfs_index += 1
    bfs.clear()
    sag__ =  sag.reshape((zdim,ydim,xdim))
    return sag__




cpdef findbasins(np.ndarray[np.uint32_t,ndim=3] sag):
    cdef int zdim,ydim,xdim,counts0,idx,next_id,bfs_index,me,d,him,it
    cdef list dirmask,counts,bfs
    cdef np.uint32_t a
    cdef np.ndarray[np.uint32_t,ndim=1] seg
    cdef np.ndarray[np.uint32_t,ndim=3] seg_
    cdef np.ndarray[np.int64_t,ndim=1] dir

    
    #(zdim,ydim,xdim) = sag.shape
    zdim = sag.shape[0]
    ydim = sag.shape[1]
    xdim = sag.shape[2]
    
    dir = np.array([-1, -xdim, -xdim*ydim, 1, xdim, xdim*ydim], dtype="int64")
    dirmask  = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20]
    
    #for sub function (high_bit() and low_bits() )
    a = sag[0,0,0]

    counts0 = 0  # number of background voxels
    counts = []  # voxel counts for each basin
    bfs = []
    
    next_id = 1   # initialize basin ID
    
    seg = sag.flatten()
    for idx in range(seg.size):
        if seg[idx] == 0:   # background voxel (no edges at all)
            seg[idx] = seg[idx] | high_bit(a)   # mark as assigned
            counts0 += 1
        elif (seg[idx] & high_bit(a))==0:
            bfs.append(idx)
            seg[idx] = seg[idx] | 0x40

            bfs_index = 0
            while(bfs_index < len(bfs)):
                me = bfs[bfs_index]
                for d in range(6):
                    if (seg[me] & dirmask[d]) != 0:
                        him = me + dir[d]
                        if (seg[him] & high_bit(a)) != 0:
                            for it in bfs:
                                seg[it]  = seg[him]
                            counts[ (seg[him] & low_bits(a))-1 ] += len(bfs)
                            bfs = []
                            break
                        else:
                            if ( seg[him] & 0x40 ) == 0:
                                seg[him] = seg[him] | 0x40
                                bfs.append(him)
                bfs_index += 1

            if len(bfs) != 0:
                counts.append(len(bfs))
                for it in bfs:
                    seg[it] = high_bit(a) | next_id
                next_id += 1
                bfs = []


    print("found: %d components" % (next_id-1))
    
    for idx in range(seg.size):
        seg[idx] = seg[idx] & low_bits(a)
    
    
    bfs = []
    seg_ = seg.reshape((zdim,ydim,xdim))
    return seg_, counts, counts0



cdef high_bit(np.uint32_t x):
        return 0x80000000
    
cdef low_bits(np.uint32_t x):
    return 0x7FFFFFFF






cpdef regiongraph(np.ndarray[np.float32_t, ndim=4] aff, np.ndarray[np.uint32_t,ndim=3] seg, int max_segid):
    
    cdef int zdim,ydim,xdim,z,y,x,nedges
    cdef np.uint32_t ZERO_SEG
    cdef tuple p,key
    cdef list rg,rg_
    
    #(zdim,ydim,xdim) = seg.shape
    zdim = seg.shape[0]
    ydim = seg.shape[1]
    xdim = seg.shape[2]
    
    ZERO_SEG = 0
    
    edges = OrderedDict()
    #sizehint
    
    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                if seg[z,y,x] != ZERO_SEG:
                    if (x > 0) and (seg[z,y,x-1]!=ZERO_SEG) and (seg[z,y,x]!=seg[z,y,x-1]):
                        p = (min(seg[z,y,x], seg[z,y,x-1]), max(seg[z,y,x], seg[z,y,x-1]))
                        if p in edges.keys():
                            edges[p] = max(edges[p], aff[0,z,y,x])
                        else:
                            edges[p] = aff[0,z,y,x]
                    if (y > 0) and (seg[z,y-1,x]!=ZERO_SEG) and (seg[z,y,x]!=seg[z,y-1,x]):
                        p = (min(seg[z,y,x], seg[z,y-1,x]), max(seg[z,y,x], seg[z,y-1,x]))
                        if p in edges.keys():
                            edges[p] = max(edges[p], aff[1,z,y,x])
                        else:
                            edges[p] = aff[1,z,y,x]
                    if (z > 0) and (seg[z-1,y,x]!=ZERO_SEG) and (seg[z,y,x]!=seg[z-1,y,x]):
                        p = (min(seg[z,y,x], seg[z-1,y,x]), max(seg[z,y,x], seg[z-1,y,x]))
                        if p in edges.keys():
                            edges[p] = max(edges[p], aff[2,z,y,x])
                        else:
                            edges[p] = aff[2,z,y,x]
    
    # separate weights and vertices in two arrays
    nedges = len(edges)
    print("Region graph size: %d" % nedges)
    
    # repackage in array of typles
    #rg = []
    #for key in edges:
    #    rg.append((edges[key],key[0],key[1]))
    rg = [(edges[key],key[0],key[1]) for key in edges]
    rg_ = sorted(rg, key=get_weight, reverse=True)
    return rg_

cdef get_weight(tuple edge):
    return edge[0]






cpdef baseseg(np.ndarray[np.float32_t, ndim=4] aff, float low, float high, list thresholds, float dust_size, bint is_threshold_relative):
    
    #return seg,rg,counts
    
    cdef np.ndarray[np.uint32_t, ndim=3] seg1, seg2, seg3#, seg4
    cdef int counts0
    cdef list counts, rg#, new_rg
    
    if is_threshold_relative:
        low, high, thresholds = relative2absolute(aff, low, high, thresholds)

    #print("absolute watershed threshold: low: %f, high: %f, thresholds: %f, dust: %d" % low,high,thresholds,dust)
    # this seg is a steepest ascent graph, it was named as such for in-place computation to reduce memory comsuption
    print("steepestascent...")
    seg1 = steepestascent(aff, low, high)
    print("divideplateaus...")
    seg2 = divideplateaus(seg1)
    print("findbasins!")
    (seg3, counts, counts0) = findbasins(seg2)
    print("regiongraph...")
    rg = regiongraph(aff, seg3, len(counts))
    #print("mergeregions...")
    #new_rg, seg4 = mergeregions(seg3, rg, counts, thresholds, dust_size)
    return seg3, rg, counts




cpdef segment(np.ndarray[np.float32_t, ndim=4] aff, float low, float high,  bint is_threshold_relative):
    
    #return only seg
    #without thresholds and dust_size
    
    cdef np.ndarray[np.uint32_t, ndim=3] seg1, seg2, seg3#, seg4
    cdef int counts0
    cdef list counts#, rg, new_rg
    
    if is_threshold_relative:
        low, high = relative2absolute_onlyseg(aff, low, high)

    #print("absolute watershed threshold: low: %f, high: %f, thresholds: %f, dust: %d" % low,high,thresholds,dust)
    # this seg is a steepest ascent graph, it was named as such for in-place computation to reduce memory comsuption
    print("steepestascent...")
    seg1 = steepestascent(aff, low, high)
    print("divideplateaus...")
    seg2 = divideplateaus(seg1)
    print("findbasins!")
    (seg3, counts, counts0) = findbasins(seg2)
    #print("regiongraph...")
    #rg = regiongraph(aff, seg3, len(counts))
    #print("mergeregions...")
    #new_rg, seg4 = mergeregions(seg3, rg, counts, thresholds, dust_size)
    return seg3


cpdef relative2absolute(np.ndarray[np.float32_t, ndim=4] aff, float low, float high, list thresholds):
    
    #use for baseseg()

    """
    use percentage threshold: low $(low), high $high, thresholds $thresholds
    """

    cdef int i
    cdef tuple h
    
    if len(aff) > 3*1024*1024*128:
        h = np.histogram(aff[:,0:min(128,aff.shape[1]),
                             0:min(1024,aff.shape[2]),0:min(1024,aff.shape[3])][:], 
                          bins = 1000000)
    else:
        h = np.histogram(aff[:], bins = 1000000)
    
    low  = percent2thd(h, low)
    high = percent2thd(h, high)
    for i in range(len( thresholds )):
        thresholds[i] = (thresholds[i][0], percent2thd(h, thresholds[i][1]))

    return low, high, thresholds


cpdef relative2absolute_onlyseg(np.ndarray[np.float32_t, ndim=4] aff, float low, float high):
    
    #use for get_seg()
    #without thresholds

    """
    use percentage threshold: low $(low), high $high
    """

    cdef tuple h
    
    if len(aff) > 3*1024*1024*128:
        h = np.histogram(aff[:,0:min(128,aff.shape[1]),
                        0:min(1024,aff.shape[2]),0:min(1024,aff.shape[3])][:], 
                          bins = 1000000)
    else:
        h = np.histogram(aff[:], bins = 1000000)
    
    low  = percent2thd(h, low)
    high = percent2thd(h, high)
    
    return low, high


cdef percent2thd(tuple hist, float percentage):
    cdef np.ndarray[np.int64_t,ndim=1] weights
    cdef np.ndarray[np.float32_t,ndim=1] edges
    cdef float totalVoxelNum,rank,accumulatedVoxelNum
    cdef int i
    
    
    weights = hist[0]
    edges = hist[1]
    # total number
    totalVoxelNum = sum(weights)
    # the rank of voxels corresponding to the threshold
    rank = totalVoxelNum * percentage
    # accumulate the voxel number
    accumulatedVoxelNum = 0
    for i in range(len(weights)):
        accumulatedVoxelNum += weights[i]
        if accumulatedVoxelNum >= rank:
            return edges[i]