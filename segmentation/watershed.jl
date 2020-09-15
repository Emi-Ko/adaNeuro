using Watershed
using EMIRT
using HDF5
#using Base.Test
using StatsBase

function _percent2thd(h::StatsBase.Histogram, percentage::AbstractFloat)
        # total number
        totalVoxelNum = sum(h.weights)
        # the rank of voxels corresponding to the threshold
        rank = totalVoxelNum * percentage
        # accumulate the voxel number
        accumulatedVoxelNum = 0
        for i in 1:length(h.weights)
            accumulatedVoxelNum += h.weights[i]
            if accumulatedVoxelNum >= rank
                return h.edges[1][i]
            end
        end
end



key  = "main";
aff_path = "/tmp/in.h5";
out_path = "/tmp/out.h5";

low  = 0.1;
high = 0.8;
dust_size = 600;
thresholds = [(800,0.2)];
is_threshold_relative = true;

aff = h5read(aff_path, key);

if is_threshold_relative
    if length(aff) > 3*1024*1024*128
        h = StatsBase.fit(Histogram,
                          aff[1:min(1024,size(aff,1)),
                              1:min(1024,size(aff,2)),
                              1:min(128, size(aff,3)),:][:]; 
                          nbins = 1000000)
    else
        h = StatsBase.fit(Histogram, aff[:]; nbins = 1000000)
    end
    low  = _percent2thd(h, low)
    high = _percent2thd(h, high)
    for i = 1:length( thresholds )
        thresholds[i] = tuple(thresholds[i][1], _percent2thd(h, thresholds[i][2]))
    end
end
@info("absolute watershed threshold: low: $low, high: $high, thresholds: $(thresholds), dust: $(dust_size)")
# this seg is a steepest ascent graph, it was named as such for in-place computation to reduce memory comsuption
println("steepestascent...")
seg = steepestascent(aff, low, high)
println("divideplateaus...")
divideplateaus!(seg)
println("findbasins!")
(seg, counts, counts0) = findbasins!(seg)
println("regiongraph...")
rg = regiongraph(aff, seg, length(counts))

h5write(out_path, key, seg)