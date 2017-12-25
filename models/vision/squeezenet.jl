using Knet, KnetModules
import KnetModules.forward

type Fire <: KnetModule
    squeeze::Conv
    expand1x1::Conv
    expand3x3::Conv
end

function Fire(inp::Int, sq::Int, e1x1::Int, e3x3::Int)
    squeeze   = Conv(1, 1, inp, sq)
    expand1x1 = Conv(1, 1, sq, e1x1)
    expand3x3 = Conv(3, 3, sq, e3x3; padding=1)
    return Fire(squeeze, expand1x1, expand3x3)
end

function forward(ctx, f::Fire, x)
    o = relu.(@mc f.squeeze(x))
    return ccat(relu.(@mc f.expand1x1(o)),
                relu.(@mc f.expand3x3(o)))
end

# FIXME: too many memory allocations
# FIXME: model is too slow in gpu, possibly because of this
function ccat(a1, a2)
    @assert size(a1, 1,2,4) == size(a2, 1,2,4)
    a1_ = mat(permutedims(a1, (1,2,4,3)))
    a2_ = mat(permutedims(a2, (1,2,4,3)))
    out = hcat(a1_, a2_)
    out = reshape(out, (size(a1, 1,2,4)...,
                        size(out, 2)))
    return permutedims(out, (1,2,4,3))
end

"""
# Usage
    s = SqueezeNet(;num_classes=1000, pdrop=0.5, trained=true)
    @run s(x)
"""
type SqueezeNet <: KnetModule
    features::Sequential
    last_conv::Conv
    pdrop
end

function SqueezeNet(;num_classes=1000, pdrop=0.5, trained=true)
    features = Sequential(
        Conv(3, 3, 3, 64; stride=2),
        ReLU(),
        MaxPool(;window=(3,3), stride=(2,2)),
        Fire(64, 16, 64, 64),
        Fire(128, 16, 64, 64),
        #Fire(128, 32, 128, 128),
        MaxPool(;window=(3,3), stride=(2,2)),
        Fire(128, 32, 128, 128),
        Fire(256, 32, 128, 128),
        MaxPool(;window=(3,3), stride=(2,2)),
        Fire(256, 48, 192, 192),
        Fire(384, 48, 192, 192),
        Fire(384, 64, 256, 256),
        Fire(512, 64, 256, 256)
    )
    last_conv = Conv(1, 1, 512, num_classes;
                     winit=(et, a...)->et(0.01) .* rand(et, a...))
    return SqueezeNet(features, last_conv, pdrop)
end

function forward(ctx, sn::SqueezeNet, x)
    o = @mc sn.features(x)
    o = dropout(o, sn.pdrop)
    o = @mc sn.last_conv(o)
    o = pool(o; window=size(o,1,2), mode=2)
    return mat(o)
end


