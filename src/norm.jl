using Knet: BNMoments

"""Abstract type for normalization layers"""
abstract type AbstractNorm <: KnetModule end


"""
`BatchNorm`: Performs the operation batchnorm

# Constructors
    `BatchNorm(input::Int; kwargs)`

# Fields
    `w`: Affine params or nothing
    `moments`: batchnorm moments
    `opt`: kwargs of `batchnorm`, provided in initialization
    `train`: Bool or nothing, dictates mode

# Keywords
    `affine=true`         Whether or not to add bias
    `train=nothing`       Training mode, see `batchnorm`
    `dtype=Float32`       Element type of affine params
    `moments=bnmoments()` Struture stored running mean and var, nullable
    `remove_initfns=true` If true, function fields in moments are removed after being used once
    `opt...`              See `Knet.batchnorm` kwargs

# Forward execution
    `forward(ctx, bn::BatchNorm, x)`
    `@mc bn(x)`
    `@run bn(x)
"""
type BatchNorm <: AbstractNorm
    w::Union{Void, Param}
    moments::Union{Void, BNMoments}
    opt
    train::Union{Void, Bool}
    remove_initfns::Bool
end

function BatchNorm(input::Int;
                   affine=true,
                   train=nothing, #default option
                   dtype=Float32,
                   moments=bnmoments(),
                   remove_initfns=true, #serialization hack (should go to Knet?)
                   o...)
    w = Param(bnparams(dtype, input))
    return BatchNorm(w, moments, o, train, remove_initfns)
end

function forward(ctx, bn::BatchNorm, x)
    o = batchnorm(x, bn.moments, val(ctx, bn.w);
                  bn.opt..., training=bn.train)
    if bn.remove_initfns
        bn.moments.meaninit = nothing
        bn.moments.varinit  = nothing
    end
    return o
end
