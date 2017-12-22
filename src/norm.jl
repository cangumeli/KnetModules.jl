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
    `opt...`              See `batchnorm` kwargs

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
end

function BatchNorm(input::Int;
                   affine=true,
                   train=nothing, #default option
                   dtype=Float32,
                   moments=bnmoments(),
                   o...)
    w = Param(bnparams(dtype, input))
    return BatchNorm(w, moments, o, train)
end

forward(ctx, bn::BatchNorm, x) = 
    batchnorm(x, bn.moments, val(ctx, bn.w);
              bn.opt...,
              training=bn.train)
