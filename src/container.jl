"""Abstract type for container modules -Modules that specify
execution method"""
abstract type KnetContainer <: KnetModule end


type Sequential <: KnetContainer
    layers::Array{Union{KnetModule, Function}, 1}
end

Sequential(ls...) = Sequential([l for l in ls])

add!(s::Sequential, m...) = push!(s.layers, m...)

function forward(ctx, s::Sequential, x)
    o = x
    for l in s.layers
        o = isa(l, Function) ? l(o) : @mc l(o)
    end
    return o
end
