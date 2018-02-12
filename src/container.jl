"""Abstract type for container modules -Modules that specify an
execution structure for a given group of modules"""
abstract type KnetContainer <: KnetModule end

"""
Sequential <: KnetModule

# Constructor
    Sequential(ls...) adds layers in ls.
# Fields
    layers::Array{Union{KnetModule, Function}, 1}

# Usage
    s = Sequential(Linear(...), ReLU(), Linear(...),...)
    @mc s(x)
    @run r(x)

 Layers stored in layers field is executed in order, where layer
 n+1 takes layer n's output as its input. Layer inputs and outputs
 should be consistent.

See `add!` function to add additional layers after construction

# Note on storing functions in layers
    Currently, JLD throws error if you add functions in layers,
    but it is still kept for convenience, considering the future
    support. Currently, using functional modules is a better practice.
    See FnModule types.
"""
type Sequential <: KnetContainer
    layers::Array{Union{KnetModule, Function}, 1}
    #layers::Array{KnetModule, 1}
end

function Sequential(ls...)
    new = Sequential([l for l in ls])
    if any(x->isa(x, Function), new.layers)
        warn("Serialization functions like save_module and load_module",
             " may not work when you use functions in Sequential")
    end
    return new
end

"""
add!(s::Sequential, m...) adds modules m... to the end of layers.
Identical to push!(s.layers, m...)
"""
add!(s::Sequential, m...) = push!(s.layers, m...)

function (s::Sequential)(ctx, x)
    for l in s.layers
        if isa(l, Function) || isa(l, FnModule)
            x = l(x)
        else
            x = l(ctx, x)
        end
    end
    return x
end
