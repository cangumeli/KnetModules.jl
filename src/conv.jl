"""
`conv_mode!(m::KnetModule, mode::Int)` sets all convolutions
to `mode=mode` in a module.
"""
function conv_mode!(m::KnetModule, mode::Int)
    for m in modules(m)
        if isa(m, Conv)
            m.opt = filter(x->x[1]!==:mode, m.opt)
            push!(m.opt, (:mode, mode))
        end
    end
end


"""
`kaiming(et, h, w, i, o)` Default initialization for conv
"""
kaiming(et, dims...) =
    et(sqrt(2 / prod(dims[1:end-1]))) .* randn(et, dims...)


"""
`Conv`: Performs the operation conv4(w, x; opt...) [.+ b]

# Constructors
    `Conv(r::Int, c::Int, i::Int, o::Int; kwargs)`


# Keywords (Constructor)
     `bias=true` Whether or not to add bias
     `winit=kaiming`  Weight initialization `(eltype, dims...)->array`
     `binit=zeros`    Bias initialization `(eltype, dims...)->array`
     `dtype=Float32`  Element type of weight and bias
     `opt...`         See `conv4` kwargs


# Fields
    `w`: Weight
    `b`: Bias
    `opt`: kwargs of `conv4`, provided in initialization

# Forward execution
    `c(ctx, x)
"""
type Conv <: KnetModule
    w::Param
    b::Union{Param, Void}
    opt
end

function Conv(dims...;
              winit=kaiming,
              binit=zeros,
              bias=true,
              dtype=Float32,
              opt...)
    w = Param(winit(dtype, dims...))
    b = nothing
    if bias
        b = Param(binit(dtype, 
                        [1 for i=1:length(dims)-2]..., 
                        dims[end], 1))
    end
    return Conv(w, b, opt)
end

function (c::Conv)(ctx, x; o...) #forward(ctx, c::Conv, x)
    o = conv4(val(ctx, c.w), x;
              c.opt..., o...)
    if c.b !== nothing
        o = o .+ val(ctx, c.b)
    end
    return o
end
