let #convolution config
    mode = 0
    global mode_cross!, mode_conv!, conv_mode
    
    """Returns mode of convolution (1 cross, 0 conv)"""
    conv_mode() = mode

    """Swithces cross correlation mode in all convs"""
    mode_cross!() = (mode=1)

    """Swithces convolution mode in all convs"""
    mode_conv!() = (mode=0)
end


"""
`kaiming(et, h, w, i, o)` Default initialization for conv
"""
kaiming(et, h, w, i, o) =
    et(sqrt(2 / (w * h * o))) .* randn(et, h, w, i, o)


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
    `forward(ctx, c::Conv, x)`
    `@mc c(x)`
    `@run c(x)`
"""
type Conv <: KnetModule
    w::Param
    b::Union{Param, Void}
    opt
end

function Conv(r::Int, c::Int, i::Int, o::Int;
              winit=kaiming,
              binit=zeros,
              bias=true,
              dtype=Float32,
              opt...)
    w = Param(winit(dtype, r, c, i, o))
    b =  bias ? Param(binit(dtype, 1, 1, o, 1)) : nothing
    return Conv(w, b, opt)
end

function forward(ctx, c::Conv, x)
    o = conv4(val(ctx, c.w), x;
              mode=conv_mode(),
              c.opt...)
    if c.b !== nothing
        o = o .+ val(ctx, c.b)
    end
    return o
end
