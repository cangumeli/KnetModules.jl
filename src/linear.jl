"""
`Linear`: Performs the operation w * x [.+ b]
# Constructors
    `Linear(output::Int, input::Int; kwargs)`

# Fields
    `w`: Weight
    `b`: Bias

# Keywords
      `bias=true` Whether or not to add bias
      `winit=xavier`  Weight initialization `(eltype, dims...)->array`
      `binit=zeros`   Bias initialization `(eltype, dims...)->array`
      `dtype=Float32` Element type of weight and bias

# Forward execution
    `forward(ctx, l::Linear, x)`
    `@mc l(x)`
    `@run l(x)

  If `ndims(x)` > 2, `mat(x)` is called to reshape `x`.
"""
type Linear <: KnetModule
    w::Param
    b::Union{Param, Void}
end


function Linear(output::Int, input::Int;
                bias=true,
                winit=xavier,
                binit=zeros,
                dtype=Float32)
    w = Param(winit(dtype, output, input))
    b = bias ? Param(binit(dtype, output, 1)) : nothing
    return Linear(w, b)
end

#function forward(ctx, l::Linear, x)
function (l::Linear)(ctx, x)
    if ndims(x) > 2;  x = mat(x); end
    o = val(ctx, l.w) * x
    if l.b !== nothing
        o = o .+ val(ctx, l.b)
    end
    return o
end
