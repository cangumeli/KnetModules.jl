#=
This file contains functional modules (modules without any learnable contents)
=#

abstract type FnModule <: KnetModule end

abstract type Activation <: FnModule end

# Activation
for (Act, fn) in zip([:ReLU, :Sigmoid, :Tanh], [:relu, :sigm, :tanh])
    eval(:(
        begin
           type $(Act) <: Activation; ;end
           forward(ctx, m::$(Act), x) = $(fn).(x)
        end
    ))
end


abstract type Pool <: FnModule end

# Pooling
type MaxPool <: Pool; opt; end

MaxPool(;o...) = MaxPool(o)

forward(ctx, m::MaxPool, x) = pool(x; m.opt..., mode=0)


type AvgPool <: Pool
    opt
    include_pad::Bool
end

AvgPool(;include_pad=true, o...) = AvgPool(o, include_pad)

forward(ctx, m::AvgPool, x) = pool(x; m.opt..., mode=1+Int(m.include_pad))


# Dropout
type Dropout <: FnModule
    p::AbstractFloat
end

forward(ctx, d::Dropout, x) = dropout(x, d.p)

