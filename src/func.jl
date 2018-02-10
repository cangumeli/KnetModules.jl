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
           (m::$(Act))(ctx, x) = $(fn).(x)
           #forward(ctx, m::$(Act), x) = $(fn).(x)
        end
    ))
end


abstract type Pool <: FnModule end

# Pooling
type MaxPool <: Pool; opt; end

MaxPool(;o...) = MaxPool(o)

(m::MaxPool)(ctx, x) = pool(x; m.opt..., mode=0)


type AvgPool <: Pool
    opt
    include_pad::Bool
end

AvgPool(;include_pad=true, o...) = AvgPool(o, include_pad)

(m::AvgPool)(ctx, x) =
    pool(x; m.opt..., mode=1+Int(m.include_pad))


# Dropout
type Dropout <: FnModule
    p::AbstractFloat
    train::Union{Bool, Void}
end

Dropout(pdrop; train=nothing) = Dropout(p, train)

(d::Dropout)(ctx, x) =
    isa(d.train, Bool) ? dropout(x, d.p; training=d.train) : dropout(x, d.p)

