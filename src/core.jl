#=
This file contains ParamCtx, Param and KnetModule abstractions.
=#


"""Data structure used to store parameters. Currently, it is a 1d Any array"""
const ParamCtx = Array{Any, 1}

# The context abstraction

let
    global active_ctx, switch_ctx!, default_ctx, reset_ctx!
    
    default = []
    active = default
    
    """`active_ctx()::ParamCtx` returns the context in use"""
    active_ctx()::ParamCtx = active

    """`switch_ctx!(new)::ParamCtx` switches to a new context and returns it"""
    switch_ctx!(new::ParamCtx)::ParamCtx = (active=new)

    """`default_ctx()::ParamCtx` returns the default context provided by the system"""
    default_ctx()::ParamCtx = default

    """`reset_ctx!()::ParamCtx` removes everything from the context"""
    reset_ctx!()::ParamCtx = (active=ParamCxt())
end

"""`Param` stores an address to the parameters value.
Trainable arrays should be stored as `Param` inside modules.
User is responsible to keep track of `ParamCtx` object the `Param`
belongs to.

# Constructors
    `Param(w; ctx=active_ctx())` registers a parameter to the context ctx.
"""
type Param
    index::Integer
    Param(w; ctx::ParamCtx=active_ctx()) = Param(length(push!(ctx, w)))
end


"""
`val(ctx::ParamCtx, p::Union{Param, Void})` returns the value of parameter in
ctx. 
This function should be used used to access parameters for AutoGrad to work.
"""
val(ctx::ParamCtx, p::Union{Param, Void}) = (p==nothing) ? nothing : ctx[p.index]


"""
`aval(p::Param)` returns value of p in the current `active_ctx()`. 
It may stop differentiation when used.
"""
aval(p::Param) = val(active_ctx(), p)


"""
`setval!(p::Param, w; ctx=active_ctx())` sets the value of p to w in ctx.
"""
setval!(p::Param, w; ctx=active_ctx()) = (ctx[p.index] = w)


"""`KnetModule` is an abstract type inherited by other modules."""
abstract type KnetModule end

import AutoGrad.grad

"""
`grad(m::KnetModule, loss)` returns a grad function of the form `(args..., y)->g`.
Recorded parameters are lived in `active_ctx`.

 # Arguments

    `m`: A regular knet module

    `loss`: A function of the form `(ypred, ygold)->scalar_loss`.
"""
function grad(m::KnetModule, loss::Function)
    _predict(w, x...) = forward(w, m, x...)
    _loss(w, args...) = loss(_predict(w, args[1:end-1]...), args[end])
    lossgrad = grad(_loss)
    return (args...)->lossgrad(active_ctx(), args[1:end-1]..., args[end])
end

"""
`getgrad(p::Param, grads)` returns the corresponding gradient of `p` in
the grad array returned by any gradfn (generated by `grad` functions of Knet
or KnetModules).
"""
getgrad(p::Param, grads) = grads[p.index]


import Knet.optimizers

"""`optimizers(m::KnetModule, ofn::Function)` creates a group of optimizers. 
`ofn` should be of the form ofn(p), where o is a parameter."""
optimizers(m::KnetModule, ofn::Function) = map(p->ofn(p), params(m))


import Knet.update!
"""
`update!(m::KnetModule, grads, optims)`: Apply optimizers to grads
one by one. `optims` should have returned by
`KnetModules.optimizers`.
"""
function update!(m::KnetModule, grads, optims)
    for (p, o) in zip(params(m), optims)
        update!(aval(p), getgrad(p, grads), o)
    end
end


function _populate_recursive(m, list, match_type)
    if isa(m, match_type)
        push!(list, m)
    end
    
    if isa(m, KnetModule)
        for fn in sort(fieldnames(m))
            _populate_recursive(getfield(m, fn), list, match_type)
        end
    elseif isa(m, Array) || isa(m, Tuple)
        for l in m
            _populate_recursive(l, list, match_type)
        end
    elseif isa(m, Associative)
        for k in sort(keys(m))
            _populate_recursive(m[k], list, match_type)
        end
    else
        return
    end
end


"""
`params(m::KnetModule; kwargs)` returns parameters of the KnetModule m.
These include any value of type `Param` stored in fields, sub-modules, 
dictionaries, arrays and tuples. 

# Keywords

   `sorted=true`: If true, parameters are returned sorted based on
their locations in the context
"""
function params(m::KnetModule; sorted=false)
    res = []
    _populate_recursive(m, res, Param)
    if sorted
        sort!(res; lt=(r1, r2)->r1.index < r2.index)
    end
    return res
end


"""
`submodules(m::KnetModule)` returns list of modules contained in s.
These include any value of type `Param` stored in fields, sub-modules, 
dictionaries, arrays and tuples. 
"""
function submodules(m::KnetModule)
    res = []
    _populate_recursive(m, res, KnetModule)
    return res
end


"""
`gpu!(m::KnetModule)` transfers all parameters in `m` to supported
gpu devices, which is identical to converting their valus to `KnetArray`.
If no gpu is avaliable, a warning is given and transfer is ignored without
any error thrown.
"""
function gpu!(m::KnetModule)
    if gpu() < 0
        warn("Gpu transfer is ignored, no available gpu")
        return
    end
    for p in params(m)
        setval!(p, KnetArray(aval(p)))
    end
end


"""`cpu!(m::KnetModule)` transfers all parameters in `m` cpu, 
which is identical to converting their values to `Array`"""
function cpu!(m::KnetModule)
    for p in params(m)
        setval!(p; Array(aval(p)))
    end
end


"""
`training!(m::KnetModule)` makes the fields called `train` true if exists, 
which may effect the execution mode in some modules.
"""
function training!(m::KnetModule)
    for sm in submodules(m)
        if :train in fieldnames(m)
            m.train = true
        end
    end
end


"""
`testing!(m::KnetModule)` makes the fields called `train` `false` if exists, 
which may effect the execution mode in some modules.
"""
function testing!(m::KnetModule)
    for sm in submodules(m)
        if :train in fieldnames(m)
            m.train = false
        end
    end
end


# Macros for simple module operations
# TODO: support kwargs
"""
`@mc expr` Converts the expression `m(a...)` to `forward(ctx, m, a...)`. 
`ctx` should come from the upper scope.

# Example
    l = Linear(...)
    @mc l(x)
"""
macro mc(expr)
    if typeof(expr) == Expr && expr.head == :call
        return esc(:(forward(ctx, $(expr.args[1]),
                             $(expr.args[2:end]...))))
    end
    return expr
end

"""
`@run expr` Converts the expression m(a...) to forward(active_ctx(), m, a...).

# Example
    l = Linear(...)
    @run l(x)
"""
macro run(expr)
    if typeof(expr) == Expr && expr.head == :call
        return esc(:(forward(active_ctx(),
                             $(expr.args[1]),
                             $(expr.args[2:end]...))))
    end
    return expr
end


forward(ctx, m::KnetModule, args...) =
    error("Forward is not implemented for abstract types ",
          "and/or type ", typeof(m))
