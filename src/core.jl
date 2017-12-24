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
    reset_ctx!()::ParamCtx = (active=ParamCtx())
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
    Param(w; ctx::ParamCtx=active_ctx()) = new(length(push!(ctx, w)))
end


"""
`val(ctx, p::Union{Param, Void})` returns the value of parameter in
ctx. 
This function should be used used to access parameters for AutoGrad to work.
"""
val(ctx, p::Union{Param, Void}) = ctx[p.index]


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

"""`optimizers(m::KnetModule, otype; sorted=true, o...)` creates a group of optimizers. 
otype specifies a Knet optimizer and `o...` is its options. `sorted` is a boolean
passed to the params.
"""
optimizers(m::KnetModule, otype; sorted=true, o...) =
    map(_->otype(;o...), params(m; sorted=sorted))


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


function ctx_dict(m::KnetModule)
    cd = Dict()
    for p in params(m)
        cd[p.index] = aval(p)
    end
    return cd
end

# This thing is used with a loaded (not initialized) Knet
# module.
function from_ctx_dict!(m::KnetModule, cd::Dict; ctx=nothing)::ParamCtx
    if ctx == nothing; ctx = ParamCtx(); end
    for p in params(m)
        # FIXME: try not to break the abstraction
        p.index = length(push!(ctx, cd[p.index]))
    end
    return ctx
end


if Pkg.installed("JLD") !== nothing
    import JLD
else
    JLD = nothing
end

@inline assert_jld() =
    @assert (JLD !== nothing) "Install JLD for using this function"

function save_module(filename::String, m::KnetModule)
    assert_jld()
    JLD.save(filename, "model", m, "ctx_dict", ctx_dict(m))
end

function load_module(filename::String; ctx_switch=false)
    assert_jld()
    loaded = JLD.load(filename)
    model = loaded["model"]
    cdict = loaded["ctx_dict"]
    if ctx_switch
        switch_ctx!(from_ctx_dict!(model, cdict))
    else
        from_ctx_dict!(model, cdict; ctx=active_ctx())
    end
    return model
end

# assumes indices are consistent, to be used during training
function restore_module!(filename::String, target::KnetModule)
    assert_jld()
    cdict = JLD.load(filename)["ctx_dict"]
    for p in params(target)
        copy!(aval(p), cdict[p.index])
    end
end

#=let JLD=nothing
    global jld
    jld() = JLD==nothing && (JLD = Pkg.installed("JLD") !== nothing) || JLD
end=#


#="""
`new_cxt!(m::KnetModule; old_dict=true)` creates a new context, 
copy the model to that context and returns the created context. 

You should use `switch_ctx!(new_ctx!(model))` to continue
using the model properly, otherwise old ctx will be in use.

If `old_dict=true`, a dictionary maps old indices to new
ones is returned.
"""
function new_ctx!(m::KnetModule; old_dict=true)
    ctx = ParamCtx()
    dict = old_dict ? Dict() : nothing
    for p in params(m)
        ind = length(push!(ctx, aval(p)))
        if old_dict
            dict[p.index] = ind
        end
        p.index = ind
    end
    return old_dict ? ctx : (ctx, old_dict)
end


"""
`is_ctx_clean(m::KnetModules)::Bool` returns
whether or not the whole `active_ctx()` is
covered by the model m.
"""
function is_ctx_clean(m::KnetModule)
    ps = Set{Int}(map(x->x.index, params(m)))
    as = Set{Int}(1:length(active_ctx()))
    return length(setdiff(ps,as)) == 1
end=#
