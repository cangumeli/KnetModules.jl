using Knet, KnetModules

#=
First specify the model
(Param) is a type used to register
our trainable weights to global context.
=#
type Lreg <: KnetModule
    w::Param
    b::Param
end

# Now, let's write a constructor
function Lreg(input::Int)
    w = Param(0.1randn(1, input))
    b = Param(zeros(1,1))
    return Lreg(w, b)
end

# Last step is the declare the forward execution logic,
# by adding a method to KnetModules.forward function
import KnetModules.forward

# Here, ctx is a dependency where parameters are stored
forward(ctx, l::Lreg, x) =
    val(ctx, l.w) * x .+ val(ctx, l.b)


# Now, let's load the data and train our model
include(Knet.dir("data","housing.jl"))
x,y = housing()
model = Lreg(size(x,1))

# We built the entire forward pass logic, now the backward 
loss(ypred, ygold) = mean(abs2, ygold .- ypred)
gradfn = grad(model, loss)

# Now, gradfn is a function (x, y) -> grads
lr = 0.1
for i = 1:10
    grads = gradfn(x, y)
    #=
    - params(model) returns the parameters in model
    - aval(p) returns the default active context
    - getgrad(p, g) returns grad of p in returned grads g
    
    params will return any parameter that lives in
    - Fields of type Param
    - Param fields of associative fields
    - Param fields of Array or Tuple
    - Param fields of other fields that inherit KnetModule
    This process is recursively continued
    =#
    for p in params(model)
        setval!(p, aval(p) - lr * getgrad(p, grads))
    end
    # @run evaluates the model
    ypred = @run model(x)
    println("Loss at iter $i ->", loss(ypred, y))
end

# The above training loop is very low-level, but it
# introduces you important accessor functions available

# let's now create the model
info("Resetting model")
model = switch_clean_ctx!(Lreg(size(x,1)))
# you don't have to call switch_clean_ctx!, but
# if you don't do it, old model's weights will continue to be stored
gradfn = grad(model, loss)
# optimizers are similar to Knet.optimizers, works with KnetModule objects
optims = optimizers(model, Sgd; lr=0.1)
for i = 1:10
    grads = gradfn(x, y)
    update!(model, grads, optims)
    ypred = @run model(x)
    println("Loss at iter $i -> ", loss(ypred, y))
end

info("Saving the final model")
save_module("lreg.jld", model)









