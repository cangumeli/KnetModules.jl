using Knet, KnetModules

# Model defition
type MLP <: KnetModule
    hiddens::Array{Linear, 1}
    output::Linear
    actfn::Activation
end

function MLP(ninput::Int, nhidden::Int, noutput::Int;
             nhiddens=1, actfn=ReLU())
    hiddens = []
    for i = 1:nhiddens
        push!(hiddens, Linear(nhidden, ninput))
        ninput = nhidden
    end
    output = Linear(noutput, nhidden)
    return MLP(hiddens, output, actfn)
end

function forward(ctx, m::MLP, x)
    for hidden in m.hiddens
        x = @mc m.actfn(@mc hidden(x))
    end
    return @mc m.output(x)
end


# Training epuch
function train(model, gradfn, data, optim)
    for (x, y) in data
        grads = gradfn(x, y)
        update!(model, grads, optim)
    end
end


include(Knet.dir("data","mnist.jl"))
xtrn, ytrn, xtst, ytst = mnist()
dtrn = minibatch(xtrn, ytrn, 100)
dtst = minibatch(xtst, ytst, 100)

model = MLP(784, 64, 10)
gradfn = grad(model, nll)
optim = optimizers(model, Adam)

println("epoch 0",
        "\ntrn accuracy: ", accuracy(model, dtrn),
        "\ntst accuracy: ", accuracy(model, dtst))
for epoch=1:10
    train(model, gradfn, dtrn, optim)
    println("\nepoch $epoch",
            "\ntrn accuracy: ", accuracy(model, dtrn),
            "\ntst accuracy: ", accuracy(model, dtst))
end

# Open this to play with serialization
const ser_demo = false
if ser_demo
    # Serialization
    const filename = "mnist_mlp.jld"
    info("Saving model")
    # save_module filename, model and optional addition state (here, it is optim)
    save_module(filename, model, Dict(["optim"=>optim]))

    reset_ctx!() # clear old model for memory efficiency
    model, state = load_module(filename)
    println("Accuracy of restored ", accuracy(model, dtst))

    optim = state["optim"]
    gradfn = grad(model, nll)
    train(model, gradfn, dtrn, optim)
    println("Test accuracy after one step more training ", accuracy(model, dtst))
end
