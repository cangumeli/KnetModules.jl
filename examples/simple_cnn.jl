using Knet, KnetModules

include(Knet.dir("data","mnist.jl"))
xtrn, ytrn, xtst, ytst = mnist()
dtrn = minibatch(xtrn, ytrn, 100)
dtst = minibatch(xtst, ytst, 100)

function train(model, gradfn, data, optim)
    for (x, y) in data
        grads = gradfn(x, y)
        update!(model, grads, optim)
    end
end

model = Sequential(
    Conv(5, 5, 1,  20), ReLU(), MaxPool(),
    Conv(5, 5, 20, 50), ReLU(), MaxPool(),
    Linear(500, 800),   ReLU(),
    Linear(10,  500)
)
gpu!(model) #ignored if no GPU available

gradfn = grad(model, nll)
optim = optimizers(model, Adam)
report(epoch) =
    println((:epoch, epoch,
             :trn, accuracy(model, dtrn),
             :tst, accuracy(model, dtst)))

report(0)
for epoch = 1:10
    train(model, gradfn, dtrn, optim)
    report(epoch)
end

ser = false
if ser
    info("Saving module")
    save_module("lenet_seq.jld", model,
                Dict(["optim"=>optim]))
end




