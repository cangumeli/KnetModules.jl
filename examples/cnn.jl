using Knet, KnetModules
# for correct dispatch
import KnetModules.forward

#=
Declare a simple cnn model.
=#
type MyCNN <: KnetModule
    c1::Conv;  b1::BatchNorm
    c2::Conv;  b2::BatchNorm
    c3::Conv;  b3::BatchNorm
    h::Linear; bh::BatchNorm
    out::Linear
end

MyCNN(;chs=[16,32,64], hsize=100, nclasses=10) = MyCNN(
    Conv(3,3,3,     chs[1];  padding=1, bias=false), BatchNorm(chs[1]),
    Conv(3,3,chs[1],chs[2];  padding=1, bias=false), BatchNorm(chs[2]),
    Conv(3,3,chs[2],chs[3];  padding=1, bias=false), BatchNorm(chs[3]),
    Linear(hsize, 8 * 8 * chs[3]; bias=false),       BatchNorm(hsize),
    Linear(10, hsize)
)

function forward(ctx, m::MyCNN, x)
    x = pool(relu.(@mc m.b1(@mc m.c1(x))))
    x = pool(relu.(@mc m.b2(@mc m.c2(x))))
    x = relu.(@mc m.b3(@mc m.c2(x)))
    x = relu.(@mc m.bh(@mc m.h(x)))
    return @mc m.out(x)
end


# Load the data
include(Pkg.dir("Knet", "data", "cifar.jl"))

function loaddata()
    xtrn, ytrn, xtst, ytst = cifar10()
    mn = mean(xtrn, 4)
    xtrn = xtrn .- mn
    xtst = xtst .- mn
    return (xtrn, ytrn), (xtst, ytst)
end


# Perform training
function epoch!(model, gradfn, dtrn, optims;  mbatch=64)
    data = minibatch(dtrn[1], dtrn[2], mbatch;
                     shuffle=true,
                     xtype=atype())
    
    for (x, y) in data
        # Note the gradfn call
        g = gradfn(x, y)
        update!(model, g, optims)
    end
end

# Accuracy computation
function acc(model, xtst, ytst; mbatch=64)
    data = minibatch(xtst, ytst, mbatch;
                     partial=true,
                     xtype=atype())
    return accuracy(model, data; average=true)
end


function report(epoch, model, dtrn, dtst)
    println("epoch: ", epoch)
    println("training accuracy ", acc(model, dtrn...))
    println("test accuracy ",     acc(model, dtst...))
    println()
end


function train(;epochs=10)
    dtrn, dtst = loaddata()
    model = MyCNN()
    gradfn = grad(model, nll)
    optims = optimizers(model, Momentum; lr=.1)
    for i = 1:epochs
        epoch!(model, gradfn, dtrn, optims)
        report(i, model, dtrn, dtst)
    end
    return model
end

endswith(string(PROGRAM_FILE), "cnn.jl") && train()
