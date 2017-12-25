using Knet, KnetModules
# for correct dispatch
import KnetModules.forward

#=
Declare a simple cnn model.
=#
type MyCNN <: KnetModule
    c1::Conv;    bc1::BatchNorm
    c2::Conv;    bc2::BatchNorm
    c3::Conv;    bc3::BatchNorm
    fc1::Linear; bfc1::BatchNorm
    fc2::Linear
end

MyCNN(;chs=[16,32,64], hsize=100, nclasses=10) = MyCNN(
    Conv(3,3,3,     chs[1];  padding=1, bias=false), BatchNorm(chs[1]),
    Conv(3,3,chs[1],chs[2];  padding=1, bias=false), BatchNorm(chs[2]),
    Conv(3,3,chs[2],chs[3];  padding=1, bias=false), BatchNorm(chs[3]),
    Linear(hsize, 8 * 8 * chs[3]; bias=false),       BatchNorm(hsize),
    Linear(10, hsize)
)

# Forward pass executes a regular julia code with macros, functions
# and program control if necessary
function forward(ctx, m::MyCNN, x)
    x = pool(relu.(@mc m.bc1(@mc m.c1(x))))
    x = pool(relu.(@mc m.bc2(@mc m.c2(x))))
    x = relu.(@mc m.bc3(@mc m.c3(x)))
    x = relu.(@mc m.bfc1(@mc m.fc1(x)))
    return @mc m.fc2(x)
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

const atype = gpu() >= 0 ? KnetArray : Array

# Perform training
function epoch!(model, gradfn, dtrn, optims;  mbatch=64)
    data = minibatch(dtrn[1], dtrn[2], mbatch;
                     shuffle=true,
                     xtype=atype)
    
    for (x, y) in data
        # Note the gradfn call just needs data
        grads = gradfn(x, y)
        update!(model, grads, optims)
    end
end

# Accuracy computation
function acc(model, xtst, ytst; mbatch=64)
    data = minibatch(xtst, ytst, mbatch;
                     partial=true,
                     xtype=atype)
    # Accuracy can directly take model and data
    return accuracy(model, data; average=true)
end


function report(epoch, model, dtrn, dtst)
    println("epoch: ", epoch)
    println("training accuracy ", acc(model, dtrn...))
    println("test accuracy ",     acc(model, dtst...))
end


function train(;epochs=5,
               checkpoint_filename="cnn_train",
               start_epoch=0,
               from_checkpoint=false)
    dtrn, dtst = loaddata()
    jld_file(epoch) = string(checkpoint_filename, "_epoch_",
                             epoch, ".jld")
    #model = MyCNN()
    if from_checkpoint && start_epoch > 0
        # ctx switch -> clean up the old model contents
        model = load_module(jld_file(start_epoch); ctx_switch=true)
    else
        model = MyCNN()
    end
    gpu!(model) # ignored if gpu id is -1
    gradfn = grad(model, nll)
    optims = optimizers(model, Momentum; lr=.1)
    for i = 1:epochs
        epoch!(model, gradfn, dtrn, optims)
        report(i, model, dtrn, dtst)
        info("Backing up model...")
        save_module(jld_file(start_epoch+i), model)
        println()
    end
    return model
end

endswith(string(PROGRAM_FILE), "cnn.jl") && train()
