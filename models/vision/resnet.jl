using Knet, KnetModules
import KnetModules.forward

# -- ImageNet --
const basic_channels = [64, 128, 256, 512]
const bneck_channels = 4basic_channels

resnet18(;o...) =
    ResNet(BasicBlock, basic_channels, [2, 2, 2, 2]; o...)

resnet34(;o...) =
    ResNet(BasicBlock, basic_channels, [3, 4, 6, 3]; o...)
    
resnet50(;o...) =
    ResNet(Bottleneck, bneck_channels, [3, 4, 6, 3]; o...)
 
resnet101(;o...) =
    ResNet(Bottleneck, bneck_channels, [3, 4, 23, 3]; o...)

resnet152(;o...) =
    ResNet(Bottleneck, bneck_channels, [3, 8, 36, 3]; o...)


# -- CIFAR --
for (m, d) in zip([:resnet20, :resnet32, :resnet56, :resnet110],
                  [20, 32, 56, 110])
    eval(:(
        $(m)(;o...) = ResNetCifar($(d); o...)
    ))
end


type BasicBlock <: KnetModule
    conv1; bn1
    conv2; bn2
    downsample
end

function BasicBlock(input::Int, output::Int;
                    stride=1, downsample=nothing)
    conv1 = Conv(3,3,input, output;
                  padding=1, stride=stride,
                  bias=false)
    bn1 = BatchNorm(output)
    conv2 = Conv(3, 3, output, output;
                  padding=1, stride=1,
                  bias=false)
    bn2 = BatchNorm(output)
    return BasicBlock(conv1, bn1, conv2, bn2, downsample)
end

function forward(ctx, b::BasicBlock, x)
    o = relu.(@mc b.bn1(@mc b.conv1(x)))
    o = @mc b.bn2(@mc b.conv2(o))
    if b.downsample !== nothing
        x = @mc b.downsample(x)
    end
    return relu.(o .+ x)
end


type Bottleneck <: KnetModule
    conv1; bn1
    conv2; bn2
    conv3; bn3
    downsample
end

let s = true
    global stride3!, stride1!, stride3
    stride3!() = (s=true)
    stride1!() = (s=false)
    stride3() = s
end

function Bottleneck(input::Int, output::Int;
                    stride=1,downsample=nothing)
    # middle dim.
    planes = div(output, 4)
    # layers
    conv1 = Conv(1, 1, input, planes;
                 bias=false,
                 stride=(stride3() ? 1 : stride))
    bn1 = BatchNorm(planes)
    conv2 = Conv(3, 3, planes, planes;
                 bias=false,
                 stride=(stride3() ? stride: 1),
                 padding=1)
    bn2 = BatchNorm(planes)
    conv3 = Conv(1, 1, planes, output;
                  bias=false)
    bn3 = BatchNorm(output)
    # return
    return Bottleneck(conv1, bn1, conv2, bn2,
                      conv3, bn3, downsample)
end


function forward(ctx, b::Bottleneck, x)
    o = relu.(@mc b.bn1(@mc b.conv1(x)))
    o = relu.(@mc b.bn2(@mc b.conv2(o)))
    o = @mc b.bn3(@mc b.conv3(o))
    if b.downsample !== nothing
        x = @mc b.downsample(x)
    end
    return relu.(o .+ x)
end


function _make_layer(block::Type, input::Int, output::Int,
                     nblocks::Int; stride=1)
    downsample = nothing
    if stride !== 1 || input !== output
        downsample = Sequential(
            Conv(1, 1, input, output;
                stride=stride, bias=false),
            BatchNorm(output))
    end
    modules = Sequential()
    add!(modules, block(input, output;
                        stride=stride,
                        downsample=downsample))
    for i = 2:nblocks
        add!(modules, block(output, output))
    end
    return modules
end


type ResNet <: KnetModule
    conv1::Conv
    bn1::BatchNorm
    layers::Array{Sequential, 1}
    output::Union{Void, Linear}
    dataset::Symbol
    stage::Int
    avg_pool::Bool
end

function ResNet(block::Type,
                nchannels::Array{Int, 1},
                nrepeats::Array{Int, 1};
                nclasses=1000,
                stage=0)
    conv1 = Conv(7, 7, 3, 64; stride=2, padding=3, bias=false)
    bn1 = BatchNorm(64)
    layers = []
    nchannels = [64, nchannels...]
    for i = 1:4
        push!(layers, _make_layer(
            block, nchannels[i],
            nchannels[i+1], nrepeats[i];
            stride = 1+Int(i>1)))
        (i == stage || (stage==5 && i==4)) &&
            return ResNet(conv1, bn1, layers, nothing, stage==5)
    end
    
    output = Linear(nclasses, nchannels[end])
    return ResNet(conv1, bn1, layers, output, :imagenet, stage, true)
end

function ResNetCifar(depth::Int; nclasses=10)
    conv1 = Conv(3, 3, 3, 16; bias=false, padding=1)
    bn1 = BatchNorm(16)
    n = Int((depth-2)/3)
    layers = [
        _make_layer(BasicBlock, 16, 16, n),
        _make_layer(BasicBlock, 16, 32, n; stride=2),
        _make_layer(BasicBlock, 32, 64, n; stride=2),
    ]
    output = Linear(nclasses, 64)
    return ResNet(conv1, bn1, layers, output, :cifar, true)
end

function forward(ctx, r::ResNet, x)
    # initial conv
    if r.dataset == :cifar
        o = relu.(@mc r.bn1(@mc r.conv1(x)))
    else
        o = pool(relu.(@mc r.bn1(@mc r.conv1(x)));
                 window=3, stride=2, padding=Int(stride3()))
    end
    # residual blocks
    for (i,l) in enumerate(r.layers)
        o = @mc l(o)
        r.stage == i && return o
    end
    # output layer
    if r.avg_pool
        o = pool(o; mode=2, window=size(o,1,2))
        r.stage == 5 && return o
    end
    if r.output !== nothing
        o = @mc r.output(o)
    end
    return o
end



