using Knet, KnetModules
import KnetModules.forward

function vgg16(;trained=true, o...)
    model = VGG([64, 128, 256, 512, 512],
                [2,  2,   3,   3,   3];
                o...)
    # TODO: load pre-trained weights
    return model
end

function vgg19(;trained=true, o...)
    model = VGG([64, 128, 256, 512, 512],
                [2,  2,   4,   4,   4];
                o...)
    return model
end

type VGG <: KnetModule
    conv::Sequential
    fc::Union{Sequential, Void}
    top::Union{Linear, Void}
end

function VGG(channels::Array{Int,1}, repeats::Array{Int,1};
             fc=true, top=true, pdrop=0.5)
    @assert (fc || ~top) "Top cannot exist without fc layers"
    # Conv
    conv = Sequential()
    i = 3
    for (c, r) in zip(channels, repeats)
        block = Sequential()
        for _ = 1:r
            add!(block, Conv(3, 3, i, c; padding=1))
            add!(block, ReLU())
            i = c
        end
        add!(block, MaxPool())
        add!(conv, block)
    end
    # FC
    fcin = i * div(224, 2^length(channels))^2
    fc = fc ? Sequential(
        Linear(4096, fcin), ReLU(), Dropout(pdrop),
        Linear(4096, 4096), ReLU(), Dropout(pdrop)
    ) : nothing
    # output
    top = top ? Linear(1000, 4096) : nothing
    return VGG(conv, fc, top)
end

function forward(ctx, vgg::VGG, x)
    o = @mc vgg.conv(x)
    if vgg.fc !== nothing
        o = @mc vgg.fc(o)
        if vgg.top !== nothing
            o = @mc vgg.top(o)
        end
    end
    return o
end

