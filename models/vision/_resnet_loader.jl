# This script has a heavy dependency on PyCall and PyTorch,
# so we should re-distribute the model and make this unnecessary
using PyCall
@pyimport torchvision.models as models

const model_dict = Dict([
        101=>models.resnet101,
        50=>models.resnet50,
        152=>models.resnet152,
        18=>models.resnet18,
        34=>models.resnet34
])


function load_resnet!(model, depth=101)
    conv_mode!(model, 1)
    stride3!()
    ptdict = state_dict(models, depth)    
    #println(keys(ptdict))
    parse_conv!(model.conv1, ptdict, "conv1")
    parse_bn!(model.bn1, ptdict, "bn1")
    # Load the layers
    for i = 1:length(model.layers)
        seq = model.layers[i]
        for j = 1:length(seq.layers)
            bneck = seq.layers[j]
            key = string("layer",i, ".", j-1)
            for k = 1:(isa(bneck, Bottleneck) ? 3 : 2) #convs and bnorms
                ckey = string(key, ".conv",k)
                bkey = string(key, ".bn", k)
                parse_conv!(
                    getfield(bneck, Symbol(string("conv", k))),
                    ptdict, ckey)
                parse_bn!(
                    getfield(bneck, Symbol(string("bn", k))),
                    ptdict, bkey)
            end
            if bneck.downsample !== nothing
                conv = bneck.downsample.layers[1]
                bn = bneck.downsample.layers[2]
                parse_conv!(conv, ptdict, string(
                    key, ".", "downsample.0"
                ))
                parse_bn!(bn, ptdict, string(
                    key, ".", "downsample.1"
                ))
            end
        end
    end
    if model.output !== nothing
        parse_linear!(model.output, ptdict, "fc")
    end
end


state_dict(models, depth) = 
    model_dict[depth](pretrained=true)[:state_dict]()

function parse_linear!(lin, ptdict, key)
    setval!(lin.w, reshape(
        ptdict[string(key, ".weight")][:numpy](),
        size(aval(lin.w))))

    setval!(lin.b, reshape(
        ptdict[string(key, ".bias")][:numpy](),
        size(aval(lin.b))))
end

function parse_conv!(conv, ptdict, key)
    setval!(conv.w, permutedims(
        ptdict[string(key, ".weight")][:numpy](), (4, 3, 2, 1)))
end

function parse_bn!(bn, ptdict, key)
    setval!(bn.w, vcat(ptdict[string(key, ".weight")][:numpy]()[:],
                       ptdict[string(key, ".bias")][:numpy]()[:]))
    sz = (1, 1, div(length(aval(bn.w)), 2), 1)
    bn.moments.mean = 
        reshape(ptdict[string(key, ".running_mean")][:numpy](), sz)
    bn.moments.var = 
        reshape(ptdict[string(key, ".running_var")][:numpy](), sz)
    if bn.remove_initfns
        bn.moments.meaninit = nothing
        bn.moments.varinit  = nothing
    end
end



