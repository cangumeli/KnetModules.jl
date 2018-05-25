module KnetModules

using Knet, AutoGrad

include("core.jl")
export
    # Ctx
    ParamCtx, active_ctx, switch_ctx!, reset_ctx!, actx,
    # Param
    Param, aval, val, setval!,
    # KnetModule
    KnetModule, getgrad, params, modules,
    convert_params!, convert_buffers!,
    gpu!, cpu!, training!, testing!,
    clean_ctx!, is_ctx_clean,
    save_module, load_module


include("linear.jl")
export Linear


include("conv.jl")
export Conv, kaiming, conv_mode!


include("norm.jl")
export Norm, BatchNorm


include("func.jl")
export
    FnModule,
    Activation, ReLU, Sigmoid, Tanh,
    Pool, MaxPool, AvgPool,
    Dropout


include("container.jl")
export KnetContainer, Sequential, add!


include("rnn.jl")
export AbstractRNN, RNNTanh, RNNRelu, LSTM, GRU


include("embedding.jl")
export Embedding, EmbeddingMul, EmbeddingLookup


include("data.jl")

end #module
