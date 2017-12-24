module KnetModules

using Knet

include("core.jl")
export
    # Ctx
    ParamCtx, active_ctx, switch_ctx!, default_ctx, reset_ctx!,
    # Param
    Param, aval, val, setval!,
    # KnetModule
    KnetModule, getgrad, params, submodules,
    gpu!, cpu!, training!, testing!,
    forward, @mc, @run,
    save_module, load_module, restore_module!


include("linear.jl")
export Linear


include("conv.jl")
export Conv, kaiming, conv_mode, mode_conv!, mode_cross!


include("norm.jl")
export AbstractNorm, BatchNorm


include("container.jl")
export KnetContainer, Sequential


include("func.jl")
export
    FnModule,
    Activation, ReLU, Sigmoid, Tanh,
    Pool, MaxPool, AvgPool,
    Dropout


include("rnn.jl")
export AbstractRNN, RNNTanh, RNNRelu, LSTM, GRU


include("embedding.jl")
export AbstractEmbedding, Embedding, EmbeddingLookup


include("data.jl")

end #module
