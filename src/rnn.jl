using Knet:RNN

"""Abstract type for RNN modules"""
abstract type AbstractRNN <: KnetModule end


for (RNNModule, rnnType) in zip([:RNNRelu, :RNNTanh, :LSTM, :GRU],
                                [:(:relu), :(:tanh), :(:lstm), :(:gru)])

    eval(:(begin
      type $(RNNModule) <: AbstractRNN
           w::Param
           cfg::RNN
           y::Bool
           hy::Bool
           cy::Bool
           _opt
      end

     rname = split(string($(RNNModule)), ".")[end]
"""        
# Constructor
    $(rname)(input::Int, hidden::Int; y=true, o...)
        `y` determines whether or not to return output (state history)
        `o` might contain all keyword arguments to `Knet.rnnforw` and `Knet.rnninit`

# Fields
    `w::Param` stores trainable parameters
    `cfg::Knet.RNN` an object stores the RNN configuration
    `y::Bool` stores whether or not to return output (state history)
    `hy::Bool` stores whether or not to return next hidden state
    `cy::Bool` stores whether or not to return next cell state

# Forward execution
    r(ctx, x[, hy, cy]; o...)
where `o...` is runtime kwargs of `Knet.rnnforw`
"""
      function $(RNNModule)(input::Int, hidden::Int;
                            y=true, hy=false, cy=false,
                            o...)
           @assert (y || hy || cy) "You must return something"
           r, w = rnninit(input, hidden;
                          usegpu=false, rnnType=$(rnnType),o...)
           return $(RNNModule)(Param(w), r, y, hy, cy, o)
      end

      (rnn::$(RNNModule))(ctx, x, hx=nothing, cx=nothing; o...) =
           _forw(ctx, rnn, x, hx, cx; o...)

     end))
end

import Knet.rnnparams

"`rnnparams(r::AbstractRNN; o...)` gets parameter matrices and biases of module `r`"
rnnparams(r::AbstractRNN; o...) = rnnparams(r.cfg, aval(r.w); o...)

function _forw(ctx, rnn::AbstractRNN, x, hx, cx; o...)
    y, hy, cy, _ = rnnforw(rnn.cfg, val(ctx, rnn.w), x, hx, cx;
                           hy=rnn.hy, cy=rnn.cy, o...)
    # Post-process state
    gets = rnn.hy || rnn.cy
    next_state = (rnn.cy && cy !== nothing) ? (hy, cy) : hy
    return (gets && rnn.y) ? (y, next_state...) : gets ? next_state : y
end

function convert_params!(rnn::AbstractRNN, atype)
    r,w = rnninit(rnn.cfg.inputSize, rnn.cfg.hiddenSize;
                  usegpu=atype<:KnetArray,
                  rnn._opt...)
    rnn.cfg = r
    setval!(rnn.w, w) 
end
