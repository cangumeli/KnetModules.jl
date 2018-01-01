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
    `forward(ctx, rnn::$(rname), seq[, prev_state])`
    `@mc rnn(seq[, prev_state]) `
    `@run rnn(seq[, prev_state])`
`seq` can be a 1d, 2d or 3d tensor or a (tensor, batchSizes) pair. See `batchSizes` kwarg in `Knet.rnnforw` for details. `prev_state` is optional initial state. It can be only the hidden state,
or tuple of (hidden_state, cell_state), if the module has any cell states.
"""
      function $(RNNModule)(input::Int, hidden::Int;
                            y=true, hy=false, cy=false,
                            o...)
           @assert (y || hy || cy) "You must return something"
           r, w = rnninit(input, hidden; o..., rnnType=$(rnnType))
           return $(RNNModule)(Param(w), r, y, hy, cy)
      end
     end))
end

import Knet.rnnparams

"`rnnparams(r::AbstractRNN; o...)` gets parameter matrices and biases of module `r`"
rnnparams(r::AbstractRNN; o...) = rnnparams(r.cfg, aval(r.w); o...)

function forward(ctx, rnn::AbstractRNN, seq, prev_state=(nothing, nothing))
    # Process sequence
    x, bsizes = isa(seq, Tuple) ? seq : (seq, nothing)
    # Process the previous state
    if ~isa(prev_state, Tuple)
        prev_state = (prev_state, nothing)
    end
    y, hy, cy, _ = rnnforw(rnn.cfg, val(ctx, rnn.w), x, prev_state...;
                           hy=rnn.hy, cy=rnn.cy, batchSizes=bsizes)
    # Post-process state
    gets = rnn.hy || rnn.cy
    next_state = (rnn.cy && cy !== nothing) ? (hy, cy) : hy
    return (gets && rnn.y) ? (y, next_state...) : gets ? next_state : y
end
