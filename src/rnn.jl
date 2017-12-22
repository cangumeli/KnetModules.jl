using Knet:RNN

"""Abstract type for RNN modules"""
abstract type AbstractRNN <: KnetModule end


for (RNNModule, rnnType) in zip([:RNNRelu, :RNNTanh, :LSTM, :GRU],
                                [:(:relu), :(:tanh), :(:lstm), :(:gru)])

    eval(:(begin
      type $(RNNModule) <: AbstractRNN
           w::Param
           cfg::RNN
           gety::Bool
           opt
      end
           
      function $(RNNModule)(input::Int, hidden::Int;
                          y=true,
                          o...)
           r, w = rnninit(input, hidden; o..., rnnType=$(rnnType))
           return $(RNNModule)(Param(w), r, y, o)
      end
     end))
end

import Knet.rnnparams
rnnparams(r::AbstractRNN; o...) = rnnparams(r.cfg, aval(r.w); o...)

function forward(ctx, rnn::AbstractRNN, seq, prev_state=(nothing, nothing))
    # Process sequence
    x, bsizes = isa(seq, Tuple) ? seq : (seq, nothing)
    # Process the previous state
    if ~isa(prev_state, Tuple)
        prev_state = (prev_state, nothing)
    end
    y, hy, cy, _ = rnnforw(rnn.cfg, val(ctx, rnn.w), x, prev_state...;
                           hy=true, cy=true,
                           rnn.opt..., batchSizes=bsizes)
    # Post-process state
    gets = (hy !== nothing)
    @assert (gets || rnn.gety) "No output can be returned, check gety and opt options"
    next_state = cy!==nothing ? (hy, cy) : hy
    return (gets && rnn.gety) ? (y, next_state...) : gets ? next_state : y
end
