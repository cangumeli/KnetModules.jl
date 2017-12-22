"""Abstract type generalizes the embedding modules"""
abstract type AbstractEmbedding <: KnetModule end


"""
`Embedding`: Basic linear embedding layer.

# Constructors
    `Embedding(esize::Int, vsize::Int; train)`
    esize is the embedding size, 
    vsize is the vocabulary size
    
    `Embedding(w; train)`
    Create embedding from an existing buffer.

train is a boolean, determines whether or not to train the embedding

# Fields
    `l::Linear`: Linear embedding operator.
    `train::Bool`: Determines whether or not to train embedding

# Forward execution
    `forward(ctx, e::Embedding, seq)
    `@mc e(x)`
    `@run e(x)`

`seq` can be a tupple of `(x, metadata)` or an array `x`.
If `x` is 2d, `(size, batchsize, time)` is assumed.
    
"""
type Embedding <: AbstractEmbedding
    l::Linear
    train::Bool
end

Embedding(esize::Int, vsize::Int; train=true) =
    Embedding(Linear(esize,vsize; bias=false), train)

function Embedding(w; train=false)
    l = Linear(size(w)...; bias=false)
    copy!(aval(l.w), w)
    return Embedding(l, train)
end
    

function forward(ctx, e::Embedding, seq)
    x, bs = isa(seq, Tuple) ? seq : (seq, nothing)
    if ndims(x) == 3
        dims = size(x, 2, 3)
        x = (reshape(x, size(x,1), size(x, 2) * size(x, 3)))
    else
        dims = size(x, 2)
    end
    emb = e.train ? (@mc e.l(x)) : (@run e.l(x))
    y = reshape(emb, (size(x,1), dims...))
    return bs == nothing ? y : (y, bs)
end



"""
`Embedding`: Basic linear embedding layer.

# Constructors
    `EmbeddingLookup(esize::Int, vsize::Int; train)`
    esize is the embedding size, 
    vsize is the vocabulary size
    
    `EmbeddingLookup(w; train)`
    Create embedding from an existing buffer.

train is a boolean, determines whether or not to train the embedding

# Fields
    `w`: Embedding vector
    

# Forward execution
    `forward(ctx, e::Embedding, indices::Array{Int,1})
    `@mc e(indices)`
    `@run e(indices)`

`seq` can be a tupple of `(x, metadata)` or an array `x`.
If `x` is 2d, `(size, batchsize, time)` is assumed.
    
"""
type EmbeddingLookup <: AbstractEmbedding
    w
    train
end

EmbeddingLookup(w; train=true) =
    EmbeddingLookup(Param(w), train)

EmbeddingLookup(emb::Int, vocab::Int; dtype=Float32, train=true) =
    EmbeddingLookup(xavier(dtype, emb, vocab), train)


function forward(ctx, el::EmbeddingLookup, indices::Array{Int, 1})
    w = el.train ? val(ctx, el.w) : aval(el.w)
    return w[:, indices]
end

