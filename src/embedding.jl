"""Abstract type generalizes the embedding modules"""
abstract type Embedding <: KnetModule end


"""
`EmbeddingMul`: Basic linear embedding layer using matrix multiplication
for embedding.

# Constructors
    `EmbeddingMul(esize::Int, vsize::Int; winit)`
    esize is the embedding size, 
    vsize is the vocabulary size

# Fields
    `l::Linear`: Linear embedding operator.

# Forward execution
    `forward(ctx, e::EmbeddingMul, seq)
    `@mc e(x)`
    `@run e(x)`

If `x` is 3d, `(size, batchsize, time)` is assumed.
    
"""
type EmbeddingMul <: Embedding
    l::Linear
end

EmbeddingMul(esize::Int, vsize::Int; winit=randn, dtype=Float32) =
    EmbeddingMul(Linear(esize, vsize;
                        winit=winit, dtype=Float32, bias=false))

function forward(ctx, e::EmbeddingMul, x)
    if ndims(x) == 3
        dims = size(x, 2, 3)
        x = reshape(x, size(x,1), size(x, 2) * size(x, 3))
    else
        dims = size(x, 2)
    end
    emb = @mc e.l(x)
    return reshape(emb, (size(emb,1), dims...))
end



"""
`EmbeddingLookup`: Basic linear embedding layer using index access.

# Constructors
    `EmbeddingLookup(esize::Int, vsize::Int; train)`
    esize is the embedding size, 
    vsize is the vocabulary size

train is a boolean, determines whether or not to train the embedding

# Fields
    `w`: Embedding vector
    

# Forward execution
    `forward(ctx, e::EmbeddingLookup, indices::Array{Int,1})
    `@mc e(indices)`
    `@run e(indices)`
    
"""
type EmbeddingLookup <: Embedding
    w::Param
end

EmbeddingLookup(emb::Int, vocab::Int; dtype=Float32, winit=randn) =
    EmbeddingLookup(Param(winit(dtype, emb, vocab)))


function forward(ctx, el::EmbeddingLookup, indices::Array{Int})
    w = val(ctx, el.w)
    emb = w[:, indices[:]]
    if ndims(indices) > 1
        emb = reshape(emb, (size(emb, 1), size(indices)...))
    end
    return emb
end

