module IndexTriplets

export combine_qubits, split_index, permute_index_mapping
export QubitEmbedding

"""
Takes three qubit values (anyon types) `i`, `j`, and `k` which
can take values 0 or 1 each and maps them to a single combined physical index.

Combinations which do not violate the fusion rules are mapped to 1:5,
while combinations which do are mapped to 6:8.
"""
function combine_qubits(i::Int, j::Int, k::Int)
    # fusion rule violating cases
    if i + j + k == 1
        if i == 1
            a = 6
        end
        if j == 1
            a = 7
        end
        if k == 1
            a = 8
        end
        return a
    end
    # nonviolating cases
    if (i == j == k)
        a = (i == 0) ? 1 : 5
    elseif (i == 0)
        a = 2
    elseif (j == 0)
        a = 3
    elseif (k == 0)
        a = 4
    end
    a
end

"""
Takes a combined index value `a` from 1:8 and maps it to three
individual qubit values. When `a` ∈ 1:5 it obeys the fusion
constraints, but when `a` ∈ 6:8 it violates them.
"""
function split_index(a::Int)
    # fusion rule violating cases
    if a > 5
        i = j = k = 0
        if a == 6
            i = 1
        end
        if a == 7
            j = 1
        end
        if a == 8
            k = 1
        end
        return i, j, k
    end
    # nonviolating cases
    i = j = k = 1
    if a == 1
        i = j = k = 0
    elseif a == 2
        i = 0
    elseif a == 3
        j = 0
    elseif a == 4
        k = 0
    end
    i, j, k
end

"""
Permutes the index mapping produced by `combine_qubits`:
in other words, if three qubits are passed into it in some
order, and you wish to know the combined index if you were
to instead encode them in a different order, you can use
this function to safely permute them and obtain the new
combined index.

This function is useful when a physical index has encoded
qubits in a geometrically problematic way, and you wish to
change the encoding to a more convenient one.
"""
function permute_index_mapping(a::Int, p::NTuple{3,Int})
    # check permutation validity
    for i in p
        i ∈ collect(1:3) || throw(ArgumentError("invalid index $i in permutation $p"))
    end
    collect(p) == unique(p) || throw(ArgumentError("invalid permutation $p: duplicate index"))
    # permute
    vals = collect(split_index(a))
    permute!(vals, collect(p))
    combine_qubits(vals...)
end

# """
# Low-level embedding of `nqubits` qubits into `N` physical indices for conversion of operators on qubits to
# operators on physical indices. Designed to be compiler-friendly by avoiding dictionary lookups or dynamic
# allocations.

# `map[i]` is a `NTuple{3,Int}` representing the qubits encoded by physical index `i`:
#   `map[i][p] = q` means position `p` of physical index `i` encodes operated-on qubit `q`
#   `map[i][p] = 0` means position `p` of physical index `i` encodes a qubit that should be unaltered by the operator
# """
# struct QubitEmbedding{N}
#     map::NTuple{N,NTuple{3,Int}}
#     nqubits::Int
# end

# function generate_nindex_operator!(
#     ::Val{N},
#     dat::AbstractArray{<:Number},
#     embed::QubitEmbedding{N};
#     arr::AbstractArray = nothing
# ) where {N}

#     n = embed.nqubits

#     # Check operator dimensions
#     expected_dims = ntuple(_ -> 2, 2n)
#     size(U) == expected_dims ||
#         throw(ArgumentError("U must have size $(expected_dims)"))

#     # Allocate output if needed
#     if arr === nothing
#         dims = ntuple(_ -> 8, 2N)
#         arr = zeros(eltype(U), dims)
#     else
#         # Check dimensions
#         size(arr) == ntuple(_ -> 8, 2N) ||
#             throw(ArgumentError("arr must have size $(ntuple(_->8,2N))"))

#         # Require zero-filled
#         any(!iszero, arr) &&
#             throw(ArgumentError("arr must be zero-filled"))
#     end

#     # Iterate over logical qubit assignments
#     # input and output bitstrings 0:(2^n - 1)
#     for in_mask in 0:(2^n - 1), out_mask in 0:(2^n - 1)

#         # Extract logical qubit bits
#         qin  = ntuple(i -> (in_mask  >> (i-1)) & 1, n)
#         qout = ntuple(i -> (out_mask >> (i-1)) & 1, n)

#         # Query operator
#         Uval = U[(qin .+ 1)..., (qout .+ 1)...]
#         iszero(Uval) && continue

#         # Build physical input/output indices
#         phys_in  = ntuple(i -> begin
#             triple = ntuple(p -> begin
#                 q = embed.map[i][p]
#                 q == 0 ? 0 : qin[q]
#             end, 3)
#             combine_qubits(triple...)
#         end, N)

#         phys_out = ntuple(i -> begin
#             triple = ntuple(p -> begin
#                 q = embed.map[i][p]
#                 q == 0 ? 0 : qout[q]
#             end, 3)
#             combine_qubits(triple...)
#         end, N)

#         # Write into array
#         arr[phys_in..., phys_out...] = Uval
#     end

#     return arr
# end



end # module IndexTriplets
