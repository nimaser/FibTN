module TensorNetworks

export IndexLabel, IndexPair
export TensorLabel, TensorNetwork, add_tensor!, add_contraction!
export tensor_with_group, indices, find_indices, find_indices_by_group, find_indices_by_port, find_index

"""
An IndexLabel is a semantic identifier for a tensor index. It must be
unique within each TensorNetwork.

Semantic here means that any two IndexLabels with the same group and
port are equal, even if they are different objects constructed at
different times and stored in different memory locations. This allows
lightweight and convenient symbolic construction of a tensor network.

The group refers to the TensorLabel the index was originally attached
to, and all indices attached to the same TensorLabel must have the same
group. See TensorLabel for more details on this design decision.

The port is any symbol which can be used by the user for the purpose of
creating contractions, through the use of IndexPairs. For example, a
tensor intended for use in PEPS could have ports :L, :R, :U, :D, and :P
(representing left, right, up, down, and physical indices).

An IndexLabel deliberately does not store the dimension or orientation
(with respect to some planar embedding) or any other metadata about an
index. Those details are left to the execution layer. It is only
intended to identify an index and allow symbolic manipulation of it.
"""
struct IndexLabel
    """the tensor this index was originally attached to"""
    group::Int
    """a name for this index with respect to its tensor"""
    port::Symbol
end

"""Lexicographic total order, with group more important than port."""
Base.isless(a::IndexLabel, b::IndexLabel)
    = a.group < b.group || (a.group == b.group && a.port < b.port)

"""
An IndexPair contains a pair of IndexLabels which are to be contracted
together. In essence, it abstractly represents a contraction, without
detailing how or when that contraction occurs.

Because IndexLabels must be unique per network, two contracted indices
cannot be the same, and the IndexPair enforces this.

Further, each IndexPair stores its IndexLabels in a deterministic
ordering irrespective of the order they are provided to the constructor.
"""
struct IndexPair
    a::IndexLabel
    b::IndexLabel
    function IndexPair(a, b)
        if a == b error("labels of contracted indices mustn't match") end
        if b < a a, b = b, a end
        new(a, b)
    end
end

"""
A TensorLabel identifies a single tensor in a network. It has a group,
which must be unique in each TensorNetwork, and a list of IndexLabels
belonging to it.

The constructor enforces that each TensorLabel's group must match the
group value of each IndexLabel assigned to it.

A TensorLabel has no numerical data, and is only intended to describe
the structure of the tensor network, rather than its evaluation.
"""
struct TensorLabel
    """an identifier for this tensor"""
    group::Int
    """the indices that belong to this tensor"""
    indices::Vector{IndexLabel}
    function TensorLabel(group, indices)
        for idx in indices
            idx.group == group || error("TensorLabel group must be the same as each of its indices")
        end
        new(group, indices)
    end
end

"""
A TensorNetwork is a lightweight yet complete symbolic representation of
a tensor network. It consists of a list of TensorLabels and a list of
IndexPairs (describing contractions).

Every index in a tensor network must be unique and belong to exactly one
TensorLabel. Every contracted index appears in exactly one IndexPair.


"""
struct TensorNetwork
    tensors::Vector{TensorLabel}
    contractions::Vector{IndexPair}
    tensor_with_index::Dict{IndexLabel, TensorLabel}
    contraction_with_index::Dict{IndexLabel, IndexPair}
    _index_use_count::Dict{IndexLabel, UInt}
    TensorNetwork() = new([], [], Dict(), Dict(), Dict())
end

function add_tensor!(tn::TensorNetwork, tl::TensorLabel)
    # check that all indices are unique
    for idx in tl.indices
        if haskey(tn._index_use_count, idx) error("index $idx already in tensor network") end
    end
    # add tensor and update bookkeeping
    push!(tn.tensors, tl)
    for idx in tl.indices
        tn.tensor_with_index[idx] = tl
        tn._index_use_count[idx] = 1
    end
end

function add_contraction!(tn::TensorNetwork, ip::IndexPair)
    # check that indices are present
    if !haskey(tn._index_use_count, ip.a) error("index $(ip.a) not found in network") end
    if !haskey(tn._index_use_count, ip.b) error("index $(ip.b) not found in network") end
    # check that indices have not already been contracted
    if tn._index_use_count[ip.a] > 1 error("index $(ip.a) has already been contracted") end
    if tn._index_use_count[ip.b] > 1 error("index $(ip.b) has already been contracted") end
    # add index pair and adjust bookkeeping
    push!(tn.contractions, ip)
    tn.contraction_with_index[ip.a] = ip
    tn.contraction_with_index[ip.b] = ip
    tn._index_use_count[ip.a] += 1
    tn._index_use_count[ip.b] += 1
end

function tensor_with_group(tn::TensorNetwork, group::Int)
    for t in tn.tensors
        if t.group == group return t end
    end
    nothing
end

indices(tn::TensorNetwork) = keys(tn.tensor_with_index)

function find_indices(f::Function, tn::TensorNetwork)
    matches = IndexLabel[]
    for idx in indices(tn)
        if f(idx) push!(matches, idx) end
    end
    matches
end

find_indices_by_group(tn::TensorNetwork, group::Int) = find_indices(idx -> idx.group == group, tn)
find_indices_by_port(tn::TensorNetwork, port::Symbol) = find_indices(idx -> idx.port == port, tn)

function find_index(tn::TensorNetwork, group::Int, port::Symbol)
    matches = find_indices(idx -> idx.group == group && idx.port == port, tn)
    if length(matches) != 1 error("no index found with group $group and port $port") end
    only(matches)
end

function combine(tn1::TensorNetwork, tn2::TensorNetwork)
    # rename
    # rename groups in tn2
    # rebuild indices
    # merge tensors and bookkeeping
end

end # module TensorNetworks
