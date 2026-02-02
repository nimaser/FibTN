module TensorNetworks

export IndexLabel, IndexContraction, TensorLabel, TensorNetwork
export get_indices, get_tensor, get_contraction, find_indices, has_index
export add_tensor!, add_contraction!, remove_tensor! remove_contraction!
export get_groups, regroup, combine!

"""
An IndexLabel is a semantic identifier for a tensor index. It must be
unique within each TensorNetwork. Each IndexLabel has a group and a port.

Semantic here means that any two IndexLabels with the same group and
port are equal, even if they are different objects constructed at
different times and stored in different memory locations. This allows
lightweight and convenient symbolic construction of a tensor network.

- The group refers to the TensorLabel the index was originally attached
to, and all indices attached to the same TensorLabel must have the same
group. See TensorLabel for more details on this design decision.

- The port is any symbol which can be used by the user for the purpose of
creating contractions and must be unique within the TensorLabel it
belongs to. For example, a tensor intended for use in PEPS could have 
ports :L, :R, :U, :D, and :P (representing left, right, up, down, and 
physical indices).

An IndexLabel deliberately does not store the dimension or orientation
(with respect to some planar embedding) or any other metadata about an
index. Those details are left to the execution layer and the user. It is
only intended to identify an index and allow symbolic manipulation of it.
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
    
"""Returns an IndexLabel with the same port but specified group."""
regroup(il::IndexLabel, group::Int) = IndexLabel(group, il.port)

"""
An IndexContraction contains a pair of IndexLabels which are to be
contracted together. It abstractly represents a contraction without
detailing how or when that contraction occurs.

Because IndexLabels must be unique per network, two contracted indices
cannot be the same, and the IndexContraction enforces this.

Further, each IndexContraction stores its IndexLabels in a deterministic
ordering irrespective of the order they are provided to the constructor.
"""
struct IndexContraction
    a::IndexLabel
    b::IndexLabel
    function IndexContraction(a::IndexLabel, b::IndexLabel)
        if a == b error("labels of contracted indices mustn't match") end
        if b < a a, b = b, a end
        new(a, b)
    end
end

"""
A TensorLabel identifies a single tensor in a network. It has a group,
which must be unique in each TensorNetwork, and a list of IndexLabels
belonging to it.

Each TensorLabel's group must match the group value of each IndexLabel
assigned to it. Each IndexLabel assigned to it must also have a distinct
port value, or in other words, each IndexLabel must be unique.

A TensorLabel has no numerical data, and is used to describe the
structure of the tensor network rather than its evaluation.

Modification of a TensorLabel after construction is undefined behavior.
"""
struct TensorLabel
    """an identifier for this tensor"""
    group::Int
    """the indices that belong to this tensor"""
    indices::Vector{IndexLabel}
    function TensorLabel(group::Int, indices::Vector{IndexLabel})
        length(indices) == length(unique(indices)) || throw(ArgumentError("duplicate IndexLabel")
        for idx in indices
            idx.group == group || throw(ArgumentError("TensorLabel and IndexLabel group mismatch: $group vs $(idx.group)"))
        end
        new(group, indices)
    end
end

"""
Returns a new TensorLabel with its and its IndexLabels' group numbers
changed to the provided value. IndexLabel ordering and ports are
preserved.
"""
regroup(tl::TensorLabel, group::Int)
    = TensorLabel(group, [regroup(il, group) for il in tl.indices])

"""
A TensorNetwork is a lightweight symbolic representation of a tensor
network. It consists of a list of TensorLabels and a list of
IndexContractions.

Every index in a tensor network must be unique and belong to exactly one
TensorLabel. Every contracted index must appear in exactly one
IndexContraction. To enforce these invariants, the `tensors` and
`contractions` fields should only be modified through the add_tensor!
and add_contraction! functions, rather than direct array access.

A TensorNetwork only encodes the structure of a network, not the
evaluation details. For more details see the execution backends.
"""
struct TensorNetwork
    tensors::Vector{TensorLabel}
    contractions::Vector{IndexContraction}
    _tensor_with_index::Dict{IndexLabel, TensorLabel}
    _contraction_with_index::Dict{IndexLabel, IndexContraction}
    TensorNetwork() = new([], [], Dict(), Dict())
end

"""Get all groups in this TensorNetwork."""
get_groups(tn::TensorNetwork)
    = [t.group for t in tensors]

"""Get all IndexLabels belonging to this TensorNetwork."""
get_indices(tn::TensorNetwork)
    = keys(tn._tensor_with_index)
    
"""Get a TensorLabel from an IndexLabel which it owns."""
get_tensor(tn::TensorNetwork, il::IndexLabel)
    = tn._tensor_with_index[il]

"""Get a TensorLabel from its group number."""
get_tensor(tn::TensorNetwork, group::Int)
    = for t in tn.tensors t.group == group && return t end; throw(KeyError(group))
    
"""Get an IndexContraction from an IndexLabel which it owns."""
get_contraction(tn::TensorNetwork, il::IndexLabel)
    = tn._contraction_with_index[il]

"""Get all IndexLabels for which `f` returns true."""
function find_indices(f::Function, tn::TensorNetwork)
    matches = Vector{IndexLabel}()
    for idx in get_indices(tn)
        if f(idx) push!(matches, idx) end
    end
    matches
end

"""Get all IndexLabels with group `group`."""
find_indices(tn::TensorNetwork, group::Int)
    = find_indices(idx -> idx.group == group, tn)
    
"""Get all IndexLabels with port `port`."""
find_indices(tn::TensorNetwork, port::Symbol)
    = find_indices(idx -> idx.port == port, tn)
    
"""Return whether `tn` has the IndexLabel `il`."""
has_index(tn::TensorNetwork, il::IndexLabel)
    = !empty(find_indices(idx -> il.group == idx.group && il.port == idx.port, tn))
    
"""
Adds the TensorLabel `tl` to the TensorNetwork `tn`, ensuring that all
indices are unique.
"""
function add_tensor!(tn::TensorNetwork, tl::TensorLabel)
    # check IndexLabel uniqueness
    for il in tl.indices
        !haskey(tn._tensor_with_index, il) || throw(ArgumentError("duplicate IndexLabel $il"))
    end
    # add and adjust bookkeeping
    push!(tn.tensors, tl)
    for idx in tl.indices tn._tensor_with_index[idx] = tl end
end

"""
Adds the IndexContraction `ic` to the TensorNetwork `tn`, ensuring that
the referenced IndexLabels exist in the network and that they are not
already part of some other IndexContraction.
"""
function add_contraction!(tn::TensorNetwork, ic::IndexContraction)
    # check that indices are present and haven't yet been contracted
    haskey(tn._tensor_with_index, ic.a) || throw(ArgumentError("index $(ic.a) not found in network"))
    haskey(tn._tensor_with_index, ic.b) || throw(ArgumentError("index $(ic.b) not found in network"))
    !haskey(tn._contraction_with_index, ic.a) || throw(ArgumentError("index $(ic.a) has already been contracted"))
    !haskey(tn._contraction_with_index, ic.b) || throw(ArgumentError("idnex $(ic.b) has already been contracted"))
    # add and adjust bookkeeping
    push!(tn.contractions, ic)
    tn._contraction_with_index[ic.a] = ic
    tn._contraction_with_index[ic.b] = ic
end

"""
Removes the TensorLabel `tl` from the TensorNetwork `tn`, if there are
no contractions involving it. Errors if any contractions involving its
indices exist.
"""
function remove_tensor!(tn::TensorNetwork, tl::TensorLabel)
    # check that tensor has no contractions
    for idx in tl.indices
        !haskey(tn._contraction_with_index, idx) || throw(ArgumentError("tensor must not have contracted indices, found $(tn._contraction_with_index[idx])"))
    end
    # remove it
    filter!(!=(tl), tn.tensors)
    for idx in tl.indices delete!(tn._tensor_with_index, idx) end
end

"""
Removes the IndexContraction `ic` from the TensorNetwork `tn`. If the
contraction is not present, throws an error.
"""
function remove_contraction!(tn::TensorNetwork, ic::IndexContraction)
    # check that contraction exists
    ic \in tn.contractions || throw(ArgumentError("contraction $ic not in tensor network"))
    # remove it
    filter!(!=(ic), tn.contractions)
    delete!(tn._contraction_with_index, ic.a)
    delete!(tn._contraction_with_index, ic.b)
end

"""
Removes all contractions with any index belonging to the specified
TensorLabel `tl`.
"""
function remove_contractions!(tn::TensorNetwork, tl::TensorLabel)
    for idx in tl.indices
        if haskey(tn._contraction_with_index, idx)
            remove_contraction!(tn, tn._contraction_with_index[idx])
        end
    end
end

"""
Combines two tensor networks into a single one by merging the second
argument into the first. Acts like graph sum (disjoint union on graphs):

- all TensorLabels and IndexLabels in the second argument are regrouped
to prevent group number collison with the first argument's TensorLabels

- all existing contractions are preserved, and no new ones are created

Returns the modified TensorNetwork and the mapping from old to new group
numbers for the second argument's TensorLabels.
"""
function combine!(tn1::TensorNetwork, tn2::TensorNetwork)
    # get mapping from extant tn2 groups to unused tn1 groups
    group_map = Dict{Int, Int}()
    extant_groups = Set(get_groups(tn1))
    new_group = 1
    for i in get_groups(tn2)
        while new_group \in extant_groups new_group += 1 end
        group_map[i] = new_group
    end
    # add tn2 tensors to tn1
    for (old, new) in group_map
        add_tensor!(tn1, regroup(get_tensor(tn2, old), new))
    end
    # add tn2 contractions to tn1
    for c in tn2.contractions
        a = regroup(oldc.a, group_map[oldc.a.group])
        b = regroup(oldc.b, group_map[oldc.b.group])
        add_contraction!(tn1, IndexContraction(a, b))
    end
    tn1
end

end # module TensorNetworks
