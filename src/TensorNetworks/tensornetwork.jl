export IndexLabel, IndexContraction, get_partner, TensorLabel, TensorNetwork
export get_tensors, get_tensor, get_contractions, get_contraction, has_contraction
export get_groups, get_indices, find_indices, find_contracted, find_uncontracted
export has_index, can_contract
export add_tensor!, add_contraction!, try_contraction!
export remove_tensor!, remove_contraction!, remove_contractions!, replace_tensor!
export get_groups, regroup, combine!, matchcombine!

"""
An `IndexLabel` is a semantic identifier for a tensor index. It must be
unique within each TensorNetwork. Each `IndexLabel` has a `group` and a `port`.

Semantic here means that any two `IndexLabel`s with the same `group` and
`port` are equal, even if they are different objects constructed at
different times and stored in different memory locations. This allows
lightweight and convenient symbolic construction of a tensor network.

- The group refers to the `TensorLabel` the index was originally attached
to, and all indices attached to the same `TensorLabel` must have the same
`group`. See `TensorLabel` for more details on this design decision.

- The `port` is any symbol which can be used by the user for the purpose of
creating contractions and must be unique within the `TensorLabel` it
belongs to. For example, a tensor intended for use in PEPS could have
ports :L, :R, :U, :D, and :P (representing left, right, up, down, and
physical indices).

An `IndexLabel` deliberately does not store the dimension or orientation
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

"""Lexicographic total order, with `group` more important than `port`."""
Base.isless(a::IndexLabel, b::IndexLabel) =
    a.group < b.group || (a.group == b.group && a.port < b.port)

"""Returns an `IndexLabel` with the same `port` but specified `group`."""
regroup(il::IndexLabel, group::Int) = IndexLabel(group, il.port)

"""
An `IndexContraction` contains a pair of `IndexLabels` which are to be
contracted together. It abstractly represents a contraction without
detailing how or when that contraction occurs.

Because `IndexLabels` must be unique per network, two contracted indices
cannot be the same, and the `IndexContraction` enforces this.

Further, each `IndexContraction` stores its `IndexLabels` in a deterministic
ordering irrespective of the order they are provided to the constructor.
"""
struct IndexContraction
    a::IndexLabel
    b::IndexLabel
    function IndexContraction(a::IndexLabel, b::IndexLabel)
        if a == b
            throw(ArgumentError("labels of contracted indices mustn't match"))
        end
        if b < a
            a, b = b, a
        end
        new(a, b)
    end
end

"""Returns the `IndexLabel` contracted with `il`."""
get_partner(ic::IndexContraction, il::IndexLabel) =
    ic.a == il ? ic.b : ic.a

"""
A `TensorLabel` identifies a single tensor in a network. It has a `group`,
which must be unique in each `TensorNetwork`, and a list of `IndexLabel`s
belonging to it.

Each `TensorLabel`'s group must match the group value of each `IndexLabel`
assigned to it. Each `IndexLabel` assigned to it must also have a distinct
port value, or in other words, each `IndexLabel` must be unique.

A `TensorLabel` has no numerical data, and is used to describe the
structure of the tensor network rather than its evaluation.

Modification of a `TensorLabel` after construction is undefined behavior.
"""
struct TensorLabel
    """an identifier for this tensor"""
    group::Int
    """the indices that belong to this tensor"""
    indices::Vector{IndexLabel}
    function TensorLabel(group::Int, indices::Vector{IndexLabel})
        length(indices) == length(unique(indices)) || throw(ArgumentError("duplicate IndexLabel"))
        for idx in indices
            idx.group == group || throw(ArgumentError("TensorLabel and IndexLabel group mismatch: $group vs $(idx.group)"))
        end
        new(group, indices)
    end
end

"""
Returns a new `TensorLabel` with its and its `IndexLabel`s' group numbers
changed to the provided value. `IndexLabel` ordering and ports are
preserved.
"""
regroup(tl::TensorLabel, group::Int) =
    TensorLabel(group, [regroup(il, group) for il in tl.indices])

"""
A `TensorNetwork` is a lightweight symbolic representation of a tensor
network. It consists of a list of `TensorLabel`s and a list of
`IndexContraction`s.

Every index in a tensor network must be unique and belong to exactly one
`TensorLabel`. Every contracted index must appear in exactly one
`IndexContraction`. To enforce these invariants, the `tensors` and
`contractions` fields should only be modified through the `add_tensor!`
and `add_contraction!` functions, rather than direct array access.

A `TensorNetwork` only encodes the structure of a network, not the
evaluation details. For more details see the execution backends.
"""
struct TensorNetwork
    _tensors::Vector{TensorLabel}
    _contractions::Vector{IndexContraction}
    _tensor_with_index::Dict{IndexLabel,TensorLabel}
    _contraction_with_index::Dict{IndexLabel,IndexContraction}
    TensorNetwork() = new([], [], Dict(), Dict())
end

### GETTERS ###

"""Get all `TensorLabel`s in `tn`."""
get_tensors(tn::TensorNetwork) =
    tn._tensors

"""Get the `TensorLabel` which owns `il`."""
get_tensor(tn::TensorNetwork, il::IndexLabel) =
    tn._tensor_with_index[il]

"""Get the `TensorLabel` with `group`."""
get_tensor(tn::TensorNetwork, group::Int) = begin
    for t in tn._tensors
        t.group == group && return t
    end
    throw(KeyError(group))
end

"""Get all `IndexContraction`s in `tn`."""
get_contractions(tn::TensorNetwork) =
    tn._contractions

"""Get the `IndexContraction` which owns `il`."""
get_contraction(tn::TensorNetwork, il::IndexLabel) =
    tn._contraction_with_index[il]

"""Return whether the IndexLabel `il` has been contracted in `tn`."""
has_contraction(tn::TensorNetwork, il::IndexLabel) =
    haskey(tn._contraction_with_index, il)

"""Get all `groups` in `tn`."""
get_groups(tn::TensorNetwork) =
    [t.group for t in get_tensors(tn)]

"""Get a `KeySet` of all `IndexLabel`s belonging to this `TensorNetwork`."""
get_indices(tn::TensorNetwork) =
    keys(tn._tensor_with_index)

"""Return whether `tn` has the IndexLabel `il`."""
has_index(tn::TensorNetwork, il::IndexLabel) =
    haskey(tn._tensor_with_index, il)

"""Return whether `tn` has `il` and `il` is uncontracted."""
can_contract(tn::TensorNetwork, il::IndexLabel) =
    has_index(tn, il) && !has_contraction(tn, il)

"""Get all IndexLabels for which `f` returns true."""
function find_indices(f::Function, tn::TensorNetwork)
    matches = Vector{IndexLabel}()
    for idx in get_indices(tn)
        if f(idx)
            push!(matches, idx)
        end
    end
    matches
end

"""Get all IndexLabels with group `group`."""
find_indices(tn::TensorNetwork, group::Int) =
    find_indices(idx -> idx.group == group, tn)

"""Get all IndexLabels with port `port`."""
find_indices(tn::TensorNetwork, port::Symbol) =
    find_indices(idx -> idx.port == port, tn)

"""Get all `IndexLabel`s which are part of a contraction."""
find_contracted(tn::TensorNetwork) =
    find_indices(il -> has_contraction(tn, il), tn)

"""Get all `IndexLabel`s which are not part of a contraction."""
find_uncontracted(tn::TensorNetwork) =
    find_indices(il -> !has_contraction(tn, il), tn)

### MUTATORS ###

"""
Adds the TensorLabel `tl` to the TensorNetwork `tn`, ensuring that all
indices are unique.
"""
function add_tensor!(tn::TensorNetwork, tl::TensorLabel)
    # check IndexLabel uniqueness
    for il in tl.indices
        !has_index(tn, il) || throw(ArgumentError("duplicate IndexLabel $il"))
    end
    # add and adjust bookkeeping
    push!(tn._tensors, tl)
    for idx in tl.indices
        tn._tensor_with_index[idx] = tl
    end
    nothing
end

"""
Adds the IndexContraction `ic` to the TensorNetwork `tn`, ensuring that
the referenced IndexLabels exist in the network and that they are not
already part of some other IndexContraction.
"""
function add_contraction!(tn::TensorNetwork, ic::IndexContraction)
    has_index(tn, ic.a) || throw(ArgumentError("index $(ic.a) not found in network"))
    has_index(tn, ic.b) || throw(ArgumentError("index $(ic.b) not found in network"))
    !has_contraction(tn, ic.a) || throw(ArgumentError("index $(ic.a) has already been contracted"))
    !has_contraction(tn, ic.b) || throw(ArgumentError("index $(ic.b) has already been contracted"))
    push!(tn._contractions, ic)
    tn._contraction_with_index[ic.a] = ic
    tn._contraction_with_index[ic.b] = ic
    nothing
end

"""
Attempts to add the IndexContraction `ic` to `tn`, returning whether it was successful.
"""
function try_contraction!(tn::TensorNetwork, ic::IndexContraction)
    success = false
    if can_contract(tn, ic.a) && can_contract(tn, ic.b)
        add_contraction!(tn, ic)
        success = true
    end
    success
end

"""
Removes the TensorLabel `tl` from the TensorNetwork `tn`, if there are
no contractions involving it. Errors if any contractions involving its
indices exist.
"""
function remove_tensor!(tn::TensorNetwork, tl::TensorLabel)
    # check that tensor has no contractions
    for idx in tl.indices
        !has_contraction(tn, idx) || throw(ArgumentError("tensor must not have contracted indices, found $(get_contraction(tn, idx))"))
    end
    # remove it
    filter!(!=(tl), tn._tensors)
    for idx in tl.indices
        delete!(tn._tensor_with_index, idx)
    end
    nothing
end

"""
Removes the IndexContraction `ic` from the TensorNetwork `tn`. If the
contraction is not present, throws an error.
"""
function remove_contraction!(tn::TensorNetwork, ic::IndexContraction)
    # check that contraction exists
    ic ∈ tn._contractions || throw(ArgumentError("contraction $ic not in tensor network"))
    # remove it
    filter!(!=(ic), tn._contractions)
    delete!(tn._contraction_with_index, ic.a)
    delete!(tn._contraction_with_index, ic.b)
    nothing
end

"""
Removes all contractions with any index belonging to the specified
TensorLabel `tl`.
"""
function remove_contractions!(tn::TensorNetwork, tl::TensorLabel)
    for idx in tl.indices
        if has_contraction(tn, idx)
            remove_contraction!(tn, get_contraction(tn, idx))
        end
    end
    nothing
end

"""
Replaces `old` with `new` in `tn`. Both must have the same group, and
every index in `old` must also appear in `new` (the new tensor may have
additional indices). By default all contractions on `old`'s indices are
preserved. If `preserve_contractions` is provided, only contractions
involving those indices are kept; others are removed.
"""
function replace_tensor!(tn::TensorNetwork, old::TensorLabel, new::TensorLabel;
                          preserve_contractions::Union{Nothing,Vector{IndexLabel}}=nothing)
    # validate replacement
    old.group == new.group || throw(ArgumentError("old and new must have the same group"))
    old_indices = Set(old.indices)
    new_indices = Set(new.indices)
    old_indices ⊆ new_indices || throw(ArgumentError("all indices in old must appear in new"))
    # validate contraction preservation
    if preserve_contractions === nothing
        preserve_contractions = old_indices
    else
        preserve_contractions = Set(preserve_contractions)
        preserve_contractions ⊆ old_indices || throw(ArgumentError("cannot preserve an index not in old tensor"))
    end
    # collect contractions to preserve
    preserved = IndexContraction[]
    for idx in old.indices
        if has_contraction(tn, idx) && idx ∈ preserve_contractions
            push!(preserved, get_contraction(tn, idx))
        end
    end
    unique!(preserved) # in case a tensor has self-contractions
    # remove all contractions on old tensor, then remove old tensor
    remove_contractions!(tn, old)
    remove_tensor!(tn, old)
    # add new tensor and re-add preserved contractions
    add_tensor!(tn, new)
    for ic in preserved
        add_contraction!(tn, ic)
    end
    nothing
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
    group_map = Dict{Int,Int}()
    extant_groups = Set(get_groups(tn1))
    new_group = 1
    for i in get_groups(tn2)
        while new_group ∈ extant_groups
            new_group += 1
        end
        group_map[i] = new_group
        push!(extant_groups, new_group)
    end
    # add tn2 tensors to tn1
    for (old, new) in group_map
        add_tensor!(tn1, regroup(get_tensor(tn2, old), new))
    end
    for oldc in get_contractions(tn2)
        a = regroup(oldc.a, group_map[oldc.a.group])
        b = regroup(oldc.b, group_map[oldc.b.group])
        add_contraction!(tn1, IndexContraction(a, b))
    end
    group_map
end

"""
Combines two tensor networks into a single one as in combine!, but then
also creates contractions between matching IndexLabels in the two
arguments. For example, suppose tn1 and tn2 both have an IndexLabel with
group 1 and port :p. In the new tn1, its (1, :p) index will stay the
same, while the (1, :p) index from tn2 will have been regrouped with the
rest of its TensorLabel by the combine! operation. However, unlike in
combine!, there will also be a contraction between that newly regrouped
index and the (1, :p) index from tn1.

Note that contractions are only formed between matching and uncontracted
IndexLabels: IndexLabels which are already contracted internally within
tn1 or tn2 will not have their contractions affected.

Returns the mapping from old to new group numbers for the second
argument's TensorLabels, like combine!.
"""
function matchcombine!(tn1::TensorNetwork, tn2::TensorNetwork)
    # find all matching indices that are not yet contracted
    matching = Vector{IndexLabel}()
    for idx in get_indices(tn1)
        if has_contraction(tn1, idx)
            continue
        end
        if has_index(tn2, idx)
            push!(matching, idx)
        end
    end
    # combine the tensor networks
    group_map = combine!(tn1, tn2)
    # create contractions
    for tn1idx in matching
        tn2idx = regroup(tn1idx, group_map[tn1idx.group])
        add_contraction!(tn1, IndexContraction(tn1idx, tn2idx))
    end
    group_map
end
