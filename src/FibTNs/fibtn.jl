import ..TensorNetworks: add_contraction!

export masks2types
export FibTN, isperiodic
export add_crossings!, remove_crossing!, add_fusions!, remove_fusion!
export add_contraction!
export fix_excitation!

using ..TensorNetworks.TOBackend
export naive_contract

"""
Assigns groups in column-major ordering. `x` is the column,
`y` is the row, and `w` is the width of the grid (num cols).
"""
_group_from_gridposition(x::Int, y::Int, w::Int) =
    (y-1)*(w) + x

"""Converts masks to segmenttensortypes."""
masks2types(masks::BoundaryConditionsDict{Periodic, Unsigned}) where {Periodic} =
    BoundaryConditionsDict{Periodic, Type{<:SegmentTensorType}}(
        size(masks)...,
        Dict{GridPosition, Type{<:SegmentTensorType}}(
            pos => SegmentTensorType{m} for (pos, m) in masks
        )
    )

"""
PEPS representing a Fibonacci string net state on a honeycomb lattice of qubits,
together with a `QubitLattice` allowing physical index values to be interpreted
as qubit values.

Stores positions of tensors and qubits to facilitate visualization.

The constructor takes a `BoundaryConditionsDict` mapping `GridPosition → Type{<:SegmentTensorType}`.
It creates a segment for each entry, then builds the typed tensor network and qubit
lattice by autocontracting adjacent virtual indices. The grid may be jagged (sparse);
missing positions are skipped.

To construct the `BoundaryConditionsDict` input, prefer the helper functions:
- `segmentgrid(w, h; periodic, middle)` — fully-connected `w×h` rectangular grid
- `segmentgrid(masks; connect_across_boundary)` — infer connections from adjacency in
  a mask dict (mask integer values combine `STT_M`, `STT_T`, `STT_E`, `STT_V` flags;
  `1` means occupied with no middle)

To modify the grid before constructing a `FibTN`, use:
- `remove_segment!(segmentgrid, x, y)` — remove a position and strip neighbour ports
- `replace_segment!(segmentgrid, x, y, new_middle_mask)` — swap middle flags, preserve ports
- `remove_segment_connection!(segmentgrid, pos1, pos2)` — strip the shared port on an edge
- `add_segment_connection!(segmentgrid, pos1, pos2)` — add the shared port on an edge

To modify the ground state into a string-net or anyonic fusion basis state, use:
- `add_crossings!`
- `add_fusions!`
- `add_contraction!`

All constructed FibTNs start with no crossings or fusions.
"""
struct FibTN{Periodic}
    segments::BoundaryConditionsDict{Periodic, Segment}
    # tensor network stuff
    nextgroup::Ref{Int}
    ttn::TypedTensorNetwork
    tpos::Dict{Int, Point2f}
    # qubit lattice stuff
    ql::QubitLattice
    ipos::Dict{IL, Point2f}
    # anyonic fusion basis states stuff
    edge_crossings::Dict{GridEdge, Vector{Int}}
    edge_reflectors::Dict{GridEdge, Vector{Int}}
    fusions::Dict{GridPosition, Vector{Int}}
    # reverse lookups: group → (location, 1-based index within that location's list)
    edgecrossing_from_group::Dict{Int, Tuple{GridEdge, Int}}
    edgereflector_from_group::Dict{Int, Tuple{GridEdge, Int}}
    fusion_from_group::Dict{Int, Tuple{GridPosition, Int}}
    function FibTN(segmenttypegrid::BoundaryConditionsDict{Periodic, <:Type{<:SegmentTensorType}}) where {Periodic}
        # validate each segment's own mask
        for (_, T) in segmenttypegrid validatemask(T)  end
        # validate that every directional port has a matching partner
        for ((x, y), T) in segmenttypegrid
            if hasU(T)
                neighbour = get(segmenttypegrid, (x, y+1), nothing)
                neighbour !== nothing && hasD(neighbour) || throw(ArgumentError("unpaired U at ($x, $y)"))
            end
            if hasR(T)
                neighbour = get(segmenttypegrid, (x+1, y), nothing)
                neighbour !== nothing && hasL(neighbour) || throw(ArgumentError("unpaired R at ($x, $y)"))
            end
            if hasD(T)
                neighbour = get(segmenttypegrid, (x, y-1), nothing)
                neighbour !== nothing && hasU(neighbour) || throw(ArgumentError("unpaired D at ($x, $y)"))
            end
            if hasL(T)
                neighbour = get(segmenttypegrid, (x-1, y), nothing)
                neighbour !== nothing && hasR(neighbour) || throw(ArgumentError("unpaired L at ($x, $y)"))
            end
        end
        # make segments
        w, h = size(segmenttypegrid)
        segments = BoundaryConditionsDict{Periodic, Segment}(w, h)
        maxgroup = 0
        for ((x, y), T) in segmenttypegrid
            segments[x, y] = Segment(T, x, y, _group_from_gridposition(x, y, w))
            maxgroup = max(maxgroup, segments[x, y].group)
        end
        # make everything else
        ttn  = _create_ttn(segments)
        tpos = _create_tpos(segments)
        ql   = _create_ql(segments)
        ipos = _create_ipos(segments)
        new{Periodic}(segments, Ref(maxgroup + 1), ttn, tpos, ql, ipos, Dict(), Dict(), Dict(), Dict(), Dict(), Dict())
    end
    # helper constructor for convenience
    FibTN(segmentmaskgrid::BoundaryConditionsDict{Periodic, Unsigned}) where {Periodic} =
        FibTN(masks2types(segmentmaskgrid))
end

"""Convenience function so `ftn[x, y]` returns `ftn.segments[x, y]`."""
Base.getindex(ftn::FibTN, x::Int, y::Int) = ftn.segments[x, y]

"""Convenience function so `ftn[x, y]` returns `ftn.segments[x, y]`."""
Base.getindex(ftn::FibTN, pos::GridPosition) = ftn[pos...]

"""Returns the grid size of this PEPS."""
Base.size(ftn::FibTN) = size(ftn.segments)

"""Returns `true` if this `FibTN` has periodic boundary conditions."""
isperiodic(::FibTN{Periodic}) where {Periodic} = Periodic

"""
Creates the typed tensor network for the given grid of segments. Each segment
gets a tensor, then all adjacent virtual index pairs (:R/:L and :U/:D) are
contracted together. Skips missing neighbours for jagged grids; wraps around
for periodic grids (handled transparently by `BoundaryConditionsDict`).
"""
function _create_ttn(segments::BoundaryConditionsDict)
    ttn = TypedTensorNetwork()
    # add one tensor per segment
    for (_, segment) in segments
        add_tensor!(ttn, segment.group, get_segmenttensortype(segment))
    end
    # autocontract adjacent pairs — only contract :R with its right neighbour
    # and :U with its upper neighbour to avoid double-contracting each edge
    for ((x, y), segment) in segments
        g = segment.group
        T = get_segmenttensortype(segment)
        # contract :R and :L
        if hasR(T) && haskey(segments, x+1, y)
            right = segments[x+1, y]
            add_contraction!(ttn.tn, IC(IL(g, :R), IL(right.group, :L)))
        end
        # contract :U and :D
        if hasU(T) && haskey(segments, x, y+1)
            above = segments[x, y+1]
            add_contraction!(ttn.tn, IC(IL(g, :U), IL(above.group, :D)))
        end
    end
    ttn
end

"""
Resets all directional qubits (:U, :R, :D, :L) to 0 on every segment and
clears interior qubits. Directional qubits default to 0 (trivial/vacuum)
so that boundary edges and sparse-grid interior boundaries are handled
correctly without special-casing them later.
"""
function _reset_qubits(segments::BoundaryConditionsDict)
    for (_, segment) in segments
        empty!(segment.qubits)
        segment.qubits[:U] = 0
        segment.qubits[:R] = 0
        segment.qubits[:D] = 0
        segment.qubits[:L] = 0
    end
end

"""
Assigns qubit numbers to each segment in `segments`. Shared virtual indices
between adjacent segments get the same qubit number. Unshared virtual indices
are assigned qubit 0 and assumed always trivial.

Interior qubits (:UM, :T, :DM) are assigned for segments with a middle physical port.
The single interior qubit :M is assigned for segments with a middle, and is assigned
to 0 for those without, and must always be trivial.

Iterates row-major (y outer, x inner) for consistent qubit numbering. Each :R/:U
assigns both its own qubit and the partner :L/:D in the neighbour immediately. For
periodic grids this means the wrap-around :L/:D entries are written retroactively,
which is correct since all assignments complete before `create_ql` runs.
"""
function _assign_qubits(segments::BoundaryConditionsDict)
    w, h = segments.w, segments.h
    # delete any qubit mappings internal to the segment, and reset directional qubits
    # to 0; shared edges will be overwritten by neighbors if necessary
    _reset_qubits(segments)
    qcounter = 1
    for y in 1:h, x in 1:w  # row-major for consistent numbering
        haskey(segments, x, y) || continue
        segment = segments[x, y]
        T = get_segmenttensortype(segment)
        # assign interior qubit(s)
        if hasMP(T)
            # three interior qubits: upper-middle, tail, lower-middle
            segment.qubits[:UM] = qcounter
            segment.qubits[:T]  = qcounter + 1
            segment.qubits[:DM] = qcounter + 2
            qcounter += 3
        elseif hasM(T)
            # one interior qubit: middle
            segment.qubits[:M] = qcounter
            qcounter += 1
        else
            # trivial interior qubit
            segment.qubits[:M] = 0
        end
        # assign :R / :L pair — partner written into right neighbour immediately
        if hasR(T)
            rsegment = segments[x+1, y]
            segment.qubits[:R]  = qcounter
            rsegment.qubits[:L] = qcounter
            qcounter += 1
        else
            segment.qubits[:R] = 0
        end
        # assign :U / :D pair — partner written into upper neighbour immediately
        if hasU(T)
            usegment = segments[x, y+1]
            segment.qubits[:U]  = qcounter
            usegment.qubits[:D] = qcounter
            qcounter += 1
        else
            segment.qubits[:U] = 0
        end
    end
end

"""Creates the `QubitLattice` for the given grid of segments."""
function _create_ql(segments::BoundaryConditionsDict)
    _assign_qubits(segments)
    ql = QubitLattice()
    for (_, segment) in segments
        T = get_segmenttensortype(segment)
        g = segment.group
        if hasTP(T)
            # top physical index encodes :U (above), :R (right), and interior qubit
            tqubits = (
                segment.qubits[:U],
                segment.qubits[:R],
                hasMP(T) ? segment.qubits[:UM] : segment.qubits[:M]
            )
            add_index!(ql, IL(g, :TP), tqubits)
        end
        if hasMP(T)
            # middle physical index encodes the three interior qubits
            mqubits = (
                segment.qubits[:UM],
                segment.qubits[:T],
                segment.qubits[:DM]
            )
            add_index!(ql, IL(g, :MP), mqubits)
        end
        if hasBP(T)
            # bottom physical index encodes :D (below), :L (left), and interior qubit
            bqubits = (
                segment.qubits[:D],
                segment.qubits[:L],
                hasMP(T) ? segment.qubits[:DM] : segment.qubits[:M]
            )
            add_index!(ql, IL(g, :BP), bqubits)
        end
    end
    ql
end

"""Returns a `Dict` from tensor group to tensor position."""
function _create_tpos(segments::BoundaryConditionsDict)
    tpos = Dict{Int, Point2f}()
    for (_, segment) in segments
        tpos[segment.group] = segment.tpos
    end
    tpos
end

"""Returns a `Dict` from physical index label to index position."""
function _create_ipos(segments::BoundaryConditionsDict)
    ipos = Dict{IL, Point2f}()
    for (_, segment) in segments
        merge!(ipos, segment.ipos)
    end
    ipos
end

### ADD/REMOVE CROSSINGS/FUSIONS ###

"""
Recomputes `tpos` entries for all crossings and reflectors on `edge`, dividing
the straight line between the two endpoint segment positions into `2n+1` equal
slots (where `n` is the number of crossing-reflector pairs). Crossing `k` sits
at slot `2k-1` and reflector `k` at slot `2k`, so they alternate evenly along
the edge.
"""
function _update_crossing_tpos!(ftn::FibTN, edge::GridEdge)
    pos1, pos2 = edge
    p1 = ftn.tpos[ftn[pos1].group]
    p2 = ftn.tpos[ftn[pos2].group]
    crossings  = ftn.edge_crossings[edge]
    reflectors = ftn.edge_reflectors[edge]
    n = length(crossings)
    for k in 1:n
        ftn.tpos[crossings[k]]  = p1 + (2k-1) / (2n+1) * (p2 - p1)
        ftn.tpos[reflectors[k]] = p1 + (2k)   / (2n+1) * (p2 - p1)
    end
end

"""Helper to allocate and return the next available group number."""
function _allocate_group!(ftn::FibTN)
    g = ftn.nextgroup[]
    ftn.nextgroup[] += 1
    g
end

"""
Inserts `n` crossing tensors (with reflectors between them) along the edge
between grid positions `pos1` and `pos2`. `pos2` must be directly right of or
directly above `pos1`, accounting for periodic wrapping. Ports `:U` and `:D`
on the crossings remain free for later contraction via `add_contraction!`.

May be called multiple times on the same edge to append more crossings; each
call splices `n` new crossings between the existing tail reflector and `seg2`.

Chain layout (horizontal edge as example):
  seg1:R → cross1:L, cross1:R → refl1:V1, refl1:V2 → cross2:L, ..., crossn:R → refln:V1, refln:V2 → seg2:L

Note that an added reflector always has the same index within the reflectors list
on this edge as its associated crossing's index within the crossings list on this
edge, as they are always added/removed in pairs.

Returns nothing.
"""
function add_crossings!(ftn::FibTN, pos1::GridPosition, pos2::GridPosition, n::Int)
    edge = (pos1, pos2)
    segs = ftn.segments
    seg1 = ftn[pos1]
    seg2 = ftn[pos2]
    # determine orientation; wrappos is identity for nonperiodic grids, so this
    # handles both cases — including wrap-around edges (e.g. (w,y)→(1,y))
    is_horizontal = wrappos(segs, pos2...) == wrappos(segs, pos1[1]+1, pos1[2])
    is_vertical   = wrappos(segs, pos2...) == wrappos(segs, pos1[1], pos1[2]+1)
    is_horizontal || is_vertical || throw(ArgumentError("pos2 must be adjacent to pos1 (right or up)"))
    # allocate groups for n new crossings and n new reflectors
    crossing_groups  = [_allocate_group!(ftn) for _ in 1:n]
    reflector_groups = [_allocate_group!(ftn) for _ in 1:n]
    for g in crossing_groups  add_tensor!(ftn.ttn, g, CROSSING)  end
    for g in reflector_groups add_tensor!(ftn.ttn, g, REFLECTOR) end
    # find the contraction that currently connects into seg2 and remove it,
    # then attach the new chain between that anchor and seg2
    if haskey(ftn.edge_crossings, edge)
        # splice after the existing tail reflector
        prev_tail_refl = ftn.edge_reflectors[edge][end]
        tail_port      = is_horizontal ? :L : :D
        remove_contraction!(ftn.ttn.tn, IC(IL(prev_tail_refl, :V2), IL(seg2.group, tail_port)))
        add_contraction!(ftn.ttn.tn, IC(IL(prev_tail_refl, :V2), IL(crossing_groups[1], :L)))
    else
        # first insertion: remove the direct seg1–seg2 contraction
        if is_horizontal
            remove_contraction!(ftn.ttn.tn, IC(IL(seg1.group, :R), IL(seg2.group, :L)))
            add_contraction!(ftn.ttn.tn, IC(IL(seg1.group, :R), IL(crossing_groups[1], :L)))
        else
            remove_contraction!(ftn.ttn.tn, IC(IL(seg1.group, :U), IL(seg2.group, :D)))
            add_contraction!(ftn.ttn.tn, IC(IL(seg1.group, :U), IL(crossing_groups[1], :L)))
        end
    end
    # connect interior: crossk:R → reflk:V1, reflk:V2 → cross(k+1):L
    for k in 1:n
        add_contraction!(ftn.ttn.tn, IC(IL(crossing_groups[k], :R), IL(reflector_groups[k], :V1)))
    end
    for k in 1:n-1
        add_contraction!(ftn.ttn.tn, IC(IL(reflector_groups[k], :V2), IL(crossing_groups[k+1], :L)))
    end
    # reconnect new tail to seg2
    if is_horizontal
        add_contraction!(ftn.ttn.tn, IC(IL(reflector_groups[n], :V2), IL(seg2.group, :L)))
    else
        add_contraction!(ftn.ttn.tn, IC(IL(reflector_groups[n], :V2), IL(seg2.group, :D)))
    end
    # append to stored lists (or initialise them) and update reverse lookups
    crossing_list  = get!(ftn.edge_crossings,  edge, Int[])
    reflector_list = get!(ftn.edge_reflectors, edge, Int[])
    for (k, g) in enumerate(crossing_groups)
        ftn.edgecrossing_from_group[g] = (edge, length(crossing_list) + k)
    end
    for (k, g) in enumerate(reflector_groups)
        ftn.edgereflector_from_group[g] = (edge, length(reflector_list) + k)
    end
    append!(crossing_list,  crossing_groups)
    append!(reflector_list, reflector_groups)
    _update_crossing_tpos!(ftn, edge)
    nothing
end

"""
Removes the crossing and its paired reflector at position `k` (1-based) in the
chain on the edge between `pos1` and `pos2`, splicing the neighbours back together.
Errors if no crossings exist on that edge or `k` is out of range.

WARNING: calling this function on crossings with 'out of edge' contractions will
result in undefined behavior.
"""
function remove_crossing!(ftn::FibTN, pos1::GridPosition, pos2::GridPosition, k::Int)
    # edge validation
    edge = (pos1, pos2)
    haskey(ftn.edge_crossings, edge) || throw(ArgumentError("no crossings on edge $edge"))
    # index validation
    crossing_list  = ftn.edge_crossings[edge]
    reflector_list = ftn.edge_reflectors[edge]
    n = length(crossing_list)
    1 <= k <= n || throw(ArgumentError("crossing index $k out of range 1:$n"))
    # we don't need to check is_vertical or if it is adjacent because edge_crossings would
    # have no key at that 'edge' if the edge is invalid, so that case gets caught above
    is_horizontal = wrappos(ftn.segments, pos2...) == wrappos(ftn.segments, pos1[1]+1, pos1[2])
    # identify left and right anchors of the pair being removed
    left_il  = k == 1 ? IL(ftn[pos1].group, is_horizontal ? :R : :U) : IL(reflector_list[k-1], :V2)
    right_il = k == n ? IL(ftn[pos2].group, is_horizontal ? :L : :D) : IL(crossing_list[k+1],  :L)
    # remove the three contractions spanning the pair
    cross_g = crossing_list[k]
    refl_g  = reflector_list[k]
    remove_contraction!(ftn.ttn.tn, IC(left_il,              IL(cross_g, :L)))
    remove_contraction!(ftn.ttn.tn, IC(IL(cross_g, :R),      IL(refl_g,  :V1)))
    remove_contraction!(ftn.ttn.tn, IC(IL(refl_g,  :V2),     right_il))
    # splice: connect left anchor directly to right anchor
    add_contraction!(ftn.ttn.tn, IC(left_il, right_il))
    # remove tensors
    remove_tensor!(ftn.ttn, cross_g)
    remove_tensor!(ftn.ttn, refl_g)
    # update lists and reverse lookups
    deleteat!(crossing_list,  k)
    deleteat!(reflector_list, k)
    delete!(ftn.edgecrossing_from_group,  cross_g)
    delete!(ftn.edgereflector_from_group, refl_g)
    # decrement indices of all entries that shifted left
    for i in k:length(crossing_list)
        ftn.edgecrossing_from_group[crossing_list[i]]   = (edge, i)
        ftn.edgereflector_from_group[reflector_list[i]] = (edge, i)
    end
    # clean up edge entries if the chain is now empty, otherwise recompute positions
    if isempty(crossing_list)
        delete!(ftn.tpos, cross_g)
        delete!(ftn.tpos, refl_g)
        delete!(ftn.edge_crossings,  edge)
        delete!(ftn.edge_reflectors, edge)
    else
        delete!(ftn.tpos, cross_g)
        delete!(ftn.tpos, refl_g)
        _update_crossing_tpos!(ftn, edge)
    end
    nothing
end

"""
Adds fusion tensors of the given type (FUSION or DOUBLEDFUSION) to the
plaquette whose bottom-right corner is at grid position `pos`. Both types
share the same :V1, :V2, :V3 ports, so they are tracked in a single list.

`positions` is a `Vector{Point2f}` giving the display position for each new
fusion tensor, in order. The number of tensors added equals `length(positions)`.

If `relative=true`, each position is interpreted as an offset from the segment
tensor at `pos` (i.e. `ftn.tpos[ftn[pos].group]`), so e.g. `(-1.0, 1.0)` places
a fusion one unit left and one unit above the segment — inside its plaquette.

All ports remain free for later contraction via `add_contraction!`.
"""
function add_fusions!(ftn::FibTN, pos::GridPosition, ::Type{T}, positions::Vector{Point2f};
                      relative::Bool=false) where {T <: FibTensorType}
    T === FUSION || T === DOUBLEDFUSION || throw(ArgumentError("only FUSION and DOUBLEDFUSION are allowed, got $T"))
    n = length(positions)
    origin = relative ? ftn.tpos[ftn[pos].group] : Point2f(0, 0)
    groups = [_allocate_group!(ftn) for _ in 1:n]
    for g in groups add_tensor!(ftn.ttn, g, T) end
    fusion_list = get!(ftn.fusions, pos, Int[])
    for (k, g) in enumerate(groups)
        ftn.fusion_from_group[g] = (pos, length(fusion_list) + k)
        ftn.tpos[g] = origin + positions[k]
    end
    append!(fusion_list, groups)
    nothing
end

"""
Removes the fusion tensor at position `k` (1-based) in the fusion list at
grid position `pos`. Errors if no fusions exist at `pos` or `k` is out of range.

WARNING: calling this function after creating contractions will fail.
"""
function remove_fusion!(ftn::FibTN, pos::GridPosition, k::Int)
    haskey(ftn.fusions, pos) || throw(ArgumentError("no fusions at position $pos"))
    fusion_list = ftn.fusions[pos]
    n = length(fusion_list)
    1 <= k <= n || throw(ArgumentError("fusion index $k out of range 1:$n"))
    g = fusion_list[k]
    remove_tensor!(ftn.ttn, g)
    deleteat!(fusion_list, k)
    delete!(ftn.fusion_from_group, g)
    delete!(ftn.tpos, g)
    # decrement indices of all entries that shifted left
    for i in k:length(fusion_list)
        ftn.fusion_from_group[fusion_list[i]] = (pos, i)
    end
    if isempty(fusion_list)
        delete!(ftn.fusions, pos)
    end
    nothing
end

# ### SEMANTIC CONTRACTION HELPERS ###

"""
Insert a single REFLECTOR between `from_group:from_port` and `to_group:to_port`,
positioning it at the midpoint of the two tensor positions.

  from_group:from_port → R:V1, R:V2 → to_group:to_port

Returns the allocated group ID of the new REFLECTOR.
"""
function _add_reflected_contraction!(ftn::FibTN,
        from_group::Int, from_port::Symbol,
        to_group::Int, to_port::Symbol)
    r = _allocate_group!(ftn)
    add_tensor!(ftn.ttn, r, REFLECTOR)
    add_contraction!(ftn.ttn.tn, IC(IL(from_group, from_port), IL(r, :V1)))
    add_contraction!(ftn.ttn.tn, IC(IL(r, :V2), IL(to_group, to_port)))
    ftn.tpos[r] = (ftn.tpos[from_group] + ftn.tpos[to_group]) / 2
    return r
end

"""
Insert a VACUUMLOOP flanked by two REFLECTORs between `from_group:from_port` and
`to_group:to_port`. The chain is:

  from_group:from_port → R1:V1, R1:V2 → B:V1, B:V2 → R2:V1, R2:V2 → to_group:to_port

`B:V1` faces `from_group`. The middle tensor is placed at the midpoint; the two
flanking reflectors are each placed at the midpoint between their neighbours.
Returns nothing.
"""
function _add_vl_contraction!(ftn::FibTN,
        from_group::Int, from_port::Symbol,
        to_group::Int, to_port::Symbol)
    b = _allocate_group!(ftn)
    add_tensor!(ftn.ttn, b, VACUUMLOOP)
    ftn.tpos[b] = (ftn.tpos[from_group] + ftn.tpos[to_group]) / 2
    _add_reflected_contraction!(ftn, from_group, from_port, b, :V1)
    _add_reflected_contraction!(ftn, b, :V2, to_group, to_port)
    nothing
end

"""
Insert a `STRINGCONTROL{S}` flanked by two REFLECTORs between `from_group:from_port`
and `to_group:to_port`. The chain is:

  from_group:from_port → R1:V1, R1:V2 → SC:V1, SC:V2 → R2:V1, R2:V2 → to_group:to_port

`SC:V1` faces `from_group`. The middle tensor is placed at the midpoint; the two
flanking reflectors are each placed at the midpoint between their neighbours.
Returns nothing.
"""
function _add_sc_contraction!(ftn::FibTN,
        from_group::Int, from_port::Symbol,
        to_group::Int, to_port::Symbol,
        s::Int)
    sc = _allocate_group!(ftn)
    add_tensor!(ftn.ttn, sc, STRINGCONTROL{s})
    ftn.tpos[sc] = (ftn.tpos[from_group] + ftn.tpos[to_group]) / 2
    _add_reflected_contraction!(ftn, from_group, from_port, sc, :V1)
    _add_reflected_contraction!(ftn, sc, :V2, to_group, to_port)
    nothing
end

"""
Insert a VACUUMLOOP followed by a `STRINGCONTROL{S}`, each flanked by REFLECTORs,
between `from_group:from_port` and `to_group:to_port`. The chain is:

  from_group:from_port → R1 → B:V1, B:V2 → R2 → SC:V1, SC:V2 → R3 → to_group:to_port

`B:V1` faces `from_group`. Middle tensors are placed at 1/3 and 2/3 of the straight
line between the two endpoint positions; reflectors land at the midpoints between
their neighbours. Returns nothing.
"""
function _add_vl_sc_contraction!(ftn::FibTN,
        from_group::Int, from_port::Symbol,
        to_group::Int, to_port::Symbol,
        s::Int)
    p1 = ftn.tpos[from_group]
    p2 = ftn.tpos[to_group]
    b  = _allocate_group!(ftn)
    sc = _allocate_group!(ftn)
    add_tensor!(ftn.ttn, b,  VACUUMLOOP)
    add_tensor!(ftn.ttn, sc, STRINGCONTROL{s})
    ftn.tpos[b]  = p1 + 1/3 * (p2 - p1)
    ftn.tpos[sc] = p1 + 2/3 * (p2 - p1)
    _add_reflected_contraction!(ftn, from_group, from_port, b,  :V1)
    _add_reflected_contraction!(ftn, b,  :V2, sc, :V1)
    _add_reflected_contraction!(ftn, sc, :V2, to_group, to_port)
    nothing
end

"""
Contracts a crossing port to a fusion port, inserting a reflector between them.
With `with_vl=true`, a BOUNDARY tensor is also inserted with its `:V1` facing
the crossing.

No validation of port validity or plaquette membership is performed.
"""
function add_contraction!(ftn::FibTN,
        edge::GridEdge, crossing_idx::Int, crossing_port::Symbol,
        plaq::GridPosition, fusion_idx::Int, fusion_port::Symbol;
        with_vl::Bool=false)
    cg = ftn.edge_crossings[edge][crossing_idx]
    fg = ftn.fusions[plaq][fusion_idx]
    if with_vl
        _add_vl_contraction!(ftn, cg, crossing_port, fg, fusion_port)
    else
        _add_reflected_contraction!(ftn, cg, crossing_port, fg, fusion_port)
    end
end

"""
Contracts a fusion port to a crossing port, inserting a reflector between them.
With `with_vl=true`, a BOUNDARY tensor is also inserted with its `:V1` facing
the fusion.

No validation of port validity or plaquette membership is performed.
"""
function add_contraction!(ftn::FibTN,
        plaq::GridPosition, fusion_idx::Int, fusion_port::Symbol,
        edge::GridEdge, crossing_idx::Int, crossing_port::Symbol;
        with_vl::Bool=false)
    fg = ftn.fusions[plaq][fusion_idx]
    cg = ftn.edge_crossings[edge][crossing_idx]
    if with_vl
        _add_vl_contraction!(ftn, fg, fusion_port, cg, crossing_port)
    else
        _add_reflected_contraction!(ftn, fg, fusion_port, cg, crossing_port)
    end
end

"""
Contracts the `:S` port of the segment at `pos` to a fusion port, inserting a
reflector between them. With `with_vl=true`, a BOUNDARY tensor is also
inserted with its `:V1` facing the segment.

Only valid for segments with an excitation tensor (`:S` port present).
"""
function add_contraction!(ftn::FibTN, pos::GridPosition,
        fusion_idx::Int, fusion_port::Symbol;
        with_vl::Bool=false)
    sg = ftn.segments[pos].group
    fg = ftn.fusions[pos][fusion_idx]
    if with_vl
        _add_vl_contraction!(ftn, sg, :S, fg, fusion_port)
    else
        _add_reflected_contraction!(ftn, sg, :S, fg, fusion_port)
    end
end

"""
Contracts a fusion port to the `:S` port of the segment at `pos`, inserting a
reflector between them. With `with_vl=true`, a BOUNDARY tensor is also
inserted with its `:V1` facing the fusion.

Only valid for segments with an excitation tensor (`:S` port present).
"""
function add_contraction!(ftn::FibTN,
        fusion_idx::Int, fusion_port::Symbol,
        pos::GridPosition;
        with_vl::Bool=false)
    fg = ftn.fusions[pos][fusion_idx]
    sg = ftn.segments[pos].group
    if with_vl
        _add_vl_contraction!(ftn, fg, fusion_port, sg, :S)
    else
        _add_reflected_contraction!(ftn, fg, fusion_port, sg, :S)
    end
end

"""
Contracts the `:S` port of the segment at `pos` to a crossing port, inserting
a reflector between them. With `with_vl=true`, a BOUNDARY tensor is also
inserted with its `:V1` facing the segment.

Only valid for segments with an excitation tensor (`:S` port present).
"""
function add_contraction!(ftn::FibTN, pos::GridPosition,
        edge::GridEdge, crossing_idx::Int, crossing_port::Symbol;
        with_vl::Bool=false)
    sg = ftn.segments[pos].group
    cg = ftn.edge_crossings[edge][crossing_idx]
    if with_vl
        _add_vl_contraction!(ftn, sg, :S, cg, crossing_port)
    else
        _add_reflected_contraction!(ftn, sg, :S, cg, crossing_port)
    end
end

"""
Contracts a crossing port to the `:S` port of the segment at `pos`, inserting
a reflector between them. With `with_vl=true`, a BOUNDARY tensor is also
inserted with its `:V1` facing the crossing.

Only valid for segments with an excitation tensor (`:S` port present).
"""
function add_contraction!(ftn::FibTN,
        edge::GridEdge, crossing_idx::Int, crossing_port::Symbol,
        pos::GridPosition;
        with_vl::Bool=false)
    cg = ftn.edge_crossings[edge][crossing_idx]
    sg = ftn.segments[pos].group
    if with_vl
        _add_vl_contraction!(ftn, cg, crossing_port, sg, :S)
    else
        _add_reflected_contraction!(ftn, cg, crossing_port, sg, :S)
    end
end

"""
Contracts two fusion ports in the same plaquette `plaq`, inserting a reflector
between them. With `with_vl=true`, a BOUNDARY tensor is also inserted
(its `:V1` faces `fusion_idx1`).
"""
function add_contraction!(ftn::FibTN, plaq::GridPosition,
        fusion_idx1::Int, fusion_port1::Symbol,
        fusion_idx2::Int, fusion_port2::Symbol;
        with_vl::Bool=false)
    fg1 = ftn.fusions[plaq][fusion_idx1]
    fg2 = ftn.fusions[plaq][fusion_idx2]
    if with_vl
        _add_vl_contraction!(ftn, fg1, fusion_port1, fg2, fusion_port2)
    else
        _add_reflected_contraction!(ftn, fg1, fusion_port1, fg2, fusion_port2)
    end
end

"""
Contracts two crossing ports, inserting a reflector chain between them.

- Default: single REFLECTOR.
- `with_vl=true`: REFLECTOR → VACUUMLOOP → REFLECTOR (`:V1` faces `crossing_idx1`).
- `with_sc=true`: REFLECTOR → STRINGCONTROL{sc} → REFLECTOR (`:V1` faces `crossing_idx1`).
- `with_vl=true, with_sc=true`: REFLECTOR → VACUUMLOOP → REFLECTOR → STRINGCONTROL{sc} → REFLECTOR.
"""
function add_contraction!(ftn::FibTN,
    edge1::GridEdge, crossing_idx1::Int, crossing_port1::Symbol,
    edge2::GridEdge, crossing_idx2::Int, crossing_port2::Symbol;
        with_vl::Bool=false, with_sc::Bool=false, sc::Int=0)
    cg1 = ftn.edge_crossings[edge1][crossing_idx1]
    cg2 = ftn.edge_crossings[edge2][crossing_idx2]
    if with_vl && with_sc
        _add_vl_sc_contraction!(ftn, cg1, crossing_port1, cg2, crossing_port2, sc)
    elseif with_vl
        _add_vl_contraction!(ftn, cg1, crossing_port1, cg2, crossing_port2)
    elseif with_sc
        _add_sc_contraction!(ftn, cg1, crossing_port1, cg2, crossing_port2, sc)
    else
        _add_reflected_contraction!(ftn, cg1, crossing_port1, cg2, crossing_port2)
    end
end

"""
Fixes the excitation control indices of the segment at `pos` to the given values
`a`, `b`, `l` (0-based qubit values). The segment must have the excitation bit set
in its mask (i.e. `hasE(get_segmenttensortype(segment))`). Adds an
`EXCITATIONCONTROL{a,b,l}` tensor contracted against the segment's `:a`, `:b`, `:l`
ports. The control tensor is not assigned a position and will be skipped during
visualization.
"""
function fix_excitation!(ftn::FibTN, pos::GridPosition, a::Int, b::Int, l::Int; root::Bool=false)
    # validate segment
    segment = ftn[pos]
    T = get_segmenttensortype(segment)
    hasE(T) || throw(ArgumentError("segment at $pos is not an excitation"))
    # create control tensor
    ctrl_g = _allocate_group!(ftn)
    add_tensor!(ftn.ttn, ctrl_g, EXCITATIONCONTROL{a,b,l})
    # add root if needed
    seg_g = segment.group
    base_g = seg_g # the tensor we contract control with
    if root
        root_g = _allocate_group!(ftn)
        add_tensor!(ftn.ttn, root_g, FUSIONTREEROOT)
        add_contraction!(ftn.ttn.tn, IC(IL(seg_g, :a), IL(root_g, :a1)))
        add_contraction!(ftn.ttn.tn, IC(IL(seg_g, :b), IL(root_g, :b1)))
        add_contraction!(ftn.ttn.tn, IC(IL(seg_g, :l), IL(root_g, :l1)))
        base_g = root_g
    end
    add_contraction!(ftn.ttn.tn, IC(IL(base_g, :a), IL(ctrl_g, :a)))
    add_contraction!(ftn.ttn.tn, IC(IL(base_g, :b), IL(ctrl_g, :b)))
    add_contraction!(ftn.ttn.tn, IC(IL(base_g, :l), IL(ctrl_g, :l)))
    nothing
end

### CONTRACTION ###

function naive_contract(ftn::FibTN)
    ttn = ftn.ttn
    es = ExecutionState(ttn)
    # classify groups by tensor type
    reflector_groups = Set(g for (g, T) in ttn.tensortype_from_group if T === REFLECTOR)
    crossing_groups  = Set(g for (g, T) in ttn.tensortype_from_group if T === CROSSING)
    # sort contractions into phases
    contractions = get_contractions(ttn.tn)
    phase1, phase2, phase3 = IC[], IC[], IC[]
    for ic in contractions
        ga = get_tensor(ttn.tn, ic.a).group
        gb = get_tensor(ttn.tn, ic.b).group
        if ga ∈ reflector_groups || gb ∈ reflector_groups
            push!(phase1, ic)
        elseif ga ∈ crossing_groups || gb ∈ crossing_groups
            push!(phase2, ic)
        else
            push!(phase3, ic)
        end
    end
    # do contraction
    for ic in Iterators.flatten((phase1, phase2, phase3))
        execute_step!(es, ContractionStep(ic))
    end
    # fetch result and return
    id = only(get_ids(es))
    et = es.tensor_from_id[id]
    et.indices, et.data
end
