"""
PEPS representing a fibonacci string net state on a honeycomb lattice of qubits;
also stores a QubitLattice allowing index values to be interpreted as qubit values.

Constructor generates the ground state. `w` and `h` are measured in tensors,
not in plaquettes. Also generates positions for each segment tensor and each physical index
to be passed to visualization functions.

Supports mutation via `set_leaf!`, `add_crossings!`, `add_fusions!`,
and semantic `add_contraction!` helpers; these functions allow the
modification of the ground state into a string-net state or an anyonic fusion basis
state.
"""
struct FibTN
    w::Int
    h::Int
    segments::Dict{GridPosition, Segment}
    nextgroup::Ref{Int}
    ttn::TypedTensorNetwork
    ql::QubitLattice
    tpos::Dict{Int, Point2f}
    ipos::Dict{IL, Point2f}
    edge_crossings::Dict{GridEdge, Vector{Int}}
    edge_reflectors::Dict{GridEdge, Vector{Int}}
    fusions::Dict{GridPosition, Vector{Int}}
    function FibTN(w::Int, h::Int)
        w > 1 && h > 1 || throw(ArgumentError("width and height must both be greater than 1"))
        segments = Dict{GridPosition, Segment}()
        maxgroup = group_from_gridposition(1, 1, w)
        for i in 1:w, j in 1:h
            segments[i, j] = Segment(i, j, w, h)
            newgroup = group_from_gridposition(i, j, w)
            maxgroup = newgroup > maxgroup ? newgroup : maxgroup
        end
        ttn = create_ttn(w, h, segments)
        ql = create_ql(w, h, segments)
        tpos = create_tpos(w, h, segments)
        ipos = create_ipos(w, h, segments)
        new(w, h, segments, Ref(maxgroup + 1), ttn, ql, tpos, ipos, Dict(), Dict(), Dict())
    end
end

"""Convenience function so `FibTN[i, j] = FibTN.segments[i, j]."""
Base.getindex(ftn::FibTN, i::Int, j::Int) =
    ftn.segments[i, j]

"""
Creates the tensornetwork associated with `sg` by first iterating through
it and creating all segment tensors. It then autocontracts all extant
virtual indices by matching adjacent pairs of :U and :D or :R and :L ports.
"""
function create_ttn(w::Int, h::Int, segments::Dict{GridPosition, Segment})
    ttn = TypedTensorNetwork()
    # create tensors
    for i in 1:w, j in 1:h
        segment = segments[i, j]
        add_tensor!(ttn, segment.group, get_segmenttensortype(segment))
    end
    # autocontract all of the other segments together: go through each segment
    # and contract to the right and up
    for i in 1:w, j in 1:h
        g = segments[i, j].group
        # right side
        if i < w
            g_right = segments[i+1, j].group
            ril = IL(g, :R)
            lil = IL(g_right, :L)
            add_contraction!(ttn.tn, IC(ril, lil))
        end
        # upper side
        if j < h
            g_up = segments[i, j+1].group
            uil = IL(g, :U)
            dil = IL(g_up, :D)
       	    add_contraction!(ttn.tn, IC(uil, dil))
        end
    end
    ttn
end

"""
Assigns qubits to each segment in `segments`, in row-major order.

For each segment, interior qubits are assigned first, then qubits to the right
and above are assigned. Qubits to the left and below are assigned while processing
the segments to the left and below respectively.

Any qubits going off the edges of the lattice are assigned to 0, and will always be
trivial.
"""
function assign_qubits(w::Int, h::Int, segments::Dict{GridPosition, Segment})
    # reset any previous qubit assignment to make sure assignment is idempotent
    for i in 1:w, j in 1:h
        segment = segments[i, j]
        empty!(segment.qubits)
    end
    # assign qubits
    qcounter = 1
    for j in 1:h, i in 1:w
        # assign qubits in interior of segment
        segment = segments[i, j]
        stt = get_segmenttensortype(segment)
        if hasTVL(stt)
            segment.qubits[:UM] = qcounter
            segment.qubits[:T] = qcounter + 1
            segment.qubits[:DM] = qcounter + 2
            qcounter += 3
        elseif stt == BELBOW || stt == TELBOW
            segment.qubits[:M] = 0
        else
            segment.qubits[:M] = qcounter
            qcounter += 1
        end
        # assign qubits on :R connection
        if i < w
            rsegment = segments[i+1, j]
            segment.qubits[:R] = qcounter
            rsegment.qubits[:L] = qcounter
            qcounter += 1
        else
            segment.qubits[:R] = 0
        end
        # assign :L qubits of left col
        if i == 1
            segment.qubits[:L] = 0
        end
        # assign qubits on :U connection
        if j < h
            usegment = segments[i, j+1]
            segment.qubits[:U] = qcounter
            usegment.qubits[:D] = qcounter
            qcounter += 1
        else
            segment.qubits[:U] = 0
        end
        # assign :D qubits of bottom row
        if j == 1
            segment.qubits[:D] = 0
        end
    end
end

"""Creates QubitLattice for `segments`."""
function create_ql(w::Int, h::Int, segments::Dict{GridPosition, Segment})
    assign_qubits(w, h, segments)
    ql = QubitLattice()
    # iterate across segments, assigning qubits to pinds in an oriented way
    for i in 1:w, j in 1:h
        segment = segments[i, j]
        stt = get_segmenttensortype(segment)
        if hasTVL(stt)
            tqubits = [segment.qubits[:U], segment.qubits[:R], segment.qubits[:UM]]
            mqubits = [segment.qubits[:UM], segment.qubits[:T], segment.qubits[:DM]]
            bqubits = [segment.qubits[:D], segment.qubits[:L], segment.qubits[:DM]]
            add_index!(ql, IL(segment.group, :tp), tqubits)
            add_index!(ql, IL(segment.group, :mp), mqubits)
            add_index!(ql, IL(segment.group, :bp), bqubits)
        elseif stt == BELBOW
            bqubits = [segment.qubits[:D], segment.qubits[:L], segment.qubits[:M]]
            add_index!(ql, IL(segment.group, :bp), bqubits)
        elseif stt == TELBOW
            tqubits = [segment.qubits[:U], segment.qubits[:R], segment.qubits[:M]]
            add_index!(ql, IL(segment.group, :tp), tqubits)
        else
            tqubits = [segment.qubits[:U], segment.qubits[:R], segment.qubits[:M]]
            bqubits = [segment.qubits[:D], segment.qubits[:L], segment.qubits[:M]]
            add_index!(ql, IL(segment.group, :tp), tqubits)
            add_index!(ql, IL(segment.group, :bp), bqubits)
        end
    end
    ql
end

"""Returns `Dict` from tensor group to tensor position."""
function create_tpos(w::Int, h::Int, segments::Dict{GridPosition, Segment})
    tpos = Dict{Int, Point2f}()
    for i in 1:w, j in 1:h
        segment = segments[i, j]
        tpos[segment.group] = segment.tpos
    end
    tpos
end

"""Returns `Dict` from physical index label to vertex position."""
function create_ipos(w::Int, h::Int, segments::Dict{GridPosition, Segment})
    ipos = Dict{IL, Point2f}()
    for i in 1:w, j in 1:h
        merge!(ipos, segments[i, j].ipos)
    end
    ipos
end

### MUTATION METHODS ###

"""Allocates and returns the next available group number."""
function _allocate_group!(ftn::FibTN)
    g = ftn.nextgroup[]
    ftn.nextgroup[] += 1
    g
end

"""
Replaces the interior segment at grid position `(i, j)` with a
`VERTEX_EXCITATION_VERTEX`, preserving all shared contractions.
Only interior segments (not on the grid boundary) can be replaced.
"""
function set_leaf!(ftn::FibTN, i::Int, j::Int)
    1 < i < ftn.w && 1 < j < ftn.h || throw(ArgumentError("($i, $j) is not an interior position"))
    segment = ftn[i, j]
    replace_tensor!(ftn.ttn, segment.group, VERTEX_EXCITATION_VERTEX)
end

"""
Inserts `n` crossing tensors (with reflectors between them) along the edge
between grid positions `pos1` and `pos2`. Ports :U and :D on the crossings
remain free for later contraction via `contract!`.

For a horizontal edge `(i,j) → (i+1,j)`:
  seg1:R → cross1:L, cross1:R → refl1:V1, refl1:V2 → cross2:L, ..., crossn:R → refln:V1, refln:V2 → seg2:L

For a vertical edge `(i,j) → (i,j+1)`:
  seg1:U → cross1:L, cross1:R → refl1:V1, refl1:V2 → cross2:L, ..., crossn:R → refln:V1, refln:V2 → seg2:D
  (ie: :L faces down, :R faces up)
"""
function add_crossings!(ftn::FibTN, pos1::GridPosition, pos2::GridPosition, n::Int)
    # check that edge has no crossings yet
    edge = (pos1, pos2)
    !haskey(ftn.edge_crossings, edge) || throw(ArgumentError("edge $edge already has crossings"))
    # get edge orientation and check that positions are valid
    is_horizontal = pos2 == (pos1[1]+1, pos1[2])
    is_vertical   = pos2 == (pos1[1], pos1[2]+1)
    is_horizontal || is_vertical || throw(ArgumentError("pos2 must be adjacent to pos1 (right or up)"))
    # segment references
    seg1 = ftn[pos1]
    seg2 = ftn[pos2]
    # allocate groups for n crossings and n reflectors
    crossing_groups = [_allocate_group!(ftn) for _ in 1:n]
    reflector_groups = [_allocate_group!(ftn) for _ in 1:n]
    # add tensors to the TTN
    for g in crossing_groups add_tensor!(ftn.ttn, g, CROSSING) end
    for g in reflector_groups add_tensor!(ftn.ttn, g, REFLECTOR) end
    # remove existing contraction
    if is_horizontal
        # seg1:R ↔ seg2:L
        remove_contraction!(ftn.ttn.tn, IC(IL(seg1.group, :R), IL(seg2.group, :L)))
    else
        # seg1:U ↔ seg2:D
        remove_contraction!(ftn.ttn.tn, IC(IL(seg1.group, :U), IL(seg2.group, :D)))
    end
    # connect ends of chain
    if is_horizontal
        # seg1:R → cross1:L
        add_contraction!(ftn.ttn.tn, IC(IL(seg1.group, :R), IL(crossing_groups[1], :L)))
        # refln:V2 → seg2:L
        add_contraction!(ftn.ttn.tn, IC(IL(reflector_groups[n], :V2), IL(seg2.group, :L)))
    else
        # seg1:U → cross1:L
        add_contraction!(ftn.ttn.tn, IC(IL(seg1.group, :U), IL(crossing_groups[1], :L)))
        # refln:V2 → seg2:D
        add_contraction!(ftn.ttn.tn, IC(IL(reflector_groups[n], :V2), IL(seg2.group, :D)))
    end
    # connect up inside of chain
    # crossk:R → reflk:V1
    for k in 1:n
        add_contraction!(ftn.ttn.tn, IC(IL(crossing_groups[k], :R), IL(reflector_groups[k], :V1)))
    end
    # reflk:V2 → cross(k+1):L
    for k in 1:n-1
        add_contraction!(ftn.ttn.tn, IC(IL(reflector_groups[k], :V2), IL(crossing_groups[k+1], :L)))
    end
    # store for later reference
    ftn.edge_crossings[edge] = crossing_groups
    ftn.edge_reflectors[edge] = reflector_groups
    nothing
end

"""
Adds `n` fusion tensors of the given type (FUSION or DOUBLEDFUSION) to the
plaquette whose bottom-right corner is at grid position `pos`. Both types
share the same :V1, :V2, :V3 ports, so they are tracked in a single list.

All ports remain free for later contraction via `add_contraction!`.
"""
function add_fusions!(ftn::FibTN, pos::GridPosition, ::Type{T}, n::Int) where {T <: FibTensorType}
    T === FUSION || T === DOUBLEDFUSION || throw(ArgumentError("only FUSION and DOUBLEDFUSION are allowed, got $T"))
    groups = [_allocate_group!(ftn) for _ in 1:n]
    for g in groups
        add_tensor!(ftn.ttn, g, T)
    end
    existing = get(ftn.fusions, pos, Int[])
    ftn.fusions[pos] = vcat(existing, groups)
    nothing
end

### SEMANTIC CONTRACTION HELPERS ###

"""
Contracts a crossing port to a fusion port. The edge must belong to the
plaquette whose bottom-right corner is `plaq`: i.e. the edge must be one of
the four edges of the square with corners at `plaq`, `plaq .+ (0,1)`,
`plaq .+ (-1,0)`, and `plaq .+ (-1,1)`.
"""
function add_contraction!(ftn::FibTN,
        edge::GridEdge, crossing_idx::Int, crossing_port::Symbol,
        plaq::GridPosition, fusion_idx::Int, fusion_port::Symbol)
    # check that the edge belongs to the plaquette
    i, j = plaq
    valid_edges = Set([
        ((i-1, j), (i, j)),     # bottom
        ((i-1, j+1), (i, j+1)), # top
        ((i-1, j), (i-1, j+1)), # left
        ((i, j), (i, j+1)),     # right
    ])
    edge ∈ valid_edges || throw(ArgumentError("edge $edge does not belong to plaquette $plaq"))
    cg = ftn.edge_crossings[edge][crossing_idx]
    fg = ftn.fusions[plaq][fusion_idx]
    add_contraction!(ftn.ttn.tn, IC(IL(cg, crossing_port), IL(fg, fusion_port)))
end

"""
Contracts the :S port of the segment at `pos` to a fusion port. The
segment and fusion must be at the same grid position (i.e. the segment
is the bottom-right corner of the plaquette the fusion belongs to).
"""
function add_contraction!(ftn::FibTN, pos::GridPosition,
        fusion_idx::Int, fusion_port::Symbol)
    sg = ftn.segments[pos].group
    fg = ftn.fusions[pos][fusion_idx]
    add_contraction!(ftn.ttn.tn, IC(IL(sg, :S), IL(fg, fusion_port)))
end

"""Contracts two fusion ports in the same plaquette `plaq`."""
function add_contraction!(ftn::FibTN, plaq::GridPosition,
        fusion_idx1::Int, fusion_port1::Symbol,
        fusion_idx2::Int, fusion_port2::Symbol)
    fg1 = ftn.fusions[plaq][fusion_idx1]
    fg2 = ftn.fusions[plaq][fusion_idx2]
    add_contraction!(ftn.ttn.tn, IC(IL(fg1, fusion_port1), IL(fg2, fusion_port2)))
end
