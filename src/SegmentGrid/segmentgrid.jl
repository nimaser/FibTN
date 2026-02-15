using FibTN
using FibTN.TensorNetworks
using FibTN.TensorNetworks.TOBackend

using FibTN.QubitLattices

using FibTN.FibTensorTypes
using FibTN.SegmentTensorTypes

using GeometryBasics
using SparseArrayKit
using Serialization

const IL = IndexLabel
const IC = IndexContraction

### GRIDSEGMENTWRAPPER ###

"""
Assigns groups in row-primary ordering.

`i` is the row, `j` is the col, and `w` is the width of the grid,
in terms of tensors, not plaquettes. The width of the grid in tensors
is one greater than the width of the grid in plaquettes.
"""
group_from_gridposition(i::Int, j::Int, w::Int) =
    (j-1)*(w) + i

"""
`i` is row, `j` is col, `w` is width, and `h` is height,
where `w` and `h` are in terms of tensors, not plaquettes.

The tensors at each corner of the grid each have their own type.
The tensors on each edge of the grid (but not on the corners) also
each have their own type. All tensors in the interior are one type.

In total there are 9 segmenttensortypes.

The 8 types on the perimeter are:
Corners:
- (1, 1) is TELBOW (bottom left)
- (1, h) is RELBOW_TVL_RELBOW (top left)
- (w, h) is BELBOW (top right)
- (w, 1) is LELBOW_LELBOW (bottom right)
Edges (* means freely ranging, but not taking the extreme values of 1 or w/h):
- (1, *) is VERTEX_TVL_RELBOW (left)
- (w, *) is LELBOW_VERTEX (right)
- (*, 1) is VERTEX_LELBOW (bottom)
- (*, h) is RELBOW_TVL_VERTEX (top)

And in the interior all tensors are VERTEX_TVL_VERTEX.
"""
function segmenttensortype_from_gridposition(i::Int, j::Int, w::Int, h::Int)
    # corners
    if i == 1 && j == 1 return TELBOW end
    if i == 1 && j == h return RELBOW_TVL_RELBOW end
    if i == w && j == h return BELBOW end
    if i == w && j == 1 return LELBOW_LELBOW end
    # edges
    if i == 1 return VERTEX_TVL_RELBOW end
    if i == w return LELBOW_VERTEX end
    if j == 1 return VERTEX_LELBOW end
    if j == h return RELBOW_TVL_VERTEX end
    # interior
    VERTEX_TVL_VERTEX
end

"""
Whether this segmenttensortype has a TVL (Tail VacuumLoop) in the middle of it. In
other words, does this segmenttensortype have a :mp physical index and a tail?

Generic type parameter should help compiler turn this into a lookup table?
"""
hasTVL(::Type{T}) where {T <: SegmentTensorType} =
    T ∈ [VERTEX_TVL_VERTEX, RELBOW_TVL_VERTEX, VERTEX_TVL_RELBOW, RELBOW_TVL_RELBOW]

"""The position of a segmenttensor is just its grid position."""
get_tpos(i::Int, j::Int) = Point2f(i, j)

"""
Calculates the positions of the physical indices belonging to this segmenttensor,
based on the provided grid position and grid size, and returns a mapping from IL to Point2f.
"""
function get_ipos(T::Type{<:SegmentTensorType}, i::Int, j::Int, w::Int)
    # get index positions
    hasmp = hasTVL(T)
    tailpos = ((i-j)*√3, (j-1+i-1)*3)
    separationdistance = 1
    topipos = Point2f(tailpos[1], tailpos[2] + separationdistance)
    midipos = tailpos
    botipos = Point2f(tailpos[1], tailpos[2] - separationdistance)
    # construct and return mapping
    g = group_from_gridposition(i, j, w)
    map = Dict(IL(g, :tp) => topipos, IL(g, :bp) => botipos)
    hasmp && push!(map, IL(g, :mp) => midipos)
    map
end

"""
Stores the segment's metadata, including group, grid, tensor, and index positions,
qubit mapping from index, and segmenttensortype. Also stores the grid width and height.
"""
struct Segment{T  <: SegmentTensorType}
    group::Int
    # grid, tensor, and index positions
    gpos::Tuple{Int, Int}
    tpos::Point2f
    ipos::Dict{IL, Point2f}
    # qubit mapping, assigned to after segment creation
    qubits::Dict{Symbol, Int}
    # grid parameters
    w::Int
    h::Int
    function Segment(i::Int, j::Int, w::Int, h::Int)
        stt = segmenttensortype_from_gridposition(i, j, w, h)
        group = group_from_gridposition(i, j, w)
        gpos = (i, j)
        tpos = get_tpos(i, j)
        ipos = get_ipos(stt, i, j, w)
        new{stt}(group, gpos, tpos, ipos, Dict(), w, h)
    end
end

"""Returns the segmenttensortype of this segmentwrapper."""
get_segmenttensortype(s::Segment{T}) where {T <: SegmentTensorType} = T

### SEGMENTGRID ###

"""
Mapping between grid position and segment. `w` and `h` are measured in tensors,
not in plaquettes. Stores the largest group assigned to any of its segments.
"""
struct SegmentGrid
    w::Int
    h::Int
    segments::Dict{Tuple{Int, Int}, Segment}
    maxgroup::Int
    function SegmentGrid(w::Int, h::Int)
        segments = Dict()
        maxgroup = group_from_gridposition(1, 1, w)
        for i in 1:w, j in 1:h
            segments[i, j] = Segment(i, j, w, h)
            newgroup = group_from_gridposition(i, j, w)
            maxgroup = newgroup > maxgroup ? newgroup : maxgroup
        end
       new(w, h, segments, maxgroup)
    end
end

"""
Creates the tensornetwork associated with `sg` by first iterating through
it and creating all segment tensors. It then autocontracts all extant
virtual indices by matching adjacent pairs of :U and :D or :R and :L ports.
"""
function create_segmentgrid_ttn(sg::SegmentGrid)
    ttn = TypedTensorNetwork()
    # create tensors
    for i in 1:sg.w, j in 1:sg.h
        segment = sg.segments[i, j]
        add_tensor!(ttn, segment.group, get_segmenttensortype(segment))
    end
    # autocontract all of the other segments together: go through each segment
    # and contract to the right and up
    for i in 1:sg.w, j in 1:sg.h
        g = sg.segments[i, j].group
        # right side
        if i < sg.w
            g_right = sg.segments[i+1, j].group
            ril = IL(g, :R)
            lil = IL(g_right, :L)
            add_contraction!(ttn.tn, IC(ril, lil))
        end
        # upper side
        if j < sg.h
            g_up = sg.segments[i, j+1].group
            uil = IL(g, :U)
            dil = IL(g_up, :D)
       	    add_contraction!(ttn.tn, IC(uil, dil))
        end
    end
    ttn
end

"""
Assigns qubits to each segment in the segment grid, going segment by segment.
For each segment, interior qubits are assigned first, then qubits to the right
and above are assigned. Qubits to the left and below are assigned while processing
the segments to the left and below respectively.

Any qubits going off the edges of the lattice are assigned to 0, and will always be
trivial.
"""
function assign_qubits(sg::SegmentGrid)
    qcounter = 1
    for j in 1:sg.h, i in 1:sg.w
        # assign qubits in interior of segment
        segment = sg.segments[i, j]
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
        if i < sg.w
            rsegment = sg.segments[i+1, j]
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
        if j < sg.h
            usegment = sg.segments[i, j+1]
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

"""Creates QubitLattice using the qubit assignments of segments in  `sg`."""
function create_segmentgrid_ql(sg::SegmentGrid)
    ql = QubitLattice()
    # iterate across segments, assigning qubits to pinds in an oriented way
    for i in 1:sg.w, j in 1:sg.h
        segment = sg.segments[i, j]
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

### INTEGRATION ###

"""
`w` and `h` are the dimension of the grid in plaquettes, not in tensors.
"""
function grid(w::Int, h::Int; displayTN=true, displayQL=true, interactiveQL=false)
    w, h = w+1, h+1 # number of tensors is one more than number of plaquettes
    sg = SegmentGrid(w, h)
    ttn = create_segmentgrid_ttn(sg)
    assign_qubits(sg)
    ql = create_segmentgrid_ql(sg)
    # create tpos, positions of tensors
    tpos = Dict{Int, Point2f}()
    for i in 1:sg.w, j in 1:sg.h
        segment = sg.segments[i, j]
        tpos[segment.group] = segment.tpos
    end
    # create ipos, positions of physical indices
    ipos = Dict{IL, Point2f}()
    for i in 1:sg.w, j in 1:sg.h
        merge!(ipos, sg.segments[i, j].ipos)
    end
    # display tensor network
    if displayTN
        tnf, _ = visualize(ttn, tpos)
        display(GLMakie.Screen(), tnf)
    end
    # calculate results
    es = ExecutionState(ttn)
    execsteps = [ContractionStep(c) for c in get_contractions(ttn.tn)]
    for execstep in execsteps
        execute_step!(es, execstep)
    end
    et = es.tensor_from_id[only(get_ids(es))]
    inds, data = et.indices, et.data
    states, amps = get_states_and_amps(ql, inds, data)
    # plot results
    if displayQL
        if interactiveQL
            qlf, _ = visualize(ql, ipos, inds, data; tail_length=√3/2)
        else
            qlf, _ = visualize(ql, ipos, states, amps; tail_length=√3/2)
        end
        display(GLMakie.Screen(), qlf)
    end
    ttn, tpos, ql, ipos, inds, data
end

function serialize_results(results::Any, casename::String)
    serialize(pwd() * "/out/" * casename, results)
end

function deserialize_results(casename::String)
    deserialize(pwd() * "/out/" * casename)
end
