export SegmentGrid

"""
Mapping between grid position and segment. `w` and `h` are measured in tensors,
not in plaquettes. Stores the largest group assigned to any of its segments.

Generates the tensor network and qubit lattice for this grid configuration.
Generates positions for each tensor and each physical index to be passed
to visualization functions.
"""
struct SegmentGrid
    w::Int
    h::Int
    segments::Dict{Tuple{Int, Int}, Segment}
    maxgroup::Int
    ttn::TypedTensorNetwork
    ql::QubitLattice
    tpos::Dict{Int, Point2f}
    ipos::Dict{IL, Point2f}
    function SegmentGrid(w::Int, h::Int)
        w > 1 && h > 1 || throw(ArgumentError("width and height must both be greater than 1"))
        segments = Dict{Tuple{Int, Int}, Segment}()
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
       new(w, h, segments, maxgroup, ttn, ql, tpos, ipos)
    end
end

"""Convenience function so `SegmentGrid[i, j] = SegmentGrid.segments[i, j]."""
Base.getindex(sg::SegmentGrid, i::Int, j::Int) =
    sg.segments[i, j]

"""
Creates the tensornetwork associated with `sg` by first iterating through
it and creating all segment tensors. It then autocontracts all extant
virtual indices by matching adjacent pairs of :U and :D or :R and :L ports.
"""
function create_ttn(w::Int, h::Int, segments::Dict{Tuple{Int, Int}, Segment})
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
function assign_qubits(w::Int, h::Int, segments::Dict{Tuple{Int, Int}, Segment})
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
function create_ql(w::Int, h::Int, segments::Dict{Tuple{Int, Int}, Segment})
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
function create_tpos(w::Int, h::Int, segments::Dict{Tuple{Int, Int}, Segment})
    tpos = Dict{Int, Point2f}()
    for i in 1:w, j in 1:h
        segment = segments[i, j]
        tpos[segment.group] = segment.tpos
    end
    tpos
end

"""Returns `Dict` from physical index label to vertex position."""
function create_ipos(w::Int, h::Int, segments::Dict{Tuple{Int, Int}, Segment})
    ipos = Dict{IL, Point2f}()
    for i in 1:w, j in 1:h
        merge!(ipos, segments[i, j].ipos)
    end
    ipos
end
