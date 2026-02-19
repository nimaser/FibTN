export Segment, get_segmenttensortype

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
