export Segment, get_segmenttensortype

"""
Returns the position of the tensor at grid position `(x, y)`,
before any rotation is applied.

Basically converts a square grid to a hexagonal one:
- (1, 1) in the square grid becomes (0, 0)
- moving up in the square is to moving up-left in the hex grid
- moving right in the square is moving up-right in the hex grid
- moving diagonally up-right in the square is moving up in the hex grid

When this is rotated 30 degrees clockwise, it makes moving right in the
square and in the hex aligned, while moving up in the square moves up-
right in the hex. In other words, the rotation lines up one of the hex
basis vectors with one of the square basis vectors.
"""
_tailpos(x::Int, y::Int) = Point2f((x - y) * √3, (y + x - 2) * 3)

"""
θ = π/6 = 30° clockwise rotation matrix.
Applied as R * p to rotate a `Point2f` clockwise by θ.
"""
const _R = let θ = π/6; [cos(θ) sin(θ); -sin(θ) cos(θ)] end

"""
The position of the segment tensor in data coordinates, equal to the
rotated position of the `:MP` physical index (the tail/excitation slot).
"""
get_tpos(x::Int, y::Int) = Point2f(_R * _tailpos(x, y))

"""
Calculates the positions of the physical indices belonging to this segment,
based on the provided grid position `x, y`. :TP and :BP are always
`separationdistance` away from :MP, which is at the same location as the
segment tensor.

Returns a mapping from `IL` to `Point2f`.
"""
function get_ipos(T::Type{<:SegmentTensorType}, x::Int, y::Int, g::Int)
    separationdistance = 1
    raw = _tailpos(x, y)
    midipos = Point2f(_R * raw)
    topipos = Point2f(_R * (raw + Point2f(0,  separationdistance)))
    botipos = Point2f(_R * (raw + Point2f(0, -separationdistance)))
    map = Dict{IL, Point2f}()
    if hasTP(T) map[IL(g, :TP)] = topipos end
    if hasMP(T) map[IL(g, :MP)] = midipos end
    if hasBP(T) map[IL(g, :BP)] = botipos end
    map
end

"""
Stores the segment's metadata, including group, grid, tensor, and index positions,
and a qubit mapping (assigned after segment creation, during FibTN creation).
"""
struct Segment{T <: SegmentTensorType}
    group::Int
    # grid, tensor, and index positions
    gpos::Tuple{Int, Int}
    tpos::Point2f
    ipos::Dict{IL, Point2f}
    # qubit mapping, assigned after segment creation
    qubits::Dict{Symbol, Int}
    function Segment(::Type{T}, i::Int, j::Int, group::Int) where {T <: SegmentTensorType}
        gpos = (i, j)
        tpos = get_tpos(i, j)
        ipos = get_ipos(T, i, j, group)
        new{T}(group, gpos, tpos, ipos, Dict())
    end
end

"""Returns the segmenttensortype of this segment."""
get_segmenttensortype(s::Segment{T}) where {T <: SegmentTensorType} = T
