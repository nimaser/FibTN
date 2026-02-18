export GridPosition, GridEdge, BoundaryConditionsDict, wrappos, ingrid

const GridPosition = Tuple{Int, Int}
const GridEdge = Tuple{GridPosition, GridPosition}

"""Wrap position `(x, y)` toroidally for periodic grids, does nothing for nonperiodic ones."""
@inline wrappos(x::Int, y::Int, w::Int, h::Int, wrap::Bool) =
    wrap ? (mod1(x, w), mod1(y, h)) : (x, y)

"""Determines whether the position `(x, y)` is in the bounds of a `w` x `h` grid."""
@inline ingrid(x::Int, y::Int, w::Int, h::Int, wrap::Bool) = begin
    x, y = wrappos(x, y, w, h, wrap)
    1 <= x <= w && 1 <= y <= h
end

"""
A `Dict{GridPosition, V}` that knows its grid dimensions and boundary conditions.

`Periodic` is a compile-time `Bool` type parameter. When `true`, index access wraps
toroidally, going from the maximum value back to 1: `d[i, j]` is equivalent to
`d[mod1(i, d.w), mod1(j, d.h)]`. When `false`, out-of-bounds indices simply return
default argument (via `get`) or throw `KeyError` (via `[]`).

Supports the same `getindex`, `setindex!`, `haskey`, `get`, `keys`, `values`, and
`iterate` interface as a plain `Dict`.
"""
struct BoundaryConditionsDict{Periodic, V}
    w::Int
    h::Int
    data::Dict{GridPosition, V}
end

Base.size(d::BoundaryConditionsDict) = d.w, d.h

"""Wraps grid indexing operations if the grid is periodic, does nothing otherwise."""
@inline wrappos(d::BoundaryConditionsDict{Periodic}, x::Int, y::Int) where {Periodic} = wrappos(x, y, d.w, d.h, Periodic)
@inline wrappos(d::BoundaryConditionsDict{Periodic}, pos::GridPosition) where {Periodic} = wrappos(pos..., d.w, d.h, Periodic)

"""Determines whether the position `(x, y)` is in the bounds of `d`'s grid."""
@inline ingrid(d::BoundaryConditionsDict{Periodic}, x::Int, y::Int) where {Periodic} = ingrid(x, y, d.w, d.h, Periodic)
@inline ingrid(d::BoundaryConditionsDict{Periodic}, pos::GridPosition) where {Periodic} = ingrid(pos..., d.w, d.h, Periodic)

"""Construct an empty `BoundaryConditionsDict` with given dimensions and periodicity."""
BoundaryConditionsDict{Periodic, V}(w::Int, h::Int) where {Periodic, V} =
    BoundaryConditionsDict{Periodic, V}(w, h, Dict{GridPosition, V}())

Base.getindex(d::BoundaryConditionsDict, x::Int, y::Int) = d.data[wrappos(d, x, y)]
Base.getindex(d::BoundaryConditionsDict, pos::GridPosition) = d.data[wrappos(d, pos...)]

Base.setindex!(d::BoundaryConditionsDict, v, x::Int, y::Int) = (d.data[wrappos(d, x, y)] = v)
Base.setindex!(d::BoundaryConditionsDict, v, pos::GridPosition) = (d.data[wrappos(d, pos...)] = v)

Base.haskey(d::BoundaryConditionsDict, x::Int, y::Int) = haskey(d.data, wrappos(d, x, y))
Base.haskey(d::BoundaryConditionsDict, pos::GridPosition) = haskey(d.data, wrappos(d, pos...))

Base.get(d::BoundaryConditionsDict, x::Int, y::Int, def) = get(d.data, wrappos(d, x, y), def)
Base.get(d::BoundaryConditionsDict, pos::GridPosition, def) = get(d.data, wrappos(d, pos...), def)

Base.delete!(d::BoundaryConditionsDict, x::Int, y::Int) = delete!(d.data, wrappos(d, x, y))
Base.delete!(d::BoundaryConditionsDict, pos::GridPosition) = delete!(d.data, wrappos(d, pos...))

Base.keys(d::BoundaryConditionsDict)   = keys(d.data)
Base.values(d::BoundaryConditionsDict) = values(d.data)
Base.iterate(d::BoundaryConditionsDict, state...) = iterate(d.data, state...)
Base.length(d::BoundaryConditionsDict) = length(d.data)
