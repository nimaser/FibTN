export smgprint
export inferdirections!, fixmiddles!, segmentmaskgrid
export remove_segment_connection!, add_segment_connection!, remove_boundary_crossings!
export remove_segment!, replace_segment!
export add_right_boundaries!

smgprint(segmentmaskgrid::BoundaryConditionsDict{Periodic, Unsigned}) where {Periodic} =
    for (pos, m) in segmentmaskgrid @show pos, m, hex2mask(m) end

"""
Modifies `segmentmaskgrid` to include directional port flags `STT_U`, `STT_R`, `STT_D`,
and `STT_L`, inferred at each site from which adjacent sites are occupied. Acts as a
convenience constructor for a valid grid of SegmentTensorType masks.

In other words, `segmentmaskgrid` should map each `GridPosition` to a mask value describing
the segment at that site. In particular, entries of `segmentmaskgrid` should describe the
middle of the segment at that site; these entries are also used to determine site occupancy.

Possible entry values:
- `0` or no entry means the site is unoccupied
- `1` means the site is occupied, but has no `STT_M` (just top/bottom tensors)
- `STT_M`
- `STT_T`
- `STT_E`
- `STT_V`
and combinations thereof. `infermask` is called on each entry value.

Passing in a mask with a value of `STT_M | 0x1` (and likewise with `T`, `E`, `V`) is invalid
and results in undefined behavior.

Directional flags (`STT_U`, `STT_R`, `STT_D`, `STT_L`) are inferred automatically: a position
gets a directional flag iff its neighbor in that direction also exists in `masks`. Periodic
wrapping is accounted for by the `BoundaryConditionsDict` if `Periodic` is true.

Returns `nothing`.
"""
function inferdirections!(masks::BoundaryConditionsDict{Periodic, Unsigned}) where {Periodic}
    # no new entries added -> no rehashing -> safe to traverse and modify dict in same loop
    for ((x, y), middle_mask) in masks
        mask = (middle_mask == 1) ? 0 : infermask(middle_mask)
        # infer directional flags from which neighbours exist
        haskey(masks, x+1, y) && (mask |= STT_R)
        haskey(masks, x-1, y) && (mask |= STT_L)
        haskey(masks, x, y+1) && (mask |= STT_U)
        haskey(masks, x, y-1) && (mask |= STT_D)
        masks[x, y] = mask
    end
    nothing
end

"""
Fixes invalid middle flags in `segmentmaskgrid` by applying two local rules at each site:

- **Add middle**: if the mask has at least one top port and at least one bottom port, but no
  middle, and either side has exactly one port (so a middle is needed to bridge it), add
  `middle_mask` (default `STT_M`). Sites with only one port total, or where both sides already
  have two ports, are left unchanged.

- **Remove middle**: if the mask has a middle but the middle is invalid because there are two
  top ports (U and R) with no bottom ports (no D and no L), or two bottom ports (D and L)
  with no top ports (no U and no R), strip the middle flags (STT_M, STT_T, STT_E, STT_V).

Modifies `masks` in place. Returns `nothing`.
"""
function fixmiddles!(masks::BoundaryConditionsDict{Periodic, Unsigned}; middle_mask::Unsigned=STT_M) where {Periodic}
    middle_bits = STT_M | STT_T | STT_E | STT_V
    for ((x, y), mask) in masks
        has_top_single  = hasU(mask) ⊻ hasR(mask)
        has_bot_single  = hasD(mask) ⊻ hasL(mask)
        has_top_both    = hasU(mask) && hasR(mask)
        has_top_neither = !hasU(mask) && !hasR(mask)
        has_bot_both    = hasD(mask) && hasL(mask)
        has_bot_neither = !hasD(mask) && !hasL(mask)
        has_top_any     = hasU(mask) || hasR(mask)
        has_bot_any     = hasD(mask) || hasL(mask)
        if !hasM(mask) && has_top_any && has_bot_any && (has_top_single || has_bot_single)
            # at least one side has a single port: need a middle to bridge top and bottom
            masks[x, y] = mask | infermask(middle_mask)
        elseif hasM(mask) && (has_top_both && has_bot_neither || has_bot_both && has_top_neither)
            # middle is invalid: two-port side with no opposite side to connect to
            masks[x, y] = mask & ~middle_bits
        end
    end
    nothing
end

"""
Convenience constructor for a fully-occupied and fully-connected grid of SegmentTensorType
masks of dimensions `w`x`h`, which all share the same middle mask `middle`. Uses `inferdirections!`
internally, so the default `middle=0x01` value corresponds to no middle connection; this default
value will also cause the internal call to `fixmiddles!` to use a `middle_mask=0` value, which
makes it a no-op.

Returns a `BoundaryConditionsDict{periodic, Unsigned}`.
"""
function segmentmaskgrid(w::Int, h::Int; periodic::Bool=false, middle::Unsigned=0x01)
    masks = BoundaryConditionsDict{periodic, Unsigned}(w, h)
    for x in 1:w, y in 1:h masks[x, y] = middle end
    inferdirections!(masks)
    fixmiddles!(masks; middle_mask=middle & (STT_M | STT_T | STT_E | STT_V))
    masks
end

"""
Modifies `segmentmaskgrid` to strip the shared directional port on both sides of the
edge between `pos1` and `pos2`. `pos2` must be directly right of or above `pos1`.

Does not modify the middle flags on any site.

Throws `BoundsError` if the positions are not in the grid.
Throws `ArgumentError` if the positions are not adjacent.
Returns `nothing`.
"""
function remove_segment_connection!(
    segmentmaskgrid::BoundaryConditionsDict{Periodic, Unsigned},
    pos1::GridPosition,
    pos2::GridPosition
) where {Periodic}
    # check that positions are in bounds
    ingrid(segmentmaskgrid, pos1) ||
        throw(BoundsError("pos1=$pos1 not in grid"))
    ingrid(segmentmaskgrid, pos2) ||
        throw(BoundsError("pos2=$pos2 not in grid"))
    # check that positions are adjacent
    is_horizontal = pos2 == wrappos(segmentmaskgrid, pos1[1]+1, pos1[2])
    is_vertical   = pos2 == wrappos(segmentmaskgrid, pos1[1], pos1[2]+1)
    is_horizontal || is_vertical ||
        throw(ArgumentError("pos2 must be directly right of or above pos1, got pos1=$pos1, pos2=$pos2"))
    # modify the connection
    if is_horizontal
        segmentmaskgrid[pos1] = segmentmaskgrid[pos1] & ~STT_R
        segmentmaskgrid[pos2] = segmentmaskgrid[pos2] & ~STT_L
    else
        segmentmaskgrid[pos1] = segmentmaskgrid[pos1] & ~STT_U
        segmentmaskgrid[pos2] = segmentmaskgrid[pos2] & ~STT_D
    end
    nothing
end

"""
Modifies `segmentmaskgrid` to add a shared directional port on both sides of the
edge between `pos1` and `pos2`. `pos2` must be directly right of or above `pos1`.

Does not modify the middle flags on any site.

Throws `BoundsError` if the positions are not in the grid.
Throws `ArgumentError` if the positions are not adjacent.
Returns `nothing`.
"""
function add_segment_connection!(
    segmentmaskgrid::BoundaryConditionsDict{Periodic, Unsigned},
    pos1::GridPosition,
    pos2::GridPosition
) where {Periodic}
    # check that positions are in bounds
    ingrid(segmentmaskgrid, pos1) ||
        throw(BoundsError("pos1=$pos1 not in grid"))
    ingrid(segmentmaskgrid, pos2) ||
        throw(BoundsError("pos2=$pos2 not in grid"))
    # check that positions are adjacent
    is_horizontal = pos2 == wrappos(segmentmaskgrid, pos1[1]+1, pos1[2])
    is_vertical   = pos2 == wrappos(segmentmaskgrid, pos1[1], pos1[2]+1)
    is_horizontal || is_vertical ||
        throw(ArgumentError("pos2 must be directly right of or above pos1, got pos1=$pos1, pos2=$pos2"))
    # modify the connection
    if is_horizontal
        segmentmaskgrid[pos1] = segmentmaskgrid[pos1] | STT_R
        segmentmaskgrid[pos2] = segmentmaskgrid[pos2] | STT_L
    else
        segmentmaskgrid[pos1] = segmentmaskgrid[pos1] | STT_U
        segmentmaskgrid[pos2] = segmentmaskgrid[pos2] | STT_D
    end
    nothing
end

"""
Strip all wrap-around connections grom `segmentmaskgrid`, i.e. the `:R/:L` edge
between col `w` and col `1`, and the `:U/:D` edge between row `h` and row `1`.

Useful after calling `segmentmaskgrid` or `inferdirections!` on a periodic
`BoundaryConditionsDict` if you only want a few crossings.
"""
function remove_boundary_crossings!(segmentmaskgrid::BoundaryConditionsDict{Periodic, Unsigned}) where {Periodic}
    w, h = segmentmaskgrid.w, segmentmaskgrid.h
    for ((x, y), _) in segmentmaskgrid
        # strip wrap-around :R/:L pair when x == w (wraps to x == 1)
        if x == w && haskey(segmentmaskgrid, (1, y))
            remove_segment_connection!(segmentmaskgrid, (w, y), (1, y))
        end
        # strip wrap-around :U/:D pair when y == h (wraps to y == 1)
        if y == h && haskey(segmentmaskgrid, (x, 1))
            remove_segment_connection!(segmentmaskgrid, (x, h), (x, 1))
        end
    end
    nothing
end

"""
Removes the segment mask at `(x, y)` from `segmentmaskgrid` and strips the inward-facing
directions flags from all four neighbors.  Does not fix any invalid masks caused by this
operation.

Does not modify the middle flags on any site, other than removing the mask at `(x, y)`.

Throws BoundsError if `(x, y)` is not in the grid.
"""
function remove_segment!(segmentmaskgrid::BoundaryConditionsDict{Periodic, Unsigned}, x::Int, y::Int) where {Periodic}
    # check bounds
    ingrid(segmentmaskgrid, x, y) ||
        throw(BoundsError("pos1=$pos1 not in grid"))
    m = segmentmaskgrid[x, y]
    # strip the corresponding inward port from each neighbour that exists
    if hasR(m) && haskey(segmentmaskgrid, x+1, y)
        segmentmaskgrid[x+1, y] = m & ~STT_L
    end
    if hasL(m) && haskey(segmentmaskgrid, x-1, y)
        segmentmaskgrid[x-1, y] = m & ~STT_R
    end
    if hasU(m) && haskey(segmentmaskgrid, x, y+1)
        segmentmaskgrid[x, y+1] = m & ~STT_D
    end
    if hasD(m) && haskey(segmentmaskgrid, x, y-1)
        segmentmaskgrid[x, y-1] = m & ~STT_U
    end
    # delete the mask value
    delete!(segmentmaskgrid, x, y)
    nothing
end

remove_segment!(segmentmaskgrid::BoundaryConditionsDict{Periodic, Unsigned}, pos::GridPosition) where {Periodic} =
    remove_segment!(segmentmaskgrid, pos...)

"""
Replace only the middle flags (`STT_M`, `STT_T`, `STT_E`, `STT_V`) of the segment
at `(x, y)`, preserving all directional (`STT_U/R/D/L`) and boundary (`STT_B`) flags.

`new_middle_mask` should be a combination of `STT_M/T/E/V` flags (or `0` for no middle);
any other flags will be ignored. `infermask` is applied, so eg `STT_T` alone is accepted
and implies `STT_M`.

Throws `KeyError` if `(x, y)` is not present in `segmenttypes`.
"""
function replace_segment!(
    segmentmaskgrid::BoundaryConditionsDict{Periodic, Unsigned},
    x::Int,
    y::Int,
    new_middle_mask::Integer
) where {Periodic}
    haskey(segmentmaskgrid, x, y) || throw(KeyError((x, y)))
    # clear middle bits first while preserving U/R/D/L/B
    middle_bits = STT_M | STT_T | STT_E | STT_V
    newmask = segmentmaskgrid[x, y] & ~middle_bits
    # add new middle mask
    segmentmaskgrid[x, y] = newmask | (infermask(new_middle_mask) & middle_bits)
    nothing
end

"""
Sets `STT_B` on every segment in `masks` that has `STT_U` and `STT_R` but no `STT_M`.
All other segments are left unchanged.

Returns `nothing`.
"""
function add_right_boundaries!(masks::BoundaryConditionsDict{Periodic, Unsigned}) where {Periodic}
    for ((x, y), mask) in masks
        if hasU(mask) && hasR(mask) && !hasM(mask)
            masks[x, y] = mask | STT_B
        end
    end
    nothing
end
