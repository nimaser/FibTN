module SegmentTensorTypes

using ..FibTensorTypes
using ..TensorNetworks
import ..TensorNetworks: tensor_ports, tensor_data, tensor_color, tensor_marker
using ..TensorNetworks.TOBackend
import FibErrThresh: visualize

using GeometryBasics

export SegmentTensorType, tensor_ports, tensor_data, tensor_color, tensor_marker
export STT_U, hasU
export STT_R, hasR
export STT_D, hasD
export STT_L, hasL
export STT_M, hasM
export STT_T, hasT
export STT_E, hasE
export STT_V, hasV
export STT_B, hasB
export hasTP, hasMP, hasBP
export getmask, hex2mask, infermask, validatemask

const IL = IndexLabel
const IC = IndexContraction

### TENSOR TYPES ###

"""
Each segment is a small tensor network with a predefined structure/connectivity.
Different segment types are variants where different tensors are inserted into
different 'slots' in this prestructured tensor network. Each segment can be
conceptualized as three parts, a top, middle, and bottom:
- set of tensors with external ports :U, :R, :TP
- set of tensors with external ports :a, :b, :S, :MP
- set of tensors with external ports :D, :L, :BP
Depending on the tensors inserted, some of the ports listed above might be missing,
but none will be added. Each segment can have at maximum 11 tensors.

Each `SegmentTensorType` is generic over a `Mask` numeric value, which acts as a
bitmask specifying the variant. Because bits are known at compile-time, this doesn't
induce any runtime penalty.

The Mask specifies user-facing properties of the segment, such as what ports it has,
whether it contains certain tensors, etc, and this information is used to automatically
calculate the required tensor network.

Mask values:
- `STT_U` (has up port)
- `STT_R` (has right port)
- `STT_D` (has down port)
- `STT_L` (has left port)
- `STT_M` (has middle tensors)
- `STT_T` (has tail tensor)
- `STT_E` (has excitation tensor)
- `STT_V` (has vacuum loop tensor)
- `STT_B` (right port has boundary tensor)

Mask validity rules:
- *`STT_M` must be flagged if any of `STT_T`, `STT_E`, or `STT_V` are also flagged
- *`STT_B` can only be flagged if `STT_R` is flagged
- `STT_M` must be flagged if only one of (`STT_U` and `STT_R`) or (`STT_D` and `STT_L`)
are also flagged
- `STT_M` cannot be flagged if both `STT_U` and `STT_R` or both `STT_D` and `STT_L` are unflagged
- `STT_T` and `STT_E` are mutually exclusive

There may be multiple Mask values that logically would lead to the same SegmentTensorType.
However, we want each segment variant to be uniquely identified by one Mask value. Therefore,
the rules marked with an * above were introduce to 'canonicalize' the Mask choices so that
only one of the possibilities that would lead to a given segment variant is a valid mask. See
`infermask` for more.
"""
struct SegmentTensorType{Mask} <: TensorType end

# bitmask options, with STT_ prefix to help avoid name conflicts
const STT_U = 0b000000001
const STT_R = 0b000000010
const STT_D = 0b000000100
const STT_L = 0b000001000
const STT_M = 0b000010000
const STT_T = 0b000100000
const STT_E = 0b001000000
const STT_V = 0b010000000
const STT_B = 0b100000000

### MASK PARSING AND VALIDATION ###

"""Checks whether the flag `i` is set in `Mask`."""
@inline _check(Mask, i) = !iszero(Mask & i)

"""Returns whether mask has the `STT_U` (up) flag set."""
@inline hasU(mask::Integer) = _check(mask, STT_U)
@inline hasU(::Type{SegmentTensorType{Mask}}) where {Mask} = hasU(Mask)

"""Returns whether mask has the `STT_R` (right) flag set."""
@inline hasR(mask::Integer) = _check(mask, STT_R)
@inline hasR(::Type{SegmentTensorType{Mask}}) where {Mask} = hasR(Mask)

"""Returns whether mask has the `STT_D` (down) flag set."""
@inline hasD(mask::Integer) = _check(mask, STT_D)
@inline hasD(::Type{SegmentTensorType{Mask}}) where {Mask} = hasD(Mask)

"""Returns whether mask has the `STT_L` (left) flag set."""
@inline hasL(mask::Integer) = _check(mask, STT_L)
@inline hasL(::Type{SegmentTensorType{Mask}}) where {Mask} = hasL(Mask)

"""Returns whether mask has the `STT_M` (middle) flag set."""
@inline hasM(mask::Integer) = _check(mask, STT_M)
@inline hasM(::Type{SegmentTensorType{Mask}}) where {Mask} = hasM(Mask)

"""Returns whether mask has the `STT_T` (tail) flag set."""
@inline hasT(mask::Integer) = _check(mask, STT_T)
@inline hasT(::Type{SegmentTensorType{Mask}}) where {Mask} = hasT(Mask)

"""Returns whether mask has the `STT_E` (excitation) flag set."""
@inline hasE(mask::Integer) = _check(mask, STT_E)
@inline hasE(::Type{SegmentTensorType{Mask}}) where {Mask} = hasE(Mask)

"""Returns whether mask has the `STT_V` (vacuum loop) flag set."""
@inline hasV(mask::Integer) = _check(mask, STT_V)
@inline hasV(::Type{SegmentTensorType{Mask}}) where {Mask} = hasV(Mask)

"""Returns whether mask has the `STT_B` (boundary) flag set."""
@inline hasB(mask::Integer) = _check(mask, STT_B)
@inline hasB(::Type{SegmentTensorType{Mask}}) where {Mask} = hasB(Mask)

"""Returns whether the segment type has a top physical port `:TP`."""
@inline hasTP(mask::Integer) = hasU(mask) || hasR(mask)
@inline hasTP(::Type{SegmentTensorType{Mask}}) where {Mask} = hasTP(Mask)

"""Returns whether the segment type has a middle physical port `:MP`."""
@inline hasMP(mask::Integer) = hasT(mask) || hasE(mask)
@inline hasMP(::Type{SegmentTensorType{Mask}}) where {Mask} = hasMP(Mask)

"""Returns whether the segment type has a bottom physical port `:BP`."""
@inline hasBP(mask::Integer) = hasD(mask) || hasL(mask)
@inline hasBP(::Type{SegmentTensorType{Mask}}) where {Mask} = hasBP(Mask)

"""Returns `Mask` of a `SegmentTensorType{Mask}`."""
@inline getmask(::Type{SegmentTensorType{Mask}}) where {Mask} = Mask

"""
Returns a human-readable string for a `mask` value, showing each flag name
or `_` if that flag is not set. Flags are listed in bit order: U R D L M T E V B.

Example: `hex2mask(STT_U | STT_R | STT_M)` → `"U R _ _ M _ _ _ _"`
"""
function hex2mask(mask::Integer)
    flags = (STT_U => "U", STT_R => "R", STT_D => "D", STT_L => "L",
             STT_M => "M", STT_T => "T", STT_E => "E", STT_V => "V", STT_B => "B")
    join((_check(mask, bit) ? name : "_" for (bit, name) in flags), " ")
end

"""
Convert `mask` to the unique canonical form required by `SegmentTensorType`.

Canonical form in this context refers to the single mask value which should identify
a `SegmentTensorType` variant, even though multiple mask values could logically
refer to it. For example, consider the segments `STT_U | STT_R | STT_B` and
`STT_U | STT_B`: these are logically the same, as boundary tensors are defined to
always attach to and extend the right port. A mask validity rule makes a mask with
`STT_B` with no associated `STT_R` invalid, solely so we have a 1-to-1 mapping
between masks and segment types. This ensures caching and dispatch are as efficient
as possible.

However, it is convenient to not have to write out, for example, `STT_M | STT_E`
when we just want to add an excitation tensor somewhere. This function takes `mask`
and applies two specific modifications:
- If any of `STT_T`, `STT_E`, or `STT_V` are set, `STT_M` is also set (a middle
  tensor always requires the middle flag).
- If `STT_B` is set, `STT_R` is also set (a boundary tensor always requires a
  right port).

All other combinations are left unchanged.

This function should be able to be constant folded and inlined by the compiler.
"""
@inline function infermask(mask::Integer)
    # middle tensors imply STT_M
    if hasT(mask) || hasE(mask) || hasV(mask)
        mask |= STT_M
    end
    # boundary tensor implies STT_R
    if hasB(mask)
        mask |= STT_R
    end
    mask
end

"""
Validates `Mask` to make sure no illegal combinations have been passed.

Mask validity rules:
- *`STT_M` must be flagged if any of `STT_T`, `STT_E`, or `STT_V` are also flagged
- *`STT_B` can only be flagged if `STT_R` is flagged
- `STT_M` must be flagged if only one of (`STT_U` and `STT_R`) or (`STT_D` and `STT_L`)
are also flagged
- `STT_M` cannot be flagged if both `STT_U` and `STT_R` or both `STT_D` and `STT_L` are unflagged
- `STT_T` and `STT_E` are mutually exclusive
- 0x0 is not a valid mask

Throws an error if `Mask` is invalid, otherwise returns its argument.

This function should be able to be constant folded and inlined by the compiler.
"""
@inline function validatemask(::Type{SegmentTensorType{Mask}}) where {Mask}
    Mask != 0 || throw(ArgumentError("invalid mask: mask cannot be 0"))
    if !hasM(Mask)
        if hasU(Mask) && !hasR(Mask) || !hasU(Mask) && hasR(Mask)
            throw(ArgumentError("invalid mask $Mask: cannot have only one top port and no middle"))
        end
        if hasD(Mask) && !hasL(Mask) || !hasD(Mask) && hasL(Mask)
            throw(ArgumentError("invalid mask $Mask: cannot have only one bottom port and no middle"))
        end
        if hasT(Mask) || hasE(Mask) || hasV(Mask)
            throw(ArgumentError("invalid mask $Mask: cannot have a middle tensor but no middle flag"))
        end
    else
        if !hasU(Mask) && !hasR(Mask)
            throw(ArgumentError("invalid mask $Mask: cannot have middle but no top and no right"))
        end
        if !hasD(Mask) && !hasL(Mask)
            throw(ArgumentError("invalid mask $Mask: cannot have middle but no down and no left"))
        end
    end
    if hasT(Mask) && hasE(Mask)
        throw(ArgumentError("invalid mask $Mask: cannot have both a tail and an excitation"))
    end
    if hasB(Mask) && !hasR(Mask)
        throw(ArgumentError("invalid mask $Mask: cannot have boundary with no right port"))
    end
    SegmentTensorType{Mask} # pass through argument
end

"""
Fetches the top and bottom tensor types for this segment variant, based on `Mask`.
Undefined behavior if called on an invalid `Mask`.

If a middle is present, the top (bottom) is VERTEX, ELBOW_T2, ELBOW_T1, or nothing,
if UR, U, R, or neither (DL, D, L, or neither) are present.

If a middle is not present, then U and R (D and L) must both be present, and the
top (bottom) tensor is ELBOW_T3.

This function should be able to be constant folded and inlined by the compiler.
"""
@inline function _fetch_top_bottom(::Type{SegmentTensorType{Mask}}) where {Mask}
    # extract top and bottom
    if hasM(Mask)
        if hasU(Mask)
            toptensor = hasR(Mask) ? VERTEX : ELBOW_T2
        else
            toptensor = hasR(Mask) ? ELBOW_T1 : nothing
        end
        if hasD(Mask)
            bottensor = hasL(Mask) ? VERTEX : ELBOW_T2
        else
            bottensor = hasL(Mask) ? ELBOW_T1 : nothing
        end
    else
        toptensor = hasU(Mask) ? ELBOW_T3 : nothing
        bottensor = hasD(Mask) ? ELBOW_T3 : nothing
    end
    toptensor, bottensor
end

"""Returns the TypedTensorNetwork `ttn` representing this segment."""
function _segment_ttn(::Type{SegmentTensorType{Mask}}) where {Mask}
    ttn = TypedTensorNetwork()
    # add top and bottom tensors
    toptensor, bottensor = _fetch_top_bottom(SegmentTensorType{Mask})
    hasTP(Mask) && add_tensor!(ttn, 1, toptensor)
    hasBP(Mask) && add_tensor!(ttn, 7, bottensor)
    # add reflectors onto top tensor if it has the appropriate indices
    if hasU(Mask)
        add_tensor!(ttn, 8, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(1, :V1), IL(8, :V2)))
    end
    if hasR(Mask)
        add_tensor!(ttn, 9, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(1, :V2), IL(9, :V1)))
    end
    # add boundary extension towards the right if needed
    if hasB(Mask)
        add_tensor!(ttn, 10, BOUNDARY)
        add_tensor!(ttn, 11, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(9, :V2), IL(10, :V2)))
        add_contraction!(ttn.tn, IC(IL(10, :V1), IL(11, :V1)))
    end
    # add middle tensors
    if hasM(Mask)
        add_tensor!(ttn, 2, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(1, :V3), IL(2, :V1)))
    end
    if hasT(Mask)
        add_tensor!(ttn, 3, TAIL)
        add_tensor!(ttn, 4, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(2, :V2), IL(3, :V1)))
        add_contraction!(ttn.tn, IC(IL(3, :V2), IL(4, :V1)))
    end
    if hasE(Mask)
        add_tensor!(ttn, 3, EXCITATION)
        add_tensor!(ttn, 4, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(2, :V2), IL(3, :V1)))
        add_contraction!(ttn.tn, IC(IL(3, :V2), IL(4, :V1)))
    end
    if hasV(Mask)
        add_tensor!(ttn, 5, VACUUMLOOP)
        add_tensor!(ttn, 6, REFLECTOR)
        try_contraction!(ttn.tn, IC(IL(2, :V2), IL(5, :V1))) # no hasE/hasT
        try_contraction!(ttn.tn, IC(IL(4, :V2), IL(5, :V1))) # yes hasE/hasT
        add_contraction!(ttn.tn, IC(IL(5, :V2), IL(6, :V1)))
    end
    # try to add contractions to bottom tensor
    try_contraction!(ttn.tn, IC(IL(2, :V2), IL(7, :V3))) # no hasV;     no hasE/hasT
    try_contraction!(ttn.tn, IC(IL(4, :V2), IL(7, :V3))) # no hasV;     yes hasE/hasT
    try_contraction!(ttn.tn, IC(IL(6, :V2), IL(7, :V3))) # yes hasV;    yes hasE/hasT
    ttn
end

"""
Returns a Dict `idx_map` which maps between the segment's ports (when seen
as a single tensor) and the `IndexLabel`s of its `ttn`.
"""
function _segment_idx_map(::Type{SegmentTensorType{Mask}}) where {Mask}
    idx_map = Dict{Symbol,IL}()
    # these entries only get utilized if the tensor actually has the ports,
    # and other than :R they are all non-interfering, so there is no need to
    # overspecialize on the Mask
    idx_map[:a] = IL(3, :a)
    idx_map[:b] = IL(3, :b)
    idx_map[:l] = IL(3, :l)
    idx_map[:U] = IL(8, :V1)
    idx_map[:R] = hasB(Mask) ? IL(11, :V2) : IL(9, :V2)
    idx_map[:D] = IL(7, :V1)
    idx_map[:L] = IL(7, :V2)
    idx_map[:S] = IL(3, :S)
    idx_map[:TP] = IL(1, :P)
    idx_map[:MP] = IL(3, :P)
    idx_map[:BP] = IL(7, :P)
    idx_map
end

### TENSOR PORTS ###

const _cachedports = IdDict{DataType, Any}()

function tensor_ports(::Type{SegmentTensorType{Mask}}) where {Mask}
    # try to get the ports from the cache, else generate them
    get!(_cachedports, SegmentTensorType{Mask}) do
        _generate_tensor_ports(validatemask(SegmentTensorType{Mask}))
    end
end

"""
Excitation tensors have control indices `:a` and `:b` which come first. Next
come `:U`, `:R`, `:D`, `:L` if they exist. Then the excitation tensor's `:S`
if it exists, and finally the physical ports `:TP`, `:MP`, and `:BP`, if applicable.
"""
function _generate_tensor_ports(::Type{SegmentTensorType{Mask}}) where {Mask}
    # get ports
    ports = Symbol[]
    if hasE(Mask) push!(ports, :a, :b, :l) end
    if hasU(Mask) push!(ports, :U) end
    if hasR(Mask) push!(ports, :R) end
    if hasD(Mask) push!(ports, :D) end
    if hasL(Mask) push!(ports, :L) end
    if hasE(Mask) push!(ports, :S) end
    if hasTP(Mask) push!(ports, :TP) end
    if hasMP(Mask) push!(ports, :MP) end
    if hasBP(Mask) push!(ports, :BP) end
    Tuple(ports)
end

### TENSOR DATA ###

const _cacheddata = IdDict{DataType,Any}()

function tensor_data(::Type{SegmentTensorType{Mask}}) where {Mask}
    # try to get the data from the cache, else generate it
    get!(_cacheddata, SegmentTensorType{Mask}) do
        _generate_tensor_data(validatemask(SegmentTensorType{Mask}))
    end
end

"""
Obtains the ttn for a segment, contracts it, then permutes the data
so that it aligns with the ports outputted by `tensor_ports`.
"""
function _generate_tensor_data(::Type{SegmentTensorType{Mask}}) where {Mask}
    # build the typed tensor network
    ttn::TypedTensorNetwork = _segment_ttn(SegmentTensorType{Mask})
    # contract using execution state
    es = ExecutionState(ttn)
    execsteps = [ContractionStep(c) for c in get_contractions(ttn.tn)]
    for execstep in execsteps execute_step!(es, execstep) end
    # for the case with a top and bottom but no middle, the tensor
    # product is needed to result in only a single tensor
    if length(get_ids(es)) > 1
        step = OuterProductStep(IL(1, :P), IL(7, :P)) # we hardcode these so we get an error if we change something
        execute_step!(es, step)
    end
    # permute indices to match ports
    id = only(get_ids(es))
    idx_map = _segment_idx_map(SegmentTensorType{Mask})
    indordering = [idx_map[p] for p in tensor_ports(SegmentTensorType{Mask})]
    execute_step!(es, PermuteIndicesStep(id, indordering))
    et = es.tensor_from_id[id]
    et.data
end

### TENSOR DISPLAY PROPERTIES ###

function tensor_color(::Type{SegmentTensorType{Mask}}) where {Mask}
    # excitation tensors are red, all others are gray
    hasE(Mask) ? :red : :gray
end

function tensor_marker(::Type{SegmentTensorType{Mask}}) where {Mask}
    :rect
end

### SEGMENT TTN VISUALIZATION ###

"""Positions for each group number in the segment ttn"""
const _SEGMENT_TTN_POSITIONS = Dict{Int, Point2f}(
    1 => Point2f(0, 6),
    2 => Point2f(0, 5),
    3 => Point2f(0, 4),
    4 => Point2f(0, 3),
    5 => Point2f(0, 2),
    6 => Point2f(0, 1),
    7 => Point2f(0, 0),
    8 => Point2f(-1, 7),
    9 => Point2f(1, 7),
    10 => Point2f(2, 8),
    11 => Point2f(3, 9),
)

"""
Visualize the internal typed tensor network for segment type `T`.
"""
function visualize(::Type{SegmentTensorType{Mask}}) where {Mask}
    ttn = _segment_ttn(SegmentTensorType{Mask})
    visualize(ttn, _SEGMENT_TTN_POSITIONS)
end

end # module SegmentTensorTypes
