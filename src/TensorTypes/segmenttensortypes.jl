module SegmentTensorTypes

using ..FibTensorTypes
using ..TensorNetworks
import ..TensorNetworks: tensor_ports, tensor_data, tensor_color, tensor_marker
using ..TensorNetworks.TOBackend
import FibErrThresh: visualize

using GeometryBasics

export SegmentTensorType
export BELBOW, TELBOW, VERTEX_TVL_VERTEX
export VERTEX_TVL_RELBOW, VERTEX_LELBOW
export RELBOW_TVL_VERTEX, LELBOW_VERTEX
export RELBOW_TVL_RELBOW, LELBOW_LELBOW
export VERTEX_STRINGEND_VERTEX
export VERTEX_EXCITATION_VERTEX
export visualize

const IL = IndexLabel
const IC = IndexContraction

### TENSOR TYPES ###

abstract type SegmentTensorType <: TensorType end

# TVL = "Tail VacuumLoop"
struct BELBOW <: SegmentTensorType end
struct TELBOW <: SegmentTensorType end
struct VERTEX_TVL_VERTEX <: SegmentTensorType end

struct VERTEX_TVL_RELBOW <: SegmentTensorType end
struct RELBOW_TVL_VERTEX <: SegmentTensorType end
struct RELBOW_TVL_RELBOW <: SegmentTensorType end

struct VERTEX_LELBOW <: SegmentTensorType end
struct LELBOW_VERTEX <: SegmentTensorType end
struct LELBOW_LELBOW <: SegmentTensorType end

struct VERTEX_STRINGEND_VERTEX <: SegmentTensorType end
struct VERTEX_EXCITATION_VERTEX <: SegmentTensorType end

### TENSOR PORTS ###

tensor_ports(::Type{BELBOW}) = (:D, :L, :BP)
tensor_ports(::Type{TELBOW}) = (:U, :R, :TP)
tensor_ports(::Type{VERTEX_TVL_VERTEX}) = (:U, :R, :D, :L, :TP, :MP, :BP)

tensor_ports(::Type{VERTEX_TVL_RELBOW}) = (:U, :R, :D, :TP, :MP, :BP)
tensor_ports(::Type{RELBOW_TVL_VERTEX}) = (:R, :D, :L, :TP, :MP, :BP)
tensor_ports(::Type{RELBOW_TVL_RELBOW}) = (:R, :D, :TP, :MP, :BP)

tensor_ports(::Type{VERTEX_LELBOW}) = (:U, :R, :L, :TP, :BP)
tensor_ports(::Type{LELBOW_VERTEX}) = (:U, :D, :L, :TP, :BP)
tensor_ports(::Type{LELBOW_LELBOW}) = (:U, :L, :TP, :BP)

tensor_ports(::Type{VERTEX_STRINGEND_VERTEX}) = (:α, :β, :k, :l, :U, :R, :D, :L, :S, :TP, :MP, :BP)
tensor_ports(::Type{VERTEX_EXCITATION_VERTEX}) = (:a, :b, :U, :R, :D, :L, :S, :TP, :MP, :BP)

### TENSOR DATA ###

const _cache = IdDict{DataType,Any}()

function tensor_data(::Type{T}) where {T<:SegmentTensorType}
    # try to get the data from the cache, else generate it
    get!(_cache, T) do
        _generate_tensor_data(T)
    end
end

"""
Constructs the internal `TypedTensorNetwork` for segment type `T`,
including `REFLECTOR` tensors on the top and right indices to allow
easy tiling. This is the network that gets contracted to produce
the segment's tensor data.
"""
function _segment_ttn(::Type{T}) where {T<:SegmentTensorType}
    if T == BELBOW
        ttn = TypedTensorNetwork()
        add_tensor!(ttn, 7, ELBOW_T3)
        return ttn
    elseif T == TELBOW
        ttn = TypedTensorNetwork()
        add_tensor!(ttn, 1, ELBOW_T3)
        add_tensor!(ttn, 11, REFLECTOR)
        add_tensor!(ttn, 10, BOUNDARY)
        add_tensor!(ttn, 9, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(1, :V2), IL(11, :V1)))
        add_contraction!(ttn.tn, IC(IL(11, :V2), IL(10, :V2)))
        add_contraction!(ttn.tn, IC(IL(10, :V1), IL(9, :V1)))
    else
        ttn = _make_segment_ttn(_segment_tensors(T))
    end
    # add reflection tensors to the top and right, if those indices exist
    if has_index(ttn.tn, IL(1, :V1)) && !has_contraction(ttn.tn, IL(1, :V1))
        add_tensor!(ttn, 8, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(1, :V1), IL(8, :V2)))
    end
    if has_index(ttn.tn, IL(1, :V2)) && !has_contraction(ttn.tn, IL(1, :V2))
        add_tensor!(ttn, 9, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(1, :V2), IL(9, :V1)))
    end
    ttn
end

"""
For nontrivial segments (composed of more than one tensor), generates
the `TypedTensorNetwork` via `_segment_ttn`, contracts it, then
permutes the data so that it aligns with the ports outputted by
`tensor_ports`.
"""
function _generate_tensor_data(::Type{T}) where {T<:SegmentTensorType}
    # this case is just an alias with different port names
    if T == BELBOW return tensor_data(ELBOW_T3) end
    # build the typed tensor network
    ttn = _segment_ttn(T)
    # contract using execution state
    es = ExecutionState(ttn)
    execsteps = [ContractionStep(c) for c in get_contractions(ttn.tn)]
    for execstep in execsteps execute_step!(es, execstep) end
    # mapping establishing which indices on the segment (seen as a ttn)
    # correspond to which ports on the segment (seen as a tensor)
    idx_map = Dict{Symbol,IL}()
    push!(idx_map, :α => IL(3, :α))
    push!(idx_map, :β => IL(3, :β))
    push!(idx_map, :k => IL(3, :k))
    push!(idx_map, :l => IL(3, :l))
    push!(idx_map, :a => IL(3, :a))
    push!(idx_map, :b => IL(3, :b))
    push!(idx_map, :S => IL(3, :S))
    push!(idx_map, :U => IL(8, :V1))
    push!(idx_map, :R => IL(9, :V2))
    push!(idx_map, :D => IL(7, :V1))
    push!(idx_map, :L => IL(7, :V2))
    push!(idx_map, :TP => IL(1, :P))
    push!(idx_map, :MP => IL(3, :P))
    push!(idx_map, :BP => IL(7, :P))
    # permute indices to match ports
    id = only(get_ids(es))
    indordering = [idx_map[p] for p in tensor_ports(T)]
    execute_step!(es, PermuteIndicesStep(id, indordering))
    et = es.tensor_from_id[id]
    et.data
end

_segment_tensors(::Type{VERTEX_TVL_VERTEX}) =
    (VERTEX, REFLECTOR, ELBOW_T2, REFLECTOR, VACUUMLOOP, REFLECTOR, VERTEX)

_segment_tensors(::Type{VERTEX_TVL_RELBOW}) =
    (VERTEX, REFLECTOR, ELBOW_T2, REFLECTOR, VACUUMLOOP, REFLECTOR, ELBOW_T2)

_segment_tensors(::Type{RELBOW_TVL_VERTEX}) =
    (ELBOW_T1, REFLECTOR, ELBOW_T2, REFLECTOR, VACUUMLOOP, REFLECTOR, VERTEX)

_segment_tensors(::Type{RELBOW_TVL_RELBOW}) =
    (ELBOW_T1, REFLECTOR, ELBOW_T2, REFLECTOR, VACUUMLOOP, REFLECTOR, ELBOW_T2)

_segment_tensors(::Type{VERTEX_LELBOW}) =
    (VERTEX, REFLECTOR, ELBOW_T1)

_segment_tensors(::Type{LELBOW_VERTEX}) =
    (ELBOW_T2, REFLECTOR, VERTEX)

_segment_tensors(::Type{LELBOW_LELBOW}) =
    (ELBOW_T2, REFLECTOR, ELBOW_T1)

_segment_tensors(::Type{VERTEX_STRINGEND_VERTEX}) =
    (VERTEX, REFLECTOR, STRINGEND, REFLECTOR, VERTEX)

_segment_tensors(::Type{VERTEX_EXCITATION_VERTEX}) =
    (VERTEX, REFLECTOR, EXCITATION, REFLECTOR, VERTEX)

"""
Helper function to make the 7-tensor version of a segment. The
provided tensor types are given group numbers from 1 to 7, and
contracted according to their common structure, i.e. the top
tensor's c port is connected to 1, :a, then a contraction chain
of :b to :a is made going down to tensor 6, whose :b is then
contracted with tensor 7's :c.
"""
function _make_segment_ttn(tts::NTuple{7,DataType})
    # this is the version with TVL
    ttn = TypedTensorNetwork()
    for (i, tt) in enumerate(tts)
        add_tensor!(ttn, i, tt)
    end
    add_contraction!(ttn.tn, IC(IL(1, :V3), IL(2, :V1)))
    add_contraction!(ttn.tn, IC(IL(2, :V2), IL(3, :V1)))
    add_contraction!(ttn.tn, IC(IL(3, :V3), IL(4, :V1)))
    add_contraction!(ttn.tn, IC(IL(4, :V2), IL(5, :V1)))
    add_contraction!(ttn.tn, IC(IL(5, :V2), IL(6, :V1)))
    add_contraction!(ttn.tn, IC(IL(6, :V2), IL(7, :V3)))
    ttn
end

"""
Helper function to make the 5-tensor version of a segment. Tensor
groups at the ends are the same as the 7-tensor segment, to allow
code reuse.
"""
function _make_segment_ttn(tts::NTuple{5,DataType})
    # this is the version with STRINGEND
    ttn = TypedTensorNetwork()
    for (i, tt) in zip([1, 2, 3, 4, 7], tts)
        add_tensor!(ttn, i, tt)
    end
    add_contraction!(ttn.tn, IC(IL(1, :V3), IL(2, :V1)))
    add_contraction!(ttn.tn, IC(IL(2, :V2), IL(3, :U)))
    add_contraction!(ttn.tn, IC(IL(3, :D), IL(4, :V1)))
    add_contraction!(ttn.tn, IC(IL(4, :V2), IL(7, :V3)))
    ttn
end

"""
Helper function to make the 3-tensor version of a segment. Tensor
groups at the ends are the same as the 7-tensor segment, to allow
code reuse: the top tensor is group 1, the middle is 2, and the
bottom is 7, with contractions made between 1c and 2a and 2b and 7c.
"""
function _make_segment_ttn(tts::NTuple{3,DataType})
    # this is the version without TVL
    ttn = TypedTensorNetwork()
    for (i, tt) in zip([1, 2, 7], tts)
        add_tensor!(ttn, i, tt)
    end
    add_contraction!(ttn.tn, IC(IL(1, :V3), IL(2, :V1)))
    add_contraction!(ttn.tn, IC(IL(2, :V2), IL(7, :V3)))
    ttn
end

### TENSOR DISPLAY PROPERTIES ###

tensor_color(::Type{BELBOW }) = :blue
tensor_color(::Type{TELBOW }) = :blue
tensor_color(::Type{VERTEX_TVL_VERTEX }) = :black
tensor_color(::Type{VERTEX_TVL_RELBOW }) = :gray
tensor_color(::Type{RELBOW_TVL_VERTEX }) = :gray
tensor_color(::Type{RELBOW_TVL_RELBOW }) = :gray
tensor_color(::Type{VERTEX_LELBOW }) = :gray
tensor_color(::Type{LELBOW_VERTEX }) = :gray
tensor_color(::Type{LELBOW_LELBOW }) = :gray

tensor_marker(::Type{BELBOW }) = :rect
tensor_marker(::Type{TELBOW }) = :rect
tensor_marker(::Type{VERTEX_TVL_VERTEX }) = :rect
tensor_marker(::Type{VERTEX_TVL_RELBOW }) = :rect
tensor_marker(::Type{RELBOW_TVL_VERTEX }) = :rect
tensor_marker(::Type{RELBOW_TVL_RELBOW }) = :rect
tensor_marker(::Type{VERTEX_LELBOW }) = :rect
tensor_marker(::Type{LELBOW_VERTEX }) = :rect
tensor_marker(::Type{LELBOW_LELBOW }) = :rect

### SEGMENT TTN VISUALIZATION ###

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
function visualize(::Type{T}) where {T<:SegmentTensorType}
    ttn = _segment_ttn(T)
    visualize(ttn, _SEGMENT_TTN_POSITIONS)
end

end # module SegmentTensorTypes
