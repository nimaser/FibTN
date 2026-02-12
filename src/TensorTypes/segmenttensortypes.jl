module SegmentTensorTypes

using ..FibTensorTypes
using ..TensorNetworks
import ..TensorNetworks: tensor_ports, tensor_data, tensor_color, tensor_marker
using ..TensorNetworks.TOBackend

export SegmentTensorType
export BELBOW, TELBOW, VERTEX_TVL_VERTEX
export VERTEX_TVL_RELBOW, VERTEX_LELBOW
export RELBOW_TVL_VERTEX, LELBOW_VERTEX
export RELBOW_TVL_RELBOW, LELBOW_LELBOW

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

### TENSOR PORTS ###

tensor_ports(::Type{BELBOW}) = (:D, :L, :bp)
tensor_ports(::Type{TELBOW}) = (:U, :R, :tp)
tensor_ports(::Type{VERTEX_TVL_VERTEX}) = (:U, :R, :D, :L, :tp, :mp, :bp)

tensor_ports(::Type{VERTEX_TVL_RELBOW}) = (:U, :R, :D, :tp, :mp, :bp)
tensor_ports(::Type{RELBOW_TVL_VERTEX}) = (:R, :D, :L, :tp, :mp, :bp)
tensor_ports(::Type{RELBOW_TVL_RELBOW}) = (:R, :D, :tp, :mp, :bp)

tensor_ports(::Type{VERTEX_LELBOW}) = (:U, :R, :L, :tp, :bp)
tensor_ports(::Type{LELBOW_VERTEX}) = (:U, :D, :L, :tp, :bp)
tensor_ports(::Type{LELBOW_LELBOW}) = (:U, :L, :tp, :bp)

### TENSOR DATA ###

const _cache = IdDict{DataType,Any}()

function tensor_data(::Type{T}) where {T<:SegmentTensorType}
    # try to get the data from the cache, else generate it
    get!(_cache, T) do
        _generate_tensor_data(T)
    end
end

"""
For nontrivial segments (composed of more than one tensor), obtains
the list of tensors that makes it up, generates a `TypedTensorNetwork`
using them, adds `REFLECTOR` tensors to the top and right indices to
allow easy tiling, contracts the tensor network, then permutes the
data so that it aligns with the ports outputted by `tensor_ports`.
"""
function _generate_tensor_data(::Type{T}) where {T<:SegmentTensorType}
    # this case is just an alias with different port names
    if T == BELBOW return tensor_data(ELBOW_T) end
    # alias ELBOW_T, but also add the boundary trivializer and reflectors
    if T == TELBOW
        ttn = TypedTensorNetwork()
        add_tensor!(ttn, 1, ELBOW_T)
        add_tensor!(ttn, 11, REFLECTOR)
        add_tensor!(ttn, 10, BOUNDARY)
        add_tensor!(ttn, 9, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(1, :b), IL(11, :a)))
        add_contraction!(ttn.tn, IC(IL(11, :b), IL(10, :b)))
        add_contraction!(ttn.tn, IC(IL(10, :a), IL(9, :a)))
    else
        ttn = _make_segment_ttn(_segment_tensors(T))
    end
    # add reflection tensors to the top and right, if those indices exist
    if has_index(ttn.tn, IL(1, :a)) && !has_contraction(ttn.tn, IL(1, :a))
        add_tensor!(ttn, 8, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(1, :a), IL(8, :b)))
    end
    if has_index(ttn.tn, IL(1, :b)) && !has_contraction(ttn.tn, IL(1, :b))
        add_tensor!(ttn, 9, REFLECTOR)
        add_contraction!(ttn.tn, IC(IL(1, :b), IL(9, :a)))
    end
    # contract using execution state
    es = ExecutionState(ttn)
    execsteps = [ContractionStep(c) for c in get_contractions(ttn.tn)]
    for execstep in execsteps execute_step!(es, execstep) end
    # mapping establishing which indices on the segment (seen as a ttn)
    # correspond to which ports on the segment (seen as a tensor)
    idx_map = Dict{Symbol,IL}()
    push!(idx_map, :U => IL(8, :a))
    push!(idx_map, :R => IL(9, :b))
    push!(idx_map, :D => IL(7, :a))
    push!(idx_map, :L => IL(7, :b))
    push!(idx_map, :tp => IL(1, :p))
    push!(idx_map, :mp => IL(3, :p))
    push!(idx_map, :bp => IL(7, :p))
    # permute indices to match ports
    id = only(get_ids(es))
    indordering = [idx_map[p] for p in tensor_ports(T)]
    execute_step!(es, PermuteIndicesStep(id, indordering))
    et = es.tensor_from_id[id]
    et.data
end

_segment_tensors(::Type{VERTEX_TVL_VERTEX}) =
    (VERTEX, REFLECTOR, TAIL, REFLECTOR, VACUUMLOOP, REFLECTOR, VERTEX)

_segment_tensors(::Type{VERTEX_TVL_RELBOW}) =
    (VERTEX, REFLECTOR, TAIL, REFLECTOR, VACUUMLOOP, REFLECTOR, TAIL)

_segment_tensors(::Type{RELBOW_TVL_VERTEX}) =
    (T_ELBOW, REFLECTOR, TAIL, REFLECTOR, VACUUMLOOP, REFLECTOR, VERTEX)

_segment_tensors(::Type{RELBOW_TVL_RELBOW}) =
    (T_ELBOW, REFLECTOR, TAIL, REFLECTOR, VACUUMLOOP, REFLECTOR, TAIL)

_segment_tensors(::Type{VERTEX_LELBOW}) =
    (VERTEX, REFLECTOR, T_ELBOW)

_segment_tensors(::Type{LELBOW_VERTEX}) =
    (TAIL, REFLECTOR, VERTEX)

_segment_tensors(::Type{LELBOW_LELBOW}) =
    (TAIL, REFLECTOR, T_ELBOW)

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
    add_contraction!(ttn.tn, IC(IL(1, :c), IL(2, :a)))
    add_contraction!(ttn.tn, IC(IL(2, :b), IL(3, :a)))
    add_contraction!(ttn.tn, IC(IL(3, :c), IL(4, :a)))
    add_contraction!(ttn.tn, IC(IL(4, :b), IL(5, :a)))
    add_contraction!(ttn.tn, IC(IL(5, :b), IL(6, :a)))
    add_contraction!(ttn.tn, IC(IL(6, :b), IL(7, :c)))
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
    add_contraction!(ttn.tn, IC(IL(1, :c), IL(2, :a)))
    add_contraction!(ttn.tn, IC(IL(2, :b), IL(7, :c)))
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

end # module SegmentTensorTypes
