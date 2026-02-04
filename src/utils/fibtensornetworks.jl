using FibTN.TensorNetworks
using FibTN.FibTensorTypes

const IC = IndexContraction
const IL = IndexLabel

const GridPosition = NTuple{2, Int}

struct Segment
    start::Int
    finish::Int
    hasamp::Bool
    x::Int
    y::Int
    pos::GridPosition
    function Segment(start::Int, x::Int, y::Int; hasamp::Bool=true)
        finish = start + (hasamp ? 6 : 4)
        new(start, finish, hasamp, x, y, (x, y))
    end
end

get_groups(s::Segment) = s.start:s.finish
num_groups(s::Segment) = s.finish - s.start + 1

get_group(s::Segment, ::Type{Tail}) = s.start + 2
get_group(s::Segment, ::Type{VacuumLoop}) =
    s.hasamp ? s.start + 4 : throw(ArgumentError("provided segment has no vacuum loop tensor"))

struct FibTensorNetwork
    tn::TensorNetwork
    tensortype_from_group::Dict{Int, Type{<:AbstractFibTensorType}}
    segments::Vector{Segment}
    segment_from_position::Dict{GridPosition, Segment}
    FibTensorNetwork() = new(TensorNetwork(), Dict())
end

function get_segment(ftn::FibTensorNetwork, group::Int)
    for segment in ftn.segments
        if segment.start <= group <= segment.finish return segment end
    end
    throw(KeyError(group))
end

function _add_tensor!(ftn::FibTensorNetwork, group::Int, ::Type{T}) where T <: AbstractFibTensorType
    index_labels = [IL(group, p) for p in tensor_ports(T)]
    tensortype_from_group[group] = T
    add_tensor!(ftn.tn, TensorLabel(group, index_labels))
end

function add_segment!(ftn::FibTensorNetwork, start::Int, x::Int, y::Int; hasamp::Bool = true)
    # add vertices at top and bottom
    finish = start + (amp ? 6 : 4)
    _add_tensor!(ftn, start, Vertex)
    _add_tensor!(ftn, finish, Vertex)
    # add tensors in the middle
    _add_tensor!(ftn, start + 1, Reflector)
    _add_tensor!(ftn, start + 2, Tail)
    _add_tensor!(ftn, start + 3, Reflector)
    # add amplitude tensor
    if hasamp
        _add_tensor!(ftn, start + 4, VacuumLoop)
        _add_tensor!(ftn, start + 5, Reflector)
    end
    # add contractions from top to middle
    add_contraction!(ftn.tn, IC(IL(start, :c), IL(start+1, :a)))
    add_contraction!(ftn.tn, IC(IL(start+1, :b), IL(start+2, :a)))
    add_contraction!(ftn.tn, IC(IL(start+2, :b), IL(start+3, :a)))
    # add contractions for amplitude tensor
    if hasamp
        add_contraction!(ftn.tn, IC(IL(start+3, :b), IL(start+4, :a)))
        add_contraction!(ftn.tn, IC(IL(start+4, :b), IL(start+5, :a)))
    end
    # add final contraction from middle to bottom
    add_contraction!(ftn.tn, IC(IL(finish-1, :b), IL(finish, :c)))
    # create segment and add to ftn
    s = Segment(start, x, y; hasamp=hasamp)
    push!(ftn.segments, s)
    ftn.segment_from_pos[(x, y)] = s
end

add_segment!(ftn::FibTensorNetwork, start::Int, pos::GridPosition; hasamp::Bool = true) =
    add_segment!(ftn, start, pos[1], pos[2]; hasamp=hasamp)
    
function connect_segments!(ftn::FibTensorNetwork, pos1::GridPosition, pos2::GridPosition, group::Int; enable_adjacency_check::Bool=true, dir::Symbol=:x)
    if enable_adjacency_check
        # check that positions are either directly above or below
        xdiff = pos1[1] - pos2[1]
        ydiff = pos1[2] - pos2[2]
        xdiff == 0 || ydiff == 0 || throw(ArgumentError("positions are diagonal"))
        if xdiff < 0 pos1, pos2, xdiff, dir = pos2, pos1, -xdiff :h end
        if ydiff < 0 pos1, pos2, ydiff, dir = pos2, pos1, -ydiff :v end
        xdiff == 1 || ydiff == 1 || throw(ArgumentError("positions aren't directly adjacent"))
    else
        dir == :h || dir == :v || throw(ArgumentError("dir must be specified if adjacency check disabled"))
    end
    # fetch segments
    s1 = ftn.segment_from_position[pos1]
    s2 = ftn.segment_from_position[pos2]
    # add reflector
    _add_tensor!(ftn, group, Reflector)
    # add contractions depending on relative segment orientations
    if dir == :v
        add_contraction!(ftn.tn, IC(IL(s1.finish, :a), IL(group, :a)))
        add_contraction!(ftn.tn, IC(IL(group, :b), IL(s2.start, :a)))
    else
        add_contraction!(ftn.tn, IC(IL(s1.start, :b), IL(group, :a)))
        add_contraction!(ftn.tn, IC(IL(group, :b), IL(s2.finish, :b)))
    end
end

function grid(w::Int, h::Int)
    ftn = FibTensorNetwork()
    new_group = 1
    # make all of the segments
    for i in 1:w
        for j in 1:h
            add_segment!(ftn, new_group, w, h)
            new_group = ftn.segment_from_position[(w, h)].finish + 1
        end
    end
    # connect internal rows
    for i in 1:w-1
        for j in 1:h
            connect_segments!(ftn, (i, j), (i+1, j), new_group)
            new_group += 1
        end
    end
    # connect internal columns
    for j in 1:h-1
        for i in 1:w
            connect_segments!(ftn, (i, j), (i, j+1), new_group)
            new_group += 1
        end
    end
    # connect row wraparound
    for j in 1:h
        connect_segments!(ftn, (w, j), (1, j), new_group; enable_adjacency_check=false, dir=:h)
        new_group += 1
    end
    # connect col wraparound
    for i in 1:w
        connect_segments!(ftn, (i, h), (i, 1), new_group; enable_adjacency_check=false, dir=:v)
        new_group += 1
    end
    # make qubit lattice
    # qubits are assigned by rows, then cols, then internals
    function get_qubit_mapping(il::IndexLabel)
        s = get_segment(ftn, il.group)
        i, j = s.pos
        if s.start == il.group
            q1 = w*h + (j-1)*w + i
            q2 = (j-1)*w + i
            q3 = 2*w*h + 3*((j-1)*w + i-1) + 1
            return [q1, q2, q3]
        elseif get_group(s, Tail) == il.group
            q1 = 2*w*h + 3*((j-1)*w + i-1) + 1
            q2 = 2*w*h + 3*((j-1)*w + i-1) + 2
            q3 = 2*w*h + 3*((j-1)*w + i-1) + 3
            return [q1, q2, q3]
        elseif s.finish == il.group
            q1 = w*h + ((j == h ? 1 : j+1)-1)*w + i
            q2 = (j-1)*w + (i == 1 ? w : i-1)
            q3 = 2*w*h + 3*((j-1)*w + i-1) + 3
            return [q1, q2, q3]
        else
            error("IndexLabel provided doesn't belong to a physical index")
        end
    end
    ql = QubitLattice()
    for il in TensorNetworks.get_indices(ftn.tn)
        if il.port == :p add_index!(ql, get_qubit_mapping(il)) end
    end
    ftn, ql
end
