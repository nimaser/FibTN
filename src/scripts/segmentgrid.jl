using FibTN.Segments
using FibTN.FibTensorNetworks
using FibTN.TensorNetworks
using FibTN.FibTensorTypes
using FibTN.QubitLattices

using FibTN.TOBackend

using FibTN.TensorNetworkVisualizer
using FibTN.QubitLatticeVisualizer
using GLMakie

using SparseArrayKit

const IL = IndexLabel
const IC = IndexContraction

function get_states_and_amps(ql::QubitLattice, inds::Vector{IndexLabel}, data::SparseArray)
    states, amps = Vector{Dict{Int, Int}}(), Vector{Real}()
    for (cidx, amp) in nonzero_pairs(data)
        push!(states, idxvals2qubitvals(ql, inds, [Tuple(cidx)...]))
        push!(amps, amp)
    end
    states, amps
end

function grid1x1()
    segment_mapping = Dict{NTuple{2, Int}, Segment}()
    segment_mapping[2, 1] = Segment(Tail, false, false, Tail)
    segment_mapping[1, 2] = Segment(Tail, true, true, Tail)

    index_labels(group::Int, s::Segment) = [IL(group, p) for p in segment_ports(s)]
    index_labels(group::Int, s::Type{T}) where T <: AbstractFibTensorType = [IL(group, p) for p in tensor_ports(s)]

    tn = TensorNetwork()
    add_tensor!(tn, TensorLabel(0, index_labels(0, Boundary)))
    add_tensor!(tn, TensorLabel(1, index_labels(1, Tail)))
    add_tensor!(tn, TensorLabel(2, index_labels(2, segment_mapping[2, 1])))
    add_tensor!(tn, TensorLabel(3, index_labels(3, segment_mapping[1, 2])))
    add_tensor!(tn, TensorLabel(4, index_labels(4, Tail)))

    add_tensor!(tn, TensorLabel(10, index_labels(10, Reflector)))
    add_tensor!(tn, TensorLabel(30, index_labels(30, Reflector)))
    add_tensor!(tn, TensorLabel(12, index_labels(12, Reflector)))
    add_tensor!(tn, TensorLabel(24, index_labels(24, Reflector)))
    add_tensor!(tn, TensorLabel(34, index_labels(34, Reflector)))

    add_contraction!(tn, IC(IL(1, :a), IL(12, :b)))
    add_contraction!(tn, IC(IL(12, :a), IL(2, :bb)))
    add_contraction!(tn, IC(IL(3, :ta), IL(34, :b)))
    add_contraction!(tn, IC(IL(34, :a), IL(4, :a)))

    add_contraction!(tn, IC(IL(1, :b), IL(10, :b)))
    add_contraction!(tn, IC(IL(10, :a), IL(0, :a)))
    add_contraction!(tn, IC(IL(0, :b), IL(30, :b)))
    add_contraction!(tn, IC(IL(30, :a), IL(3, :bb)))
    add_contraction!(tn, IC(IL(2, :ta), IL(24, :b)))
    add_contraction!(tn, IC(IL(24, :a), IL(4, :b)))

    positions = Dict{Int, Point2}()
    push!(positions, 1=>(1, 1))
    push!(positions, 2=>(2, 1))
    push!(positions, 3=>(1, 2))
    push!(positions, 4=>(2, 2))
    push!(positions, 12=>(1.5, 1))
    push!(positions, 34=>(1.5, 2))

    push!(positions, 10=>(1, 1.25))
    push!(positions, 0=>(1, 1.5))
    push!(positions, 30=>(1, 1.75))

    push!(positions, 24=>(2, 1.5))

    ql = QubitLattice()
    add_index!(ql, IL(1, :p), [1, 2, 3])
    add_index!(ql, IL(4, :p), [4, 5, 6])

    add_index!(ql, IL(2, :bp), [13, 14, 1])
    add_index!(ql, IL(2, :tp), [6, 12, 13])

    add_index!(ql, IL(3, :bp), [3, 11, 10])
    add_index!(ql, IL(3, :mp), [7, 9, 10])
    add_index!(ql, IL(3, :tp), [7, 8, 4])

    qpos = Dict{IL, Point2}()
    push!(qpos, IL(1, :p) => (1, 1))
    push!(qpos, IL(4, :p) => (3, 2))

    push!(qpos, IL(2, :tp) => (3, 1))
    push!(qpos, IL(2, :bp) => (2, 0))

    push!(qpos, IL(3, :tp) => (2, 3))
    push!(qpos, IL(3, :mp) => (1.5, 2.5))
    push!(qpos, IL(3, :bp) => (1, 2))

    data_map = Dict{Int, SparseArray}()
    data_map[0] = tensor_data(Boundary)
    data_map[1] = tensor_data(Tail)
    data_map[2] = segment_data(segment_mapping[2, 1])
    data_map[3] = segment_data(segment_mapping[1, 2])
    data_map[4] = tensor_data(Tail)
    for i in [12, 10, 30, 24, 34]
        data_map[i] = tensor_data(Reflector)
    end
    es = ExecutionState(tn, data_map)
    execsteps = [ContractionStep(c) for c in tn.contractions]
    for step in execsteps execute_step!(es, step) end
    et = es.tensor_from_id[only(get_ids(es))]
    states, amps = get_states_and_amps(ql, et.indices, et.data)

    groups = [0, 1, 2, 3, 4, 12, 10, 30, 24, 34]
    positions = [positions[g] for g in groups]
    colors = [:black for g in groups]
    markers = [g < 5 ? :rect : :vline for g in groups]
    markers[1] = :xcross
    tnds = TensorNetworkDisplaySpec(groups, positions, colors, markers)
    f = Figure()
    ax = Axis(f[1, 1])
    TensorNetworkVisualizer.visualize(tn, tnds, ax)
    DataInspector(f, range=30)
    finalize!(f, [ax])
    display(GLMakie.Screen(), f)

    qlds = QubitLatticeDisplaySpec(qpos, Dict(q => :black for q in get_qubits(ql)), 0.25)
    plot_all(ql, qlds, states, amps)
    for (state, amp) in zip(states, amps)
        plot(ql, qlds, state, amp)
    end
end

function grid2x2()
    segment_mapping = Dict{NTuple{2, Int}, Segment}()
    segment_mapping[1, 2] = Segment(Vertex, false, false, Tail)
    segment_mapping[1, 3] = Segment(Tail, false, false, Tail)
    segment_mapping[2, 1] = Segment(Vertex, true, true, Tail)
    segment_mapping[2, 2] = Segment(Vertex, true, true, Vertex)
    segment_mapping[2, 3] = Segment(Tail, false, false, Vertex)
    segment_mapping[3, 1] = Segment(Tail, true, true, Tail)
    segment_mapping[3, 2] = Segment(Tail, true, true, Vertex)

    index_labels(group::Int, s::Segment) = [IL(group, p) for p in segment_ports(s)]
    index_labels(group::Int, s::Type{T}) where T <: AbstractFibTensorType = [IL(group, p) for p in tensor_ports(s)]

    tn = TensorNetwork()
    # define all main tensors
    add_tensor!(tn, TensorLabel(0, index_labels(0, Boundary)))
    add_tensor!(tn, TensorLabel(1, index_labels(1, Tail)))
    add_tensor!(tn, TensorLabel(2, index_labels(2, segment_mapping[1, 2])))
    add_tensor!(tn, TensorLabel(3, index_labels(3, segment_mapping[1, 3])))
    add_tensor!(tn, TensorLabel(4, index_labels(4, segment_mapping[2, 1])))
    add_tensor!(tn, TensorLabel(5, index_labels(5, segment_mapping[2, 2])))
    add_tensor!(tn, TensorLabel(6, index_labels(6, segment_mapping[2, 3])))
    add_tensor!(tn, TensorLabel(7, index_labels(7, segment_mapping[3, 1])))
    add_tensor!(tn, TensorLabel(8, index_labels(8, segment_mapping[3, 2])))
    add_tensor!(tn, TensorLabel(9, index_labels(9, Tail)))
    # define all intervening reflectors
    add_tensor!(tn, TensorLabel(12, index_labels(12, Reflector)))
    add_tensor!(tn, TensorLabel(23, index_labels(23, Reflector)))
    add_tensor!(tn, TensorLabel(45, index_labels(45, Reflector)))
    add_tensor!(tn, TensorLabel(56, index_labels(56, Reflector)))
    add_tensor!(tn, TensorLabel(78, index_labels(78, Reflector)))
    add_tensor!(tn, TensorLabel(89, index_labels(89, Reflector)))
    add_tensor!(tn, TensorLabel(10, index_labels(10, Reflector)))
    add_tensor!(tn, TensorLabel(40, index_labels(40, Reflector)))
    add_tensor!(tn, TensorLabel(25, index_labels(25, Reflector)))
    add_tensor!(tn, TensorLabel(36, index_labels(36, Reflector)))
    add_tensor!(tn, TensorLabel(47, index_labels(47, Reflector)))
    add_tensor!(tn, TensorLabel(58, index_labels(58, Reflector)))
    add_tensor!(tn, TensorLabel(69, index_labels(69, Reflector)))
    # add all 'horizontal' contractions
    add_contraction!(tn, IC(IL(1, :a), IL(12, :b)))
    add_contraction!(tn, IC(IL(12, :a), IL(2, :bb)))
    add_contraction!(tn, IC(IL(2, :tb), IL(23, :b)))
    add_contraction!(tn, IC(IL(23, :a), IL(3, :bb)))

    add_contraction!(tn, IC(IL(4, :tb), IL(45, :b)))
    add_contraction!(tn, IC(IL(45, :a), IL(5, :bb)))
    add_contraction!(tn, IC(IL(5, :tb), IL(56, :b)))
    add_contraction!(tn, IC(IL(56, :a), IL(6, :bb)))

    add_contraction!(tn, IC(IL(7, :ta), IL(78, :b)))
    add_contraction!(tn, IC(IL(78, :a), IL(8, :bb)))
    add_contraction!(tn, IC(IL(8, :ta), IL(89, :b)))
    add_contraction!(tn, IC(IL(89, :a), IL(9, :a)))
    # add all 'vertical' contractions
    add_contraction!(tn, IC(IL(1, :b), IL(10, :a)))
    add_contraction!(tn, IC(IL(10, :b), IL(0, :a)))
    add_contraction!(tn, IC(IL(0, :b), IL(40, :a)))
    add_contraction!(tn, IC(IL(40, :b), IL(4, :bb)))
    add_contraction!(tn, IC(IL(4, :ta), IL(47, :a)))
    add_contraction!(tn, IC(IL(47, :b), IL(7, :bb)))

    add_contraction!(tn, IC(IL(2, :ta), IL(25, :a)))
    add_contraction!(tn, IC(IL(25, :b), IL(5, :ba)))
    add_contraction!(tn, IC(IL(5, :ta), IL(58, :a)))
    add_contraction!(tn, IC(IL(58, :b), IL(8, :ba)))

    add_contraction!(tn, IC(IL(3, :ta), IL(36, :a)))
    add_contraction!(tn, IC(IL(36, :b), IL(6, :ba)))
    add_contraction!(tn, IC(IL(6, :ta), IL(69, :a)))
    add_contraction!(tn, IC(IL(69, :b), IL(9, :b)))

    positions = Dict{Int, Point2}()
    push!(positions, 1=>(1, 1))
    push!(positions, 2=>(2, 1))
    push!(positions, 3=>(3, 1))
    push!(positions, 4=>(1, 2))
    push!(positions, 5=>(2, 2))
    push!(positions, 6=>(3, 2))
    push!(positions, 7=>(1, 3))
    push!(positions, 8=>(2, 3))
    push!(positions, 9=>(3, 3))

    push!(positions, 12=>(1.5, 1))
    push!(positions, 23=>(2.5, 1))
    push!(positions, 45=>(1.5, 2))
    push!(positions, 56=>(2.5, 2))
    push!(positions, 78=>(1.5, 3))
    push!(positions, 89=>(2.5, 3))

    push!(positions, 10=>(1, 1.25))
    push!(positions, 0=>(1, 1.5))
    push!(positions, 40=>(1, 1.75))
    push!(positions, 25=>(2, 1.5))
    push!(positions, 36=>(3, 1.5))
    push!(positions, 47=>(1, 2.5))
    push!(positions, 58=>(2, 2.5))
    push!(positions, 69=>(3, 2.5))

    groups = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 23, 45, 56, 78, 89, 10, 40, 25, 36, 47, 58, 69]
    positions = [positions[g] for g in groups]
    colors = [:black for g in groups]
    markers = [g < 10 ? :rect : :vline for g in groups]
    markers[1] = :xcross
    tnds = TensorNetworkDisplaySpec(groups, positions, colors, markers)

    f = Figure()
    ax = Axis(f[1, 1])
    TensorNetworkVisualizer.visualize(tn, tnds, ax)
    DataInspector(f, range=30)
    finalize!(f, [ax])
    display(GLMakie.Screen(), f)

    ql = QubitLattice()
    add_index!(ql, IL(1, :p), [1, 2, 3])
    add_index!(ql, IL(4, :bp), [3, 4, 5])
    add_index!(ql, IL(4, :mp), [7, 6, 5])
    add_index!(ql, IL(4, :tp), [8, 35, 7])
    add_index!(ql, IL(7, :bp), [8, 9, 10])
    add_index!(ql, IL(7, :mp), [12, 11, 10])
    add_index!(ql, IL(7, :tp), [12, 13, 14])
    add_index!(ql, IL(8, :bp), [30, 14, 15])
    add_index!(ql, IL(8, :mp), [17, 16, 15])
    add_index!(ql, IL(8, :tp), [17, 18, 19])
    add_index!(ql, IL(9, :p), [19, 20, 21])
    add_index!(ql, IL(6, :tp), [21, 22, 23])
    add_index!(ql, IL(6, :bp), [24, 31, 23])
    add_index!(ql, IL(3, :tp), [24, 25, 26])
    add_index!(ql, IL(3, :bp), [26, 27, 28])
    add_index!(ql, IL(2, :tp), [34, 28, 29])
    add_index!(ql, IL(2, :bp), [29, 36, 1])
    add_index!(ql, IL(5, :tp), [30, 31, 32])
    add_index!(ql, IL(5, :mp), [32, 33, 37])
    add_index!(ql, IL(5, :bp), [34, 35, 37])

    qpos = Dict{IL, Point2}()
    push!(qpos, IL(1, :p) => (1, 1))
    push!(qpos, IL(9, :p) => (6, 4))
    push!(qpos, IL(2, :bp) => (2, 0))
    push!(qpos, IL(2, :tp) => (3, 1))
    push!(qpos, IL(3, :bp) => (4, 0))
    push!(qpos, IL(3, :tp) => (5, 1))

    push!(qpos, IL(4, :bp) => (1, 2))
    push!(qpos, IL(4, :mp) => (1.5, 2.5))
    push!(qpos, IL(4, :tp) => (2, 3))
    push!(qpos, IL(5, :bp) => (3, 2))
    push!(qpos, IL(5, :mp) => (3.5, 2.5))
    push!(qpos, IL(5, :tp) => (4, 3))
    push!(qpos, IL(6, :bp) => (5, 2))
    push!(qpos, IL(6, :tp) => (6, 3))

    push!(qpos, IL(7, :bp) => (2, 4))
    push!(qpos, IL(7, :mp) => (2.5, 4.5))
    push!(qpos, IL(7, :tp) => (3, 5))
    push!(qpos, IL(8, :bp) => (4, 4))
    push!(qpos, IL(8, :mp) => (4.5, 4.5))
    push!(qpos, IL(8, :tp) => (5, 5))

    data_map = Dict{Int, SparseArray}()
    data_map[0] = tensor_data(Boundary)
    data_map[1] = tensor_data(Tail)
    data_map[2] = segment_data(segment_mapping[1, 2])
    data_map[3] = segment_data(segment_mapping[1, 3])
    data_map[4] = segment_data(segment_mapping[2, 1])
    data_map[5] = segment_data(segment_mapping[2, 2])
    data_map[6] = segment_data(segment_mapping[2, 3])
    data_map[7] = segment_data(segment_mapping[3, 1])
    data_map[8] = segment_data(segment_mapping[3, 2])
    data_map[9] = tensor_data(Tail)
    for i in [12, 23, 45, 56, 78, 89, 10, 40, 25, 36, 47, 58, 69]
        data_map[i] = tensor_data(Reflector)
    end
    es = ExecutionState(tn, data_map)
    execsteps = [ContractionStep(c) for c in tn.contractions]
    for step in execsteps execute_step!(es, step) end
    et = es.tensor_from_id[only(get_ids(es))]
    states, amps = get_states_and_amps(ql, et.indices, et.data)

    qcolors = Dict(q => :black for q in get_qubits(ql))
    qlds = QubitLatticeDisplaySpec(qpos, qcolors, 0.25)
    plot_all(ql, qlds, states, amps)
end

function grid(w::Int, h::Int)
    tn = TensorNetwork()


end
