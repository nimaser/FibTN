using FibTN.Segments
using FibTN.FibTensorNetworks
using FibTN.TensorNetworks
using FibTN.FibTensorTypes

using FibTN.TensorNetworkVisualizer
using GLMakie

const IL = IndexLabel
const IC = IndexContraction

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
    add_tensor!(tn, TensorLabel(14, index_labels(14, Reflector)))
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
    add_contraction!(tn, IC(IL(1, :b), IL(14, :a)))
    add_contraction!(tn, IC(IL(14, :b), IL(4, :bb)))
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

    push!(positions, 14=>(1, 1.5))
    push!(positions, 25=>(2, 1.5))
    push!(positions, 36=>(3, 1.5))
    push!(positions, 47=>(1, 2.5))
    push!(positions, 58=>(2, 2.5))
    push!(positions, 69=>(3, 2.5))

    groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 23, 45, 56, 78, 89, 14, 25, 36, 47, 58, 69]
    positions = [positions[g] for g in groups]
    colors = [:black for g in groups]
    markers = [g < 10 ? :rect : :vline for g in groups]
    tnds = TensorNetworkDisplaySpec(groups, positions, colors, markers)

    f = Figure()
    ax = Axis(f[1, 1])
    TensorNetworkVisualizer.visualize(tn, tnds, ax)
    DataInspector(f, range=30)
    finalize!(f, [ax])
    display(GLMakie.Screen(), f)

    ql = QubitLattice()
    add_index!(ql, IL(1, :p), [1, 2, 3])
    add_index!(ql, IL(9, :p), [4, 5, 6])

    add_index!(ql, IL(6, :tp), [6, 7, 8])
    add_index!(ql, IL(6, :bp), [9, 10, 8])

    add_index!(ql, IL(3, :tp), [9, 11, 12])
    add_index!(ql, IL(3, :bp), [13, 14, 12])

    add_index!(ql, IL(2, :tp), [15, 14, 16])
    add_index!(ql, IL(2, :bp), [16, 17, 1])

    add_index!(ql, IL(4, :tp), [18, 19, 20])
    add_index!(ql, IL(4, :mp), [20, 21, 23])
    add_index!(ql, IL(4, :bp), [3, 22, 23])

    add_index!(ql, IL(7, :tp), [24, 25, 26])
    add_index!(ql, IL(7, :mp), [26, 27, 28])
    add_index!(ql, IL(7, :bp), [18, 29, 28])

    add_index!(ql, IL(8, :tp), [4, 30, 31])
    add_index!(ql, IL(8, :mp), [31, 32, 33])
    add_index!(ql, IL(8, :bp), [34, 24, 33])

    add_index!(ql, IL(5, :tp), [34, 10, 35])
    add_index!(ql, IL(5, :mp), [35, 36, 37])
    add_index!(ql, IL(5, :bp), [15, 19, 37])

    qpos = Dict{IL, Point2}()
    push!(qpos, IL(1, :p) => positions[1])
    push!(qpos, IL(9, :p) => positions[9])

end

function grid(w::Int, h::Int)
    tn = TensorNetwork()


end
