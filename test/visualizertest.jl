using FibTN.Indices
using FibTN.TensorNetworks
using FibTN.Visualizer
using FibTN.FibTensorTypes

using GLMakie

@testset "visualize TensorNetwork" begin
    # build network
    tn = TensorNetwork()
    add_tensor!(tn, TensorLabel(1, index_labels(Tail, 1)))
    add_tensor!(tn, TensorLabel(2, index_labels(Tail, 2)))
    add_tensor!(tn, TensorLabel(3, index_labels(Tail, 3)))
    add_tensor!(tn, TensorLabel(4, index_labels(LoopAmplitude, 4)))
    add_contraction!(tn, IndexPair(IndexLabel(1, :b), IndexLabel(4, :a)))
    add_contraction!(tn, IndexPair(IndexLabel(4, :b), IndexLabel(2, :a)))
    add_contraction!(tn, IndexPair(IndexLabel(2, :b), IndexLabel(3, :a)))
    add_contraction!(tn, IndexPair(IndexLabel(3, :b), IndexLabel(1, :a)))
    
    # build displayspec
    positions = [(0, √3), (1, 0), (-1, 0), (1/2, (√3)/2)]
    colors = [:black, :black, :black, :blue]
    markers = [:circle, :circle, :circle, :circle]
    tnds = TensorNetworkDisplaySpec(1:4, positions, colors, markers)
    
    # create figure and visualize
    f = Figure()
    ax = Axis(f[1, 1])
    hidedecorations!(ax)
    hidespines!(ax)
    visualize(tn, tnds, ax)
    display(f)
end

@testset "visualize QubitLattice" begin
    # build qubit lattice
    ql = QubitLattice()
    add_index!(ql, IndexLabel(1, :q), [1, 3, 4])
end
