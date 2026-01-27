using FibTN
using FibTN.Indices
using FibTN.TensorNetworks
using FibTN.Executor
using FibTN.QubitLattices
using FibTN.Visualizer

using FibTN.FibTensorTypes

using GLMakie
               
function evaluate_lattice(tt2gs, positions, contractions)
    # convenience data transformation
    g2tt = Dict(g => tt for (tt, gs) in tt2gs for g in gs)

    # build network
    tn = TensorNetwork()
    for (g, tt) in g2tt
        add_tensor!(tn, TensorLabel(g, index_labels(tt, g)))
    end
    for (i1, i2) in contractions
        add_contraction!(tn, IndexPair(IndexLabel(i1...), IndexLabel(i2...)))
    end

    # visualize network
    tt2color(::Type{Reflector}) = :gray
    tt2color(::Type{LoopAmplitude}) = :orange
    tt2color(::Type{Vertex}) = :red
    tt2color(::Type{Tail}) = :blue
    tt2color(::Type{Crossing}) = :green
    tt2color(::Type{Fusion}) = :teal

    groups = sort(collect(keys(g2tt)))
    colors = [tt2color(g2tt[g]) for g in groups]
    markers = [:circle for g in groups]
    tnds = TensorNetworkDisplaySpec(groups, positions, colors, markers)
        
    f = Figure()
    ax = Axis(f[1, 1])
    hidedecorations!(ax)
    hidespines!(ax)
    visualize(tn, tnds, ax)
    display(f)

    # create execution network and contract
    en = ExecNetwork(tn, Dict(g => tensor_data(tt) for (g, tt) in g2tt))
    for c in tn.contractions
        execute_step!(en, Contraction(c))
    end
    et = execute_step!(en, FetchResult())
    inds, data = et.indices, et.data
    
    # visualize results
    # TODO
end

function tail_triangle()
    # define network
    tt2gs = Dict(
                   Tail             => [1, 3, 5],
                   Reflector        => [2, 4, 6],
                  )
    positions = [
        (-1, 0),
        (-1/2, √3/2),
        (0, √3),
        (1/2, √3/2),
        (1, 0),
        (0, 0),
    ]
    contractions = [
                    (1, :b) => (2, :a),
                    (2, :b) => (3, :a),
                    (3, :b) => (4, :a),
                    (4, :b) => (5, :a),
                    (5, :b) => (6, :a),
                    (6, :b) => (1, :a),
                   ]
    evaluate_lattice(tt2gs, positions, contractions)
end

function tail_square()
    # define network
    tt2gs = Dict(
                   Tail             => [1, 3, 5, 7],
                   Reflector        => [2, 4, 6, 8],
                  )
    positions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    contractions = [
                    (1, :b) => (2, :a),
                    (2, :b) => (3, :a),
                    (3, :b) => (4, :a),
                    (4, :b) => (5, :a),
                    (5, :b) => (6, :a),
                    (6, :b) => (7, :a),
                    (7, :b) => (8, :a),
                    (8, :b) => (1, :a),
                   ]
    evaluate_lattice(tt2gs, positions, contractions)
end

function two_hexagons()
    # define network
    tt2gs = Dict(
                   Tail             => [3, 5, 7, 9, 13, 15, 17, 19],
                   LoopAmplitude    => [],
                   Reflector        => [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21],
                   Vertex           => [1, 11],
                  )
    positions = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 2),
        (4, 1),
        (4, 0),
        (4, -1),
        (3, -2),
        (2, -3),
        (1, -2),
        (0, -1),
        (-1, -2),
        (-2, -3),
        (-3, -2),
        (-4, -1),
        (-4, 0),
        (-4, 1),
        (-3, 2),
        (-2, 3),
        (-1, 2),
        (0, 0),
    ]
    contractions = []
    contractions = [
                    (1, :b) => (5, :a),
                    (5, :b) => (2, :a),
                    (2, :b) => (6, :a),
                    (6, :b) => (3, :a),
                    (3, :b) => (7, :a),
                    (7, :b) => (4, :a),
                    (4, :b) => (8, :a),
                    (8, :b) => (1, :a),
                   ]
    evaluate_lattice(tt2gs, positions, contractions)
end
