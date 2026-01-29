using FibTN.Indices
using FibTN.TensorNetworks
using FibTN.FibTensorTypes
using FibTN.IntegrationUtils
using FibTN.QubitLattices
using FibTN.Visualizer

using GLMakie

function tail_triangle()
    # tensors
    tt2gs = Dict(
                 Reflector        => [2, 4, 6],
                 Tail             => [1, 3, 5],
                )
    positions = [
        (-1, 0),
        (-1/2, √3/2),
        (0, √3),
        (1/2, √3/2),
        (1, 0),
        (0, 0),
    ]
    # contractions
    contractions = contractionchain(1, 6, :b, :a)
    push!(contractions, (6, :b) => (1, :a))
    # qubits
    qubits = Dict(
                  IndexLabel(1, :p), [3, 4, 1],
                  IndexLabel(3, :p), [1, 5, 2],
                  IndexLabel(5, :p), [2, 6, 3],
                 )
    g2tt = make_g2tt(tt2gs)
    tn = build_tn(g2tt, contractions)
    inds, data = dumb_contract_tn(tn, g2tt)
    IntegrationUtils.plot(tn, positions, g2tt)
    
    ql = QubitLattice()
    add_index!(ql, IndexLabel(1, :p), [3, 4, 1])
    add_index!(ql, IndexLabel(3, :p), [1, 5, 2])
    add_index!(ql, IndexLabel(5, :p), [2, 6, 3])
    s, a = get_states_and_amps(ql, inds, data)

    pinds = filter(idx -> idx.port == :p, collect(indices(tn)))
    pind_positions = Dict(pind => positions[pind.group] for pind in pinds)
    qubit_colors = Dict(q => v == 1 ? :red : :black for (q, v) in s[1])
    qlds = QubitLatticeDisplaySpec(pind_positions, qubit_colors, 0.5)

    f = Figure()
    ax = Axis(f[1, 1])
    hidedecorations!(ax)
    hidespines!(ax)
    visualize(ql, qlds, ax)
    DataInspector(f, range=30)
    display(f)
end

function tail_square()
    tt2gs = Dict(
                 Reflector        => [2, 4, 6, 8, 10],
                 Boundary         => [9],
                 Tail             => [1, 3, 5, 7],
                )
    positions = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (1/2, -1),
        (0, -1),
        (-1/2, -1),
    ]
    contractions = contractionchain(1, 10, :b, :a)
    push!(contractions, (10, :b) => (1, :a))
    
    g2tt = make_g2tt(tt2gs)
    tn = build_tn(g2tt, contractions)
    inds, data = dumb_contract_tn(tn, g2tt)
    plot(tn, positions, g2tt)
end

function two_hexagons()
    tt2gs = Dict(
                 Reflector        => [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 25, 27],
                 Boundary         => [21],
                 VacuumLoop       => [24, 26],
                 Tail             => [3, 5, 7, 9, 13, 15, 17, 19],
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
    contractions = contractionchain(1, 22, :b, :a)
    push!(contractions, (22, :b) => (1, :a))
    push!(contractions, (1, :c) => (23, :a))
    append!(contractions, contractionchain(23, 25, :b, :a))
    push!(contractions, (25, :b) => (26, :b), (26, :a) => (27, :a))
    push!(contractions, (27, :b) => (11, :c))
    
    g2tt = make_g2tt(tt2gs)
    tn = build_tn(g2tt, contractions)
    inds, data = dumb_contract_tn(tn, g2tt)
    plot(tn, positions, g2tt)
end
