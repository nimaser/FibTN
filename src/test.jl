using FibTN.Indices
const IP = IndexPair
const IL = IndexLabel
using FibTN.TensorNetworks
using FibTN.FibTensorTypes
using FibTN.IntegrationUtils
using FibTN.QubitLattices
using FibTN.Visualizer

using GLMakie

include("integrationutils.jl")

function tail_triangle()
    # tensors
    tt2gs = Dict(
                 Reflector        => [2, 4, 6],
                 Tail             => [1, 3, 5],
                )
    positions = insert_midpoints(triangle())
    # contractions
    contractions = contractionchain(1, 6, :b, :a)
    push!(contractions, IP(IL(6, :b), IL(1, :a)))
    # qubits
    qubits_from_index = Dict(
                             IL(1, :p) => [3, 4, 1],
                             IL(3, :p) => [1, 5, 2],
                             IL(5, :p) => [2, 6, 3],
                            )
    # tn construction
    g2tt = make_g2tt(tt2gs)
    tn = build_tn(g2tt, contractions)
    # en construction and execution
    inds, data = naive_contract_tn(tn, g2tt)
    inds2, data2 = in_edge_first_contract(tn, g2tt)
    @show inds, inds2
    @show data, data2
    # ql and data extraction
    ql = build_ql(qubits_from_index)
    s, a = get_states_and_amps(ql, inds, data)
    # visualization
    plot(tn, positions, g2tt)
    for state in s
        plot(ql, positions, state)
    end
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
