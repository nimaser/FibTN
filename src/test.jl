using FibTN.FibTensorTypes
using FibTN.IntegrationUtils

function tail_triangle()
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
    contractions = contractionchain(1, 6, :b, :a)
    push!(contractions, (6, :b) => (1, :a))
    
    g2tt = make_g2tt(tt2gs)
    tn = build_tn(g2tt, contractions)
    inds, data = dumb_contract_tn(tn, g2tt)
    plot(tn, positions, g2tt)
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
                   Reflector        => [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21],
                   VacuumLoop       => [],
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
    contractions = contractionchain(1, 20, :b, :a)
    push!(contractions, (20, :b) => (1, :a))
    push!(contractions, (1, :c) => (21, :a))
    push!(contractions, (21, :b) => (1, :c))
    
    g2tt = make_g2tt(tt2gs)
    tn = build_tn(g2tt, contractions)
    inds, data = dumb_contract_tn(tn, g2tt)
    plot(tn, positions, g2tt)
end
