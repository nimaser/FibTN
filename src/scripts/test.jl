const IC = IndexContraction
const IL = IndexLabel
using FibTN.TensorNetworks
using FibTN.FibTensorTypes
using FibTN.QubitLattices

function tail_triangle()
    # tensors
    tt2gs::Dict{Type{<:AbstractFibTensorType}, Vector{Int}} = Dict(
                 Reflector      => [2, 4, 6],
                 Tail           => [1, 3, 5],
                )
    # contractions
    contractions = contractionchain(1, 6, :b, :a)
    push!(contractions, IC(IL(6, :b), IL(1, :a)))
    # qubits
    qubits_from_index = Dict(
                             IL(1, :p) => [3, 4, 1],
                             IL(3, :p) => [1, 5, 2],
                             IL(5, :p) => [2, 6, 3],
                            )
    # positions
    positions = insert_midpoints(triangle(duplicatefirst=true))
    pop!(positions)
    # calculate result and display
    calculation(tt2gs, contractions, qubits_from_index, positions)
end

function tail_triangle_trivial_boundary()
    # tensors
    tt2gs::Dict{Type{<:AbstractFibTensorType}, Vector{Int}}  = Dict(
                 Reflector      => [2, 4, 6, 8],
                 Boundary       => [7],
                 Tail           => [1, 3, 5],
                )
    # contractions
    contractions = contractionchain(1, 8, :b, :a)
    push!(contractions, IC(IL(8, :b), IL(1, :a)))
    # qubits
    qubits_from_index = Dict(
                             IL(1, :p) => [3, 4, 1],
                             IL(3, :p) => [1, 5, 2],
                             IL(5, :p) => [2, 6, 3],
                            )
    # positions
    positions = insert_midpoints(triangle(duplicatefirst=true); counts=[1, 1, 3])
    pop!(positions)
    # calculate result and display
    calculation(tt2gs, contractions, qubits_from_index, positions)
end

function tail_triangle_trivial_boundary_vacuum_loop()
    # tensors
    tt2gs::Dict{Type{<:AbstractFibTensorType}, Vector{Int}}  = Dict(
                 Reflector      => [2, 4, 6, 8, 10],
                 Boundary       => [7],
                 VacuumLoop     => [9],
                 Tail           => [1, 3, 5],
                )
    # contractions
    contractions = contractionchain(1, 10, :b, :a)
    push!(contractions, IC(IL(10, :b), IL(1, :a)))
    # qubits
    qubits_from_index = Dict(
                             IL(1, :p) => [3, 4, 1],
                             IL(3, :p) => [1, 5, 2],
                             IL(5, :p) => [2, 6, 3],
                            )
    # positions
    positions = insert_midpoints(triangle(duplicatefirst=true); counts=[1, 1, 5])
    pop!(positions)
    # calculate result and display
    calculation(tt2gs, contractions, qubits_from_index, positions)
end

function two_triangles()
    # tensors
    tt2gs = Dict(
                 Reflector      => [2, 4, 6, 8, 10, 12, 14, 15],
                 Boundary       => [13],
                 VacuumLoop     => [3, 7],
                 Tail           => [1, 9],
                 Vertex         => [5, 11],
                )
    # contractions
    contractions = contractionchain(1, 14, :b, :a)
    push!(contractions, IC(IL(14, :b), IL(1, :a)))
    push!(contractions, IC(IL(11, :c), IL(15, :a)))
    push!(contractions, IC(IL(15, :b), IL(5, :c)))
    
    # qubits
    qubits_from_index = Dict(
                             IL(1, :p) => [1, 2, 3],
                             IL(5, :p) => [3, 4, 7],
                             IL(9, :p) => [4, 5, 6],
                             IL(11, :p) => [6, 1, 7],
                            )
    # positions
    positions = insert_midpoints(square(duplicatefirst=true); counts=[3, 3, 1, 3])
    pop!(positions)
    push!(positions, insert_midpoints([positions[5], positions[11]]; counts=[1])[2])
    
    # calculate result and display
    calculation(tt2gs, contractions, qubits_from_index, positions)
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
