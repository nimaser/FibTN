
@testset "FibTensorNetworks tail_triangle" begin
    # construct network
    ftn = FibTensorNetwork()
    add_tensor!(ftn, 1, Tail)
    add_tensor!(ftn, 3, Tail)
    add_tensor!(ftn, 5, Tail)
    add_tensor!(ftn, 2, Reflector)
    add_tensor!(ftn, 4, Reflector)
    add_tensor!(ftn, 6, Reflector)
    add_contraction!(ftn.tn, IC(IL(1, :b), IL(2, :a)))
    add_contraction!(ftn.tn, IC(IL(2, :b), IL(3, :a)))
    add_contraction!(ftn.tn, IC(IL(3, :b), IL(4, :a)))
    add_contraction!(ftn.tn, IC(IL(4, :b), IL(5, :a)))
    add_contraction!(ftn.tn, IC(IL(5, :b), IL(6, :a)))
    add_contraction!(ftn.tn, IC(IL(6, :b), IL(1, :a)))
    # contract network
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
    tt2gs::Dict{Type{<:AbstractFibTensorType}, Vector{Int}} = Dict(
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
    tt2gs::Dict{Type{<:AbstractFibTensorType}, Vector{Int}} = Dict(
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
    tt2gs::Dict{Type{<:AbstractFibTensorType}, Vector{Int}} = Dict(
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
