using FibTN.FibTensorTypes
using FibTN.TOBackend
using FibTN.FibTensorNetworks

@testset "FibTensorNetworks basics" begin
# TODO
end

function calculation(tt2gs::Dict{Type{<:AbstractFibTensorType}, Vector{Int}}, contractions::Vector{IC}, qubits_from_index::Dict{IndexLabel, Vector{Int}}, positions::Vector{Point2})
    # tn construction
    g2tt::Dict{Int, Type{<:AbstractFibTensorType}} = Dict(g => tt for (tt, gs) in tt2gs for g in gs)
    tn = TensorNetwork()
    for (g, tt) in g2tt add_tensor!(tn, TensorLabel(g, index_labels(tt, g))) end
    for ic in contractions add_contraction!(tn, ic) end
    # en construction and execution
    inds, data = naive_contract(tn, g2tt)
    # ql and data extraction
    ql = build_ql(qubits_from_index)
    s, a = get_states_and_amps(ql, inds, data)
    # visualization
    plot(tn, positions, g2tt)
    for (state, amp) in zip(s, a)
        plot(ql, positions, state, amp)
    end
end

@testset "FibTensorNetworks tail_triangle" begin
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
