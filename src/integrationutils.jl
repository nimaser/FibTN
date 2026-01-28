module IntegrationUtils

using ..FibTN.Indices
using ..FibTN.TensorNetworks
using ..FibTN.Executor
using ..FibTN.QubitLattices
using ..FibTN.Visualizer
using ..FibTN.FibTensorTypes

using GLMakie

export index_labels, make_g2tt, contractionchain, build_tn, dumb_contract_tn
export tt2color, tt2marker, plot

index_labels(::Type{T}, group::Int) where T <: AbstractFibTensorType = [IndexLabel(group, p) for p in tensor_ports(T)]

make_g2tt(tt2gs::Dict{DataType, Vector{Int}}) = Dict(g => tt for (tt, gs) in tt2gs for g in gs)

function contractionchain(n1::Int, n2::Int, s1::Symbol, s2::Symbol)
    contractions = [(i, s1) => (i+1, s2) for i in n1:n2-1]
end

function build_tn(g2tt::Dict{Int, DataType}, contractions::Vector{<: Pair})
    tn = TensorNetwork()
    for (g, tt) in g2tt
        add_tensor!(tn, TensorLabel(g, index_labels(tt, g)))
    end
    for (i1, i2) in contractions
        add_contraction!(tn, IndexPair(IndexLabel(i1...), IndexLabel(i2...)))
    end
    tn
end

function dumb_contract_tn(tn::TensorNetwork, g2tt::Dict{Int, DataType})
    en = ExecNetwork(tn, Dict(g => tensor_data(tt) for (g, tt) in g2tt))
    for c in tn.contractions execute_step!(en, Contraction(c)) end
    et = execute_step!(en, FetchResult())
    et.indices, et.data
end

function get_states_and_amps(ql::QubitLattice, inds::Vector{IndexLabel}, data::SparseArray)
    lattice_states, amplitudes = Vector{Dict{Int, Int}}, Vector{Real}
    for cidx, amp in nonzero_pairs(data)
        push!(lattice_state, get_qubit_states(ql, inds, Tuple(cidx)))
        push!(amplitudes, amp)
    end
    lattice_states, state_amplitudes
end

tt2color(::Type{Reflector}) = :gray
tt2color(::Type{Boundary}) = :black
tt2color(::Type{VacuumLoop}) = :orange
tt2color(::Type{Vertex}) = :red
tt2color(::Type{Tail}) = :blue
tt2color(::Type{Crossing}) = :green
tt2color(::Type{Fusion}) = :teal

tt2marker(::Type{Reflector}) = :vline
tt2marker(::Type{Boundary}) = :xcross
tt2marker(::Type{VacuumLoop}) = :circle
tt2marker(::Type{Tail}) = :rect
tt2marker(::Type{Vertex}) = :star6
tt2marker(::Type{Crossing}) = :star4
tt2marker(::Type{Fusion}) = :star3

function plot(tn::TensorNetwork, positions::Vector{<:Tuple{<:Real, <:Real}}, g2tt::Dict{Int, DataType})
    groups = sort(collect(keys(g2tt)))
    colors = [tt2color(g2tt[g]) for g in groups]
    markers = [tt2marker(g2tt[g]) for g in groups]
    tnds = TensorNetworkDisplaySpec(groups, positions, colors, markers)
    
    f = Figure()
    ax = Axis(f[1, 1])
    hidedecorations!(ax)
    hidespines!(ax)
    visualize(tn, tnds, ax)
    DataInspector(f, range=30)
    display(f)
end

end # module IntegrationUtils
