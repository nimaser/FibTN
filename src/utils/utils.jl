using FibTN.TensorNetworks
using FibTN.TensorNetworkVisualizer
using FibTN.TOBackend

using FibTN.QubitLattices
using FibTN.QubitLatticeVisualizer

using FibTN.FibTensorTypes

using GLMakie
using SparseArrayKit

include("geometry.jl")

function contractionchain(n1::Int, n2::Int, s1::Symbol, s2::Symbol)
    contractions = [IC(IL(i, s1), IL(i+1, s2)) for i in n1:n2-1]
end

### EXECUTION HELPERS ###

build_es(tn::TensorNetwork, g2tt::Dict{Int, Type{<:AbstractFibTensorType}}) =
    ExecutionState(tn, Dict(g=>tensor_data(tt) for (g,tt) in g2tt))

do_steps(es::ExecutionState, execsteps::Vector{<:ExecutionStep}) =
    for step in execsteps execute_step!(es, step) end

fetch_result(es::ExecutionState) = begin
    et = es.tensor_from_id[only(get_ids(es))]; et.indices, et.data
end

function specified_contract(tn::TensorNetwork, g2tt::Dict{Int, Type{<:AbstractFibTensorType}}, contractions::Vector{IC})
    execsteps = [ContractionStep(c) for c in contractions]
    es = build_es(tn, g2tt)
    do_steps(es, execsteps)
    fetch_result(es)
end

naive_contract(tn::TensorNetwork, g2tt::Dict{Int, Type{<:AbstractFibTensorType}}) =
    specified_contract(tn, g2tt, tn.contractions)

### QL HELPERS ###

function build_ql(qubits_from_index::Dict{IL, Vector{Int}})
    ql = QubitLattice()
    for (idx, q) in qubits_from_index add_index!(ql, idx, q) end
    ql
end

function get_states_and_amps(ql::QubitLattice, inds::Vector{IndexLabel}, data::SparseArray)
    states, amps = Vector{Dict{Int, Int}}(), Vector{Real}()
    for (cidx, amp) in nonzero_pairs(data)
        push!(states, idxvals2qubitvals(ql, inds, [Tuple(cidx)...]))
        push!(amps, amp)
    end
    states, amps
end

### POST PROCESSING ###

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

color_from_qubit(qubitvals::Dict{Int, Int}) = Dict(q => v == 1 ? :red : :black for (q, v) in qubitvals)

position_from_index(indices::Vector{IL}, positions::Vector{Point2}) = Dict(i => positions[i.group] for i in indices)

function calculategridsidelengths(area::Int)
    width = height = floor(sqrt(area))
    if width == sqrt(area) return Int(width), Int(height) end
    width += 1
    while width * height < area
        width += 1
    end
    Int(width), Int(height)
end

function getaxisgrid(f, area::Int; args...)
    w, h = calculategridsidelengths(area)
    axs = [Axis(f[r, c]; aspect = DataAspect(), args...) for r in 1:h, c in 1:w]
    w, h, axs
end

function finalize!(f, axs)
    for ax in axs
        hidespines!(ax)
        hidedecorations!(ax)
    end
    resize_to_layout!(f)
end

function plot(tn::TensorNetwork, positions::Vector{Point2}, g2tt::Dict{Int, Type{<:AbstractFibTensorType}})
    groups = sort(collect(keys(g2tt)))
    colors = [tt2color(g2tt[g]) for g in groups]
    markers = [tt2marker(g2tt[g]) for g in groups]
    tnds = TensorNetworkDisplaySpec(groups, positions, colors, markers)
    
    f = Figure()
    ax = Axis(f[1, 1])
    TensorNetworkVisualizer.visualize(tn, tnds, ax)
    DataInspector(f, range=30)
    finalize!(f, [ax])
    display(GLMakie.Screen(), f)
    ax
end

function plot(ql::QubitLattice, positions::Vector{Point2}, qubitvals::Dict{Int, Int}, amp::Real) 
    qlds = QubitLatticeDisplaySpec(position_from_index(collect(QubitLattices.get_indices(ql)), positions), color_from_qubit(qubitvals), 0.5)

    f = Figure()
    ax = Axis(f[1, 1])
    QubitLatticeVisualizer.visualize(ql, qlds, ax)
    ax.title = "$amp"
    DataInspector(f, range=30)
    finalize!(f, [ax])
    display(GLMakie.Screen(), f)
    ax
end

function plot_interactive(ql::QubitLattice, positions::Vector{Point2}, states::Vector{Dict{Int, Int}})
    qlds = QubitLatticeDisplaySpec(position_from_index(collect(QubitLattices.get_indices(ql)), positions), color_from_qubit(qubitvals), 0.5)
    
# f, ax, p = graphplot(g, edge_width=4, edge_color=[colorant"black" for i in 1:ne(g)])
# julia> function action(idx, event, axis)
#            p.edge_color[][idx] = rand(RGB)
#            p.edge_color[] = p.edge_color[]
#        end
# julia> register_interaction!(ax, :edgeclick, EdgeClickHandler(action))
end

### INTEGRATION ###

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

include("fibtensornetworks.jl")
