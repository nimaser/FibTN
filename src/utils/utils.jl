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
