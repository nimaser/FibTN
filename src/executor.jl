module Executor

using ..Indices
using ..TensorNetworks

using SparseArrayKit, TensorOperations

export ExecTensor, ExecNetwork, ExecStep, execute_step!
export Contraction, FetchResult, QRDecomp

struct ExecTensor
    id::Int
    groups::Set{Int}
    indices::Vector{IndexLabel}
    data::SparseArray
    function ExecTensor(id::Int, groups::Set{Int}, indices::Vector{IndexLabel}, data::AbstractArray)
        if (li = length(indices)) != (nd = ndims(data)) error("number of indices $li differs from array ndims $nd\n Indices: $indices") end
        new(id, groups, indices, SparseArray(data))
    end
end

mutable struct ExecNetwork
    tensor_from_id::Dict{Int, ExecTensor}
    id_from_index::Dict{IndexLabel, Int}
    next_id::Int
    function ExecNetwork(tn::TensorNetwork, tensordata_from_group::Dict{Int, <: AbstractArray})
        next_id = 1
        tensor_from_id = Dict{Int, ExecTensor}()
        id_from_index = Dict{IndexLabel, Int}()
        for tl in tn.tensors
            et = ExecTensor(next_id, Set(tl.group), copy(tl.indices), tensordata_from_group[tl.group])
            tensor_from_id[next_id] = et
            for index in tl.indices id_from_index[index] = next_id end
            next_id += 1
        end
        new(tensor_from_id, id_from_index, next_id)
    end
end
            
abstract type ExecStep end

struct Contraction <: ExecStep
    a::IndexLabel
    b::IndexLabel
    Contraction(ip::IndexPair) = new(ip.a, ip.b)
end

struct FetchResult <: ExecStep end

struct QRDecomp <: ExecStep
    # TODO
end

function execute_step!(en::ExecNetwork, c::Contraction)
    # make handles to id and exectensor
    ida = en.id_from_index[c.a]
    idb = en.id_from_index[c.b]
    eta = en.tensor_from_id[ida]
    etb = en.tensor_from_id[idb]
    # find contracted index positions
    pa = findfirst(==(c.a), eta.indices)
    pb = findfirst(==(c.b), etb.indices)
    # determine whether indices are on same exectensor, then use tensortrace or tensorcontract
    if eta === etb
        # build index label lists for tensortrace, all distinct except contracted indices
        IA = collect(1:length(eta.indices))
        IA[pa] = 0
        IA[pb] = 0
        
        # perform trace
        Z = tensortrace(eta.data, IA, false)
        # new indices are the uncontracted ones
        new_indices = deleteat!(copy(eta.indices), pa < pb ? (pa, pb) : (pb, pa))
    else
        # build index label lists for tensorcontract, all distinct except contracted indices
        IA = collect(1:length(eta.indices))
        IB = collect(length(eta.indices)+1:length(eta.indices)+length(etb.indices))
        IA[pa] = 0
        IB[pb] = 0
        
        # perform contraction
        Z = tensorcontract(
            eta.data, IA, false,
            etb.data, IB, false
        )
        # new indices are the uncontracted ones
        new_indices = vcat(
            deleteat!(copy(eta.indices), pa),
            deleteat!(copy(etb.indices), pb)
        )
    end
    # new tensor
    new_id = en.next_id
    new_groups = union(eta.groups, etb.groups)
    etz = ExecTensor(new_id, new_groups, new_indices, Z)
        
    #print("\n---------------------------------------------------------\n")
    #print(string(pa) * " " * string(pb))
    #print("\t" * string(c) * "\n")
    #print("a\n")
    #for idx in eta.indices
    #    print("\t" * string(idx) * "\n")
    #end
    #print("--b\n")
    #for idx in etb.indices
    #    print("\t" * string(idx) * "\n")
    #end
    #print("----z\n")
    #for idx in etz.indices
    #    print("\t" * string(idx) * "\n")
    #end
    #print(string(deleteat!(copy(eta.indices), pa)) * "\n")
    #print(string(deleteat!(copy(etb.indices), pb)) * "\n")
    
    # remove old tensors and ids: delete! is idempotent so no issues if we did the trace
    delete!(en.tensor_from_id, ida)
    delete!(en.tensor_from_id, idb)
    for idx in eta.indices delete!(en.id_from_index, idx) end
    for idx in etb.indices delete!(en.id_from_index, idx) end
    # add new tensor and id
    en.tensor_from_id[new_id] = etz
    for idx in new_indices en.id_from_index[idx] = new_id end
    # update overall id counter
    en.next_id += 1
        
    return nothing
end

function execute_step!(en::ExecNetwork, ::FetchResult)
    if length(en.tensor_from_id) != 1 error("ExecNetwork not fully contracted") end
    result_et = only(values(en.tensor_from_id))
end

function execute_step(en::ExecNetwork, qrd::QRDecomp)
    # TODO
end

end # module Executor
