module Executor

using ..Indices
using ..TensorNetworks

using SparseArrayKit, TensorOperations

export ExecTensor, ExecNetwork, ExecStep, execute_step!
export Contraction, QRDecomp

struct ExecTensor
    id::Int
    groups::Set{Int}
    indices::Vector{IndexLabel}
    data::SparseArray
    function ExecTensor(id, groups, indices, data)
        if length(indices) != ndims(data) error("number of indices differs from array ndims") end
        new(id, groups, indices, data)
    end
end

mutable struct ExecNetwork
    tensor_from_id::Dict{Int, ExecTensor}
    id_from_index::Dict{IndexLabel, Int}
    next_id::Int
    function ExecNetwork(tn::TensorNetwork, tensordata_from_group::Dict{Int, <: SparseArray})
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

struct QRDecomp <: ExecStep
    # TODO
end

function execute_step!(en::ExecNetwork, c::Contraction)
    ida = en.id_from_index[c.a]
    idb = en.id_from_index[c.b]
    Ta = en.tensor_from_id[ida]
    Tb = en.tensor_from_id[idb]
    # find index positions
    pa = findfirst(==(c.a), Ta.indices)
    pb = findfirst(==(c.b), Tb.indices)
    # build index label lists for tensorcontract
    IA = collect(1:length(Ta.indices))
    IB = collect(1:length(Tb.indices))
    # assign same label to contracted indices
    IA[pa] = 0
    IB[pb] = 0
    # perform contraction
    C = tensorcontract(
        Ta.data, IA, false,
        Tb.data, IB, false
    )
    # new indices are uncontracted ones
    new_indices = vcat(
        deleteat!(copy(Ta.indices), pa),
        deleteat!(copy(Tb.indices), pb)
    )
    # new tensor
    new_id = en.next_id
    new_groups = union(Ta.groups, Tb.groups)
    Tc = ExecTensor(new_id, new_groups, new_indices, C)

    # remove old tensors
    delete!(en.tensor_from_id, ida)
    delete!(en.tensor_from_id, idb)
    # remove old ids
    for idx in Ta.indices delete!(en.id_from_index, idx) end
    for idx in Tb.indices delete!(en.id_from_index, idx) end
    # add new tensor and id
    en.tensor_from_id[new_id] = Tc
    for idx in new_indices en.id_from_index[idx] = new_id end
    # update overall id counter
    en.next_id += 1
    return nothing
end

function execute_step(en::ExecNetwork, qrd::QRDecomp)
    # TODO
end

end # module Executor
