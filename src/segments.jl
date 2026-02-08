module Segments

using ..TensorNetworks
using ..FibTensorNetworks
using ..TOBackend

export Segment, get_ftn, segment_ports, segment_data

const IL = IndexLabel
const IC = IndexContraction

"""Completely non-errorchecked segment implementation."""
struct Segment
    toptype::Union{Type{Tail}, Type{Vector}}
    midtail::Bool
    midloop::Bool
    bottype::Union{Type{Tail}, Type{Vector}}
end

function get_ftn(s::Segment)
    ftn = FibTensorNetwork()
    # add tensors
    groupnum = 1
    add_tensor!(ftn, groupnum, toptype); groupnum += 1
    add_tensor!(ftn, groupnum, Reflector); groupnum += 1
    if midtail
        add_tensor!(ftn, groupnum, Tail); groupnum += 1
        add_tensor!(ftn, groupnum, Reflector); groupnum += 1
    end
    if midloop
        add_tensor!(ftn, groupnum, VacuumLoop); groupnum += 1
        add_tensor!(ftn, groupnum, Reflector); groupnum += 1
    end
    add_tensor!(ftn, groupnum, bottype)
    # add contractions
    topport = toptype == Tail ? :b : :c
    botport = bottype == Tail ? :a : :c
    add_contraction!(ftn.tn, IC(IL(1, topport), IL(2, :a)))
    for i in 2:2:groupnum-2
        add_contraction!(ftn.tn, IC(IL(i, :b), IL(i+1, :a)))
        add_contraction!(ftn.tn, IC(IL(i+1, :b), IL(i+2, :a)))
    end
    add_contraction!(ftn.tn, IC(IL(groupnum-1, :b), IL(groupnum, botport)))
    ftn
end

function tensor_ports(s::Segment)
    ports = []
    toptype == Vertex && push!(ports, :ta, :tb)
    toptype == Tail && push!(ports, :ta)
    bottype == Vertex && push!(ports, :ba, :bb)
    bottype == Tail && push!(ports, :bb)
    push!(ports, :tp)
    midtail && push!(ports, :mp)
    push!(ports, :bp)
end

function _il_from_port(s::Segment)
    maxgroupnum == s.hastail && s.hasloop ? 7 : s.hastail || s.hasloop ? 5 : 3
    ilmap = Dict{Symbol, IL}
    push!(ilmap, :ta => IL(1, :a))
    push!(ilmap, :tp => IL(1, :p))
    push!(ilmap, :bb => IL(maxgroupnum, :b))
    push!(ilmap, :bp => IL(maxgroupnum, :p))
    s.toptype == Vertex && push!(ilmap, :tb => IL(1, :b))
    s.bottype == Vertex && push!(ilmap, :ba => IL(maxgroupnum, :a))
    s.hastail && push!(ilmap, IL(3, :p) => :mp)
end

_segmentcache = Dict{Segment, <:AbstractArray}

function tensor_data(s::Segment)
    # fetch data if it is stored in the cache
    if haskey(_segmentcache, s) return _segmentcache[s] end
    # create the data array by creating and then contracting the associated ftn
    ftn = get_ftn(s)
    es = ExecutionState(ftn.tn, tensordata_from_group(ftn))
    execsteps = [ContractionStep(c) for c in ftn.tn.contractions]
    for execstep in execsteps execute_step!(es, execstep) end
    et = es.tensor_from_id[only(get_ids(es))]
    # permute data so index order matches ports
    il_from_port = _il_from_port(s)
    new_inds = [il_from_port[port] for port in tensor_ports(s)]
    execute_step!(es, PermuteIndicesStep(only(get_ids(es)), new_inds))
    _segmentcache[s] = et.data
    et.data
end

end # module Segments
