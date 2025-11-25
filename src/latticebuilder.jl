#=
# This module provides an API to build ground state lattices.
#
#
#
#
=#

using Printf

using Graphs
using MetaGraphsNext
using GLMakie, GraphMakie

using ITensors

###############################################################################
# GS LATTICE CONSTRUCTION
###############################################################################

### ROTATION SYSTEM GRAPH ###

const rsgEdgeCycle = Vector{Int}
const rsgVertexData = @NamedTuple{type::TensorType, ecycle::rsgEdgeCycle}
const rsgBoundaryEdgeSet = Set{Tuple{Int, Int}}

function new_plaquette(order::Int)
    # check argument
    if order <= 2 throw(ArgumentError("Plaquettes must be of order >= 3")) end

    # initialize new graph with tensor type and edge cycle stored at vertices
    rsg = MetaGraph(
        Graph()::SimpleGraph;
        label_type=Int,
        vertex_data_type=rsgVertexData,
        edge_data_type=Nothing,
        graph_data=rsgBoundaryEdgeSet()
    )

    # create new vertices
    rsg[1] = rsgVertexData((GSTriangle, [order, 2]))
    for i in 2:order-1
        rsg[i] = rsgVertexData((GSTriangle, [i-1, i+1]))
    end
    rsg[order] = rsgVertexData((GSTriangle, [order-1, 1]))

    # create new edges and add them to boundary set
    for i in 1:order-1
        rsg[i, i+1] = nothing
        push!(rsg[], (i, i+1))
    end
    rsg[order, 1] = nothing
    push!(rsg[], (order, 1))

    rsg
end

function add_plaquette!(rsg::MetaGraph, v1::Int, v2::Int, order::Int)
    # check arguments
    if order <= 2 throw(ArgumentError("Plaquettes must be of order >= 3, got $order")) end
    if v1 == v2 throw(ArgumentError("v1 and v2 must be different, got $v1, $v2")) end

    # check that v1 and v2 will not get too many edges
    if degree(rsg, code_for(rsg, v1)) != 2 throw(ErrorException("vertex $v1 must be deg 2")) end
    if degree(rsg, code_for(rsg, v2)) != 2 throw(ErrorException("vertex $v2 must be deg 2")) end

    # check that v1 and v2 are on the boundary
    v1valid = v2valid = false
    for edge in rsg[]
        if edge[1] == v1 || edge[2] == v1
            v1valid = true
        end
        if edge[1] == v2 || edge[2] == v2
            v2valid = true
        end
        if v1valid && v2valid break end
    end
    if !v1valid throw(ErrorException("vertex $v1 not on the boundary")) end
    if !v2valid throw(ErrorException("vertex $v2 not on the boundary")) end

    # make copy so we can restore state if there's an error
    B = deepcopy(rsg[])
    
    # remove boundary between v1 and v2, counting number of edges along the way
    prev = rsg[v1].ecycle[1] # v1 must be degree 2 so prev edge is at index 1
    next = v1
    numedges = 0
    while true
        previdx = findfirst(x->x==prev, rsg[next].ecycle)
        prev, next = next, rsg[next].ecycle[previdx+1]
        delete!(rsg[], (prev, next))
        numedges += 1
        if next == v2 break end
    end

    # if the plaquette is too small, revert boundary set and error
    edgestomake = order - numedges
    if edgestomake < 1
        empty!(rsg[])
        push!(rsg[], B...)
        throw(ErrorException("can't make order $order plaquette with $numedges edges between given vertices"))
    end

    # if only one edge needs to be added, add it to graph and boundary set, modify edge cycles, and dip
    if edgestomake == 1
        rsg[v1, v2] = nothing
        push!(rsg[], (v1, v2))
        insert!(rsg[v1].ecycle, 2, v2)
        insert!(rsg[v2].ecycle, 2, v1)
        return
    end

    # add new vertices
    startindex=nv(rsg) + 1
    if edgestomake == 2
        rsg[startindex] = rsgVertexData((GSTriangle, [v1, v2]))
    elseif edgestomake > 2
        rsg[startindex] = rsgVertexData((GSTriangle, [v1, startindex+1]))
        for i in startindex+1:startindex+edgestomake-3
            rsg[i] = rsgVertexData((GSTriangle, [i-1, i+1]))
        end
        rsg[startindex+edgestomake-2] = rsgVertexData((GSTriangle, [startindex+edgestomake-3, v2]))
    end

    # modify edge cycles of v1 and v2
    insert!(rsg[v1].ecycle, 2, startindex)
    insert!(rsg[v2].ecycle, 2, startindex+edgestomake-2)

    # add new edges and add them to the boundary
    rsg[v1, startindex] = nothing
    push!(rsg[], (v1, startindex))
    for i in startindex:startindex+edgestomake-3
        rsg[i, i + 1] = nothing
        push!(rsg[], (i, i+1))
    end
    rsg[startindex+edgestomake-2, v2] = nothing
    push!(rsg[], (startindex+edgestomake-2, v2))

    nothing
end

function add_cap!(rsg::MetaGraph, v::Int)
    # check that v will not get too many edges
    if degree(rsg, code_for(rsg, v)) >= 3 throw(ErrorException("Cannot add cap to vertex $v")) end

    # add new vertex
    nextindex = nv(rsg) + 1
    rsg[nextindex] = rsgVertexData((StringTripletVector, [v]))

    # modify edge cycle in v - cap will be on the outside, though I don't think it matters
    insert!(rsg[v].ecycle, 2, nextindex)

    # add new edge
    rsg[v, nextindex] = nothing

    nothing
end

function cap_all!(rsg::MetaGraph)
    for v in collect(labels(rsg))
        if degree(rsg, code_for(rsg, v)) == 2 add_cap!(rsg, v) end
    end
end

### INDEX GRAPH ###

const igVertexData = @NamedTuple{type::TensorType, vinds::Vector{Index}, pinds::Vector{Index}}

function rsg2ig(rsg::MetaGraph)
    # initialize index graph with tensor indices at each vertex and contracted indices at edges
    ig = MetaGraph(
        Graph()::SimpleGraph;
        label_type=Int,
        vertex_data_type=Union{igVertexData, Nothing},
        edge_data_type=Vector{Index},
        graph_data=rsg
    )

    # create all vertices to start
    for v in labels(rsg)
        ig[v] = nothing
    end
    
    for v in labels(rsg)
        # create indices for v
        vinds, pinds = make_tensor_indices(v, rsg[v].type)
        ig[v] = igVertexData((rsg[v].type, vinds, pinds))
        

        # copy edges attached to v and create index sets if they haven't already been made
        for endpoint in rsg[v].ecycle
            if !haskey(ig, v, endpoint)
                ig[v, endpoint] = []
            end
        end

        # assign indices to edges in clockwise order
        if length(rsg[v].ecycle) != length(vinds)
            throw(ErrorException("number of vinds doesn't match number of contractions at vertex $v"))
        end
        for (endpoint, vind) in zip(rsg[v].ecycle, vinds)
            push!(ig[v, endpoint], vind)
        end
    end
    
    ig
end

### QUBIT GRAPH ###

function ig2qg(ig::MetaGraph)
    # initialize tensor graph with qubit values at edges
    qg = MetaGraph(
        Graph()::SimpleGraph;
        label_type=Int,
        vertex_data_type=Nothing,
        edge_data_type=Bool,
        graph_data=ig
    )

    # copy vertices which border edges with qubits
    for v in labels(ig)
        if length(ig[v].pinds) > 0
            qg[v] = nothing
        end
    end

    # copy edges with qubits on them
    for e in edge_labels(ig)
        if haskey(qg, e[1]) && haskey(qg, e[2])
            qg[e...] = false
        end
    end
    
    qg
end

function fillfrompvals(qg::MetaGraph, pvals::Dict{<:Index, Int})
    ig = qg[]
    rsg = ig[]

    for e in edge_labels(qg)
        # get index of edge in ecycle
        v1, v2 = e[1], e[2]
        v2idx = findfirst(x->x==v2, rsg[v1].ecycle)
        
        # get ijk at v1
        pind = ig[v1].pinds[1]
        ijk = p2ijk(pvals[pind])

        # set qubit value on edge
        qg[e...] = ijk[v2idx] == FibonacciAnyon(:τ)
    end
end

### TENSOR GRAPH ###

mutable struct tgVertexData type::TensorType; tensor::ITensor end

function ig2tg(ig::MetaGraph)
    # initialize tensor graph with tensors at each vertex and reflection tensors at each edge
    tg = MetaGraph(
        Graph()::SimpleGraph;
        label_type=Int,
        vertex_data_type=tgVertexData,
        edge_data_type=Set{ITensor},
        graph_data=ig
    )

    # create all vertices and their tensors
    for v in labels(ig)
        tg[v] = tgVertexData(ig[v].type, make_tensor(ig[v].type, ig[v].vinds, ig[v].pinds))
    end

    # create all edges and their tensors
    for e in edge_labels(ig)
        tg[e...] = Set([make_tensor(StringTripletReflector, ig[e...], Index[])])
    end

    tg
end

function contractedge!(tg::MetaGraph, v1::Int, v2::Int)
    # check that an edge actually exists between v1 and v2
    if !haskey(tg, v1, v2) throw(ErrorException("no edge between vertices $v1 and $v2")) end

    # contract tensors on v1, edge, and v2, going contraction by contraction to reduce memory usage;
    # we use v1 as the new result vertex
    for t in tg[v1, v2]
        tg[v1].tensor *= t
    end
    tg[v1].tensor *= tg[v2].tensor
    tg[v1].type = Composite

    # get list of vertices which are contracted with v2, not including v1
    nbs2 = setdiff(collect(neighbor_labels(tg, v2)), [v1])

    # vertices which contract with both v1 and v2 would induce multiple edges between the new v1 and
    # them, but we don't want to use multigraphs, so we just toss all of the contractions from those
    # multiple edges into a single edge's tensor set
    for nb in nbs2
        if haskey(tg, v1, nb)
            push!(tg[v1, nb], tg[v2, nb]...)
        else
            tg[v1, nb] = tg[v2, nb]
        end
        rem_edge!(tg, code_for(tg, v2), code_for(tg, nb))
    end

    # remove v2
    rem_vertex!(tg, code_for(tg, v2))

    nothing
end

function contractcaps!(tg::MetaGraph, displayintermediateresults::Bool=false)
    caps = [l for l in labels(tg) if degree(tg, code_for(tg, l)) == 1]
    for cap in caps 
        v = collect(neighbor_labels(tg, cap))[1]
        contractedge!(tg, v, cap)
        if displayintermediateresults
            f, ax, p = tgplot(tg)
            display(f)
            readline()
        end
    end
end

function contractsequence!(tg::MetaGraph, sequence::Vector{Int}, displayintermediateresults::Bool=false)
    while length(sequence) > 1
        contractedge!(tg, sequence[1], sequence[2])
        deleteat!(sequence, 2)
        if displayintermediateresults && length(collect(labels(tg))) > 1
            f, ax, p = tgplot(tg)
            display(f)
            readline()
        end
    end
end

function contractgraph!(tg::MetaGraph)
    # contracting the caps first saves a decent chunk of memory
    contractcaps!(tg)
    
    # contract everything else at once
    while nv(tg) > 1
        e = collect(edge_labels(tg))[1]
        contractedge!(tg, e...)
    end
end

function contractionresult(tg::MetaGraph)
    labs = collect(labels(tg))
    if length(labs) != 1 throw(ErrorException("Tensor Graph has not been fully contracted")) end
    T = tg[labs[1]].tensor
end

function tensor2states(T::ITensor)
    # physical indices are the only ones left after contraction is finished
    pinds = inds(T)

    # get nonzero entries/states
    Tarr = array(T)
    nzidxs = findall(!iszero, Tarr) # each entry describes a state in terms of its pind values

    # get mapping from pind to pval for a specific state s (ie entry in nzidxs)
    pind2pval = s -> Dict(pinds[i]=>v for (i, v) in enumerate(s))

    # make dict from state to (dict from pind to pval), state amplitude
    states = Dict(s=>(pind2pval(Tuple(s)), Tarr[s]) for s in nzidxs)
end

function normalizestates!(states::Dict{<:CartesianIndex, <:Tuple{<:Dict{<:Index, Int}, Float64}})
    magnitude = 0
    for (idx, (pvals, amp)) in states
        magnitude += amp
    end

    for (idx, (pvals, amp)) in states
        states[idx] = (pvals, amp/magnitude)
    end
end

