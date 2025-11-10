using Graphs
using MetaGraphsNext
using GraphRecipes, Plots

using ITensors
using ITensorUnicodePlots

###############################################################################
# GS LATTICE CONSTRUCTION
###############################################################################

@enum TensorType GSTriangle GSCap GSSquare GSCircle GSTail

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
        @show prev, next
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

    # add new vertices
    @show edgestomake
    startindex=nv(rsg) + 1
    if edgestomake == 2
        rsg[startindex] = rsgVertexData((GSTriangle, [v1, v2]))
    else
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

function add_cap!(rsg::MetaGraph, v)
    # check that v will not get too many edges
    if degree(rsg, code_for(rsg, v)) >= 3 throw(ErrorException("Cannot add cap to vertex $v")) end

    # add new vertex
    nextindex = nv(g) + 1
    rsg[nextindex] = rsgVertexData((GSCap, [v]))

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

#function make_ig(rsg::MetaGraph)
#    # initialize index graph with tensor indices at each vertex and contracted indices at edges
#    ig = MetaGraph(
#        Graph()::SimpleGraph;
#        label_type=Int,
#        vertex_data_type=Set{Index},
#        edge_data_type=Set{Index},
#        graph_data=nothing
#    )
#
#    for v in labels(rsg)
#        # copy vertex and create indices
#        v1 = Index(5, "virt,$(v)-v1")
#        v2 = Index(5, "virt,$(v)-v2")
#        v3 = Index(5, "virt,$(v)-v3")
#        p1 = Index(5, "phys,$(v)-p1")
#        ig[v] = Set(indices)
#        
#        # copy edges and create index sets if they haven't already been made
#        for endpoint in rsg[v]
#            if !haskey(ig, v, endpoint)
#                ig[v, endpoint] = Set()
#            end
#        end
#
#        # assign indices to edges in clockwise order
#        
#
#    end
#    
#    ig
#end

### TENSOR GRAPH ###




### QUBIT GRAPH ###


