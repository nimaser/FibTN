using Graphs
using MetaGraphsNext
using GraphRecipes, Plots

using ITensors
using ITensorUnicodePlots

###############################################################################
# GS LATTICE CONSTRUCTION
###############################################################################

@enum TensorType GSTriangle GSCap GSSquare GSCircle GSTail Composite

function make_tensor_indices(tensorlabel::Any, type::TensorType)
    if type == GSTriangle
        v1 = Index(5, "virt,$(tensorlabel)-v1")
        v2 = Index(5, "virt,$(tensorlabel)-v2")
        v3 = Index(5, "virt,$(tensorlabel)-v3")
        p1 = Index(5, "phys,$(tensorlabel)-p1")
        return [v1, v2, v3], [p1]
    end
    if type == GSCap
        v1 = Index(5, "virt,$(tensorlabel)-v1")
        return [v1], []
    end
end

function get_readable_index_id(i::Index)
    for tag in tags(i)
        if '-' ∈ string(tag) return tag end
    end
end

function make_tensor(type::TensorType, vinds::Vector{Index}, pinds::Vector{Index})
    if type == GSTriangle
        return GSTriangle(vinds, pinds)
    end
    if type == GSCap
        return GSCap(vinds)
    end
end

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

    # add new vertices
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

function add_cap!(rsg::MetaGraph, v::Int)
    # check that v will not get too many edges
    if degree(rsg, code_for(rsg, v)) >= 3 throw(ErrorException("Cannot add cap to vertex $v")) end

    # add new vertex
    nextindex = nv(rsg) + 1
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

function rsgplot(rsg::MetaGraph)
    type2shape = t -> t == GSTriangle ? :hexagon : t == GSCap ? :rect : :circle

    commonargs = Dict(:title=>"Rotation System Graph (rsg)",
                      :curves=>false,
                      :fontsize=>4,
                      :fillcolor=>:lightgray,
                      :nodesize=>0.3,
                     )
    # graphplot uses Graphs.jl codes, so provide vertex properties in vertex code ordered list
    # and provide edge properties as a dict of of (code1, code2) => edgeprop
    labelstrings = map(string, labels(rsg))
    separateargs = Dict(:names=>collect(labelstrings),
                        :node_weights=>[1/length(l) for l in labelstrings],
                        :edgecolor=>Dict(code_for.((rsg,),t) => t ∈ rsg[] || reverse(t) ∈ rsg[] ? :red : :black for t in edge_labels(rsg)),
                        :nodeshape=>collect(map(type2shape, [rsg[l].type for l in labels(rsg)]))
                       )
    rsg_plot = GraphRecipes.graphplot(rsg; commonargs..., separateargs...)
end

### INDEX GRAPH ###

const igVertexData = @NamedTuple{type::TensorType, vinds::Vector{Index}, pinds::Vector{Index}}

function make_ig(rsg::MetaGraph)
    # initialize index graph with tensor indices at each vertex and contracted indices at edges
    ig = MetaGraph(
        Graph()::SimpleGraph;
        label_type=Int,
        vertex_data_type=Union{igVertexData, Nothing},
        edge_data_type=Set{Index},
        graph_data=nothing
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
                ig[v, endpoint] = Set()
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

function igplot(ig::MetaGraph)
    type2shape = t -> t == GSTriangle ? :hexagon : t == GSCap ? :rect : :circle
    idxset2label = s -> join([get_readable_index_id(i) for i in s], ';')

    commonargs = Dict(:title=>"Index Graph (ig)",
                      :curves=>false,
                      :fontsize=>4,
                      :fillcolor=>:lightgray,
                      :nodesize=>0.3,
                     )
    # graphplot uses Graphs.jl codes, so provide vertex properties in vertex code ordered list
    # and provide edge properties as a dict of of (code1, code2) => edgeprop
    labelstrings = map(string, labels(ig))
    separateargs = Dict(:names=>collect(labelstrings),
                        :node_weights=>[1/length(l) for l in labelstrings],
                        :edgelabel=>Dict(code_for.((ig,),t) => idxset2label(ig[t...]) for t in edge_labels(ig)),
                        :nodeshape=>collect(map(type2shape, [ig[l].type for l in labels(ig)]))
                       )
    ig_plot = GraphRecipes.graphplot(ig; commonargs..., separateargs...)
end

### TENSOR GRAPH ###

const tgVertexData = @NamedTuple{type::TensorType, tensor::ITensor}

function make_tg(ig::MetaGraph)
    # initialize tensor graph with tensors at each vertex and reflection tensors at each edge
    tg = MetaGraph(
        Graph()::SimpleGraph;
        label_type=Int,
        vertex_data_type=tgVertexData,
        edge_data_type=ITensor,
        graph_data=nothing
    )

    # create all vertices and their tensors
    for v in labels(ig)
        tg[v] = tgVertexData(ig[v].type, make_tensor(ig[v].type, ig[v].vinds, ig[v].pinds))
    end

    # create all edges and their tensors
    for e in edge_labels(ig)
        tg[e...] = StringTripleReflector(ig[e...]...)
    end
    
    tg
end

function tgplot(tg::MetaGraph)
    type2shape = t -> t == GSTriangle ? :hexagon : t == GSCap ? :rect : :circle

    commonargs = Dict(:title=>"Tensor Graph (tg)",
                      :curves=>false,
                      :fontsize=>4,
                      :fillcolor=>:lightgray,
                      :nodesize=>0.3,
                     )
    # graphplot uses Graphs.jl codes, so provide vertex properties in vertex code ordered list
    # and provide edge properties as a dict of of (code1, code2) => edgeprop
    labelstrings = map(string, labels(tg))
    separateargs = Dict(:names=>collect(labelstrings),
                        :node_weights=>[1/length(l) for l in labelstrings],
                        :edgelabel=>Dict(code_for.((tg,),t) => idxset2label(inds(tg[t...])) for t in edge_labels(tg)),
                        :nodeshape=>collect(map(type2shape, [tg[l].type for l in labels(tg)]))
                       )
    tg_plot = GraphRecipes.graphplot(tg; commonargs..., separateargs...)
end

function contractedge!(tg::MetaGraph, v1::Int, v2::Int)
    # check that an edge actually exists between v1 and v2
    if !haskey(tg, v1, v2) throw(ErrorException("no edge between vertices $v1 and $v2")) end

    # contract tensors on v1, edge, and v2
    temp = @visualize tg[v1].tensor * tg[v1, v2] * tg[v2].tensor
    
    # get references to all edge tensors

    # contract the tensors from the two vertices and edge, then store in remaining vertex
    g[v].tensor = g[v].tensor *g[v, c].tensor * g[c].tensor

    # remove cap and connecting edge
    rem_edge!(g, code_for(g, v), code_for(g, c))
    rem_vertex!(g, code_for(g, c))
end

function contractcaps!(g::MetaGraph)
    caps = [l for l in collect(labels(g)) if degree(g, code_for(g, l)) == 1]
    for cap in caps contractcap!(g, cap) end
end

function contractedge!(g::MetaGraph, v1::Int, v2::Int)
    # T = g[v1].tensor * g[v1, v2].tensor * g[v2].tensor
end

function contractgraph(g::MetaGraph)
    # contracting the caps first saves a decent chunk of memory
    contractcaps!(g)
    
    # contract everything else in bulk
    tensors = [g[e...].tensor for e in collect(edge_labels(g))]
    append!(tensors, [g[v].tensor for v in collect(labels(g))])
    
    T = @visualize *(tensors...)
end

### QUBIT GRAPH ###


