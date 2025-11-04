using Graphs
using MetaGraphsNext
using GraphRecipes, Plots

using ITensors
using ITensorUnicodePlots

mutable struct VertexData
    tensor::ITensor
end

mutable struct EdgeData
    indices::Vector{Index}
    tensor::ITensor
    EdgeData(indices) = new(indices)
end

###############################################################################
# GS LATTICE CONSTRUCTION
###############################################################################

function new_plaquette(order::Int)
    # initialize new graph
    g = MetaGraph(
        Graph()::SimpleGraph;
        label_type=Int,
        vertex_data_type=VertexData,
        edge_data_type=EdgeData,
        graph_data=nothing
    )
    add_plaquette!(g, order)
    g
end

function add_plaquette!(g::MetaGraph, order::Int)
    if order == 2 throw(ArgumentError("Plaquettes cannot be of order two")) end

    # add new vertices
    startindex = nv(g)
    for i in 1:order
        g[startindex + i] = VertexData(GSTriangle())
    end
    # add new edges
    for i in 1:order-1
        g[startindex + i, startindex + i + 1] = EdgeData([])
    end
    g[startindex + order, startindex + 1] = EdgeData([])
end

function add_chain!(g::MetaGraph, links::Int, v1::Int, v2::Int, vdata=(), edata=nothing)
    if degree(g, v1) >= 3 throw(ErrorException("Cannot add edge to degree 3 vertex $v1")) end 
    if degree(g, v2) >= 3 throw(ErrorException("Cannot add edge to degree 3 vertex $v2")) end
    if links == 1 && haskey(g, v1, v2)
        throw(ErrorException("Cannot add multiple edges between vertices"))
    end
    if v1 == v2
        if links == 1 && degree(g, v1) > 1
            throw(ErrorException("Cannot add two edges to degree $(degree(g, v1)) vertex"))
        end
        if links == 2
            throw(ArgumentError("Plaquettes cannot be of order two"))
        end
    end

    # add new vertices
    startindex=nv(g)
    for i in 1:links-1
        g[startindex + i] = VertexData(GSTriangle())
    end

    # add new edges
    g[v1, startindex + 1] = EdgeData([])
    for i in 1:links-2
        g[startindex + i, startindex + i + 1] = EdgeData([])
    end
    g[startindex + links - 1, v2] = EdgeData([])
end

function add_cap!(g::MetaGraph, v)
    if degree(g, v) >= 3 throw(ErrorException("Cannot add cap to degree 3 vertex $v")) end
 
    # add new vertex
    nextindex = nv(g) + 1
    g[nextindex] = VertexData(StringTripletVector(1))

    # add new edge
    g[v, nextindex] = EdgeData([])
end

function cap_remaining!(g::MetaGraph)
    for v in collect(labels(g))
        if degree(g, v) == 2 add_cap!(g, v) end
    end
end

###############################################################################
# GS LATTICE INITIATION
###############################################################################

function initvertex(g::MetaGraph, v::Int)
    # put an index on each edge
    nbs = neighbor_labels(g, v)
    virtids = virtualindices(g[v].tensor)
    if length(nbs) != length(virtids)
        throw(ErrorException("must have one virt idx per edge"))
    end
    for (nb, idx) in zip(nbs, virtids)
        push!(g[v, nb].indices, idx)
    end
end

function initedge(g::MetaGraph, v1::Int, v2::Int)
    if length(g[v1, v2].indices) != 2 throw(ErrorException("edge has too many indices")) end
    # g[v1, v2].tensor = StringTripletReflector(g[v1, v2].indices...)
    g[v1, v2].tensor = ITensors.δ(g[v1, v2].indices...)
end

function populatetensors(g::MetaGraph)
    # iterate over vertices, giving them each a new tensor of the appropriate type,
    # and placing their indices in the appropriate edges
    for v in collect(labels(g))
        initvertex(g, v)
    end

    # iterate over edges, placing a reflection tensor on each one
    for (v1, v2) in collect(edge_labels(g))
        initedge(g, v1, v2)
    end
end

###############################################################################
# GS LATTICE CONTRACTION
###############################################################################

function contractcap!(g::MetaGraph, c::Int)
    nbs = collect(neighbor_labels(g, c))
    if length(nbs) != 1 throw(ArgumentError("vertex $c is not a cap; it has degree >1")) end
    v = nbs[1]

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

###############################################################################
# GS LATTICE VISUALIZATION 
###############################################################################

function metagraphplot(mg::MetaGraph, vlabels::Bool=true, elabels::Bool=true)
    # graphplot indices labels by Graphs.jl codes, so we create an array of
    # vlabels in code order, and a dict of (code1, code2) => elabel
    args = Dict(:curves=>false, :nodesize=>0.5, :node_weights=>ones(length(labels(mg))))
    labelargs = []
    if vlabels push!(labelargs, :names=>collect(map(string, labels(mg)))) end
    if elabels push!(labelargs, :edgelabel=>Dict(code_for.((mg,), t) => mg[t...] for t in collect(edge_labels(mg)))) end
    GraphRecipes.graphplot(mg; args..., labelargs...)
end

function alignmentlattice(g::MetaGraph)
    a = MetaGraph(
        Graph()::SimpleGraph;
        label_type=Int,
        vertex_data_type=Vector{Index},
        edge_data_type=Vector{Index},
        graph_data = nothing
    )
    for v in collect(labels(g))
        a[v] = [i for i in inds(g[v].tensor)]
    end
    for (v1, v2) in collect(edge_labels(g))
        a[v1, v2] = g[v1, v2].indices
    end
    a
end

function edge2physicalindexandorientation(a::MetaGraph, v1::Int, v2::Int)
    # given an edge, get a virtual index along it
    i = a[v1, v2][1]
    # find out which virtual index it was among the tensor it came from
    if hastags(i, "i1") o = 1
    elseif hastags(i, "i2") o = 2
    elseif hastags(i, "i3") o = 3
    end
    # determine the physical index of the tensor it came from
    p = i ∈ a[v1] ? physicalindices(a[v1])[1] : physicalindices(a[v2])[2]
    p, o
end

function qubitlattice(a::MetaGraph)
    q = MetaGraph(
        Graph()::SimpleGraph;
        label_type=Int,
        vertex_data_type=Nothing,
        edge_data_type=Bool,
        graph_data=nothing
    )
    for v in collect(labels(a))
        q[v] = nothing
    end
    for (v1, v2) in collect(edge_labels(a))
        q[v1, v2] = false
    end
    q
end

function getnzstates(T)
    Tarr = array(T)
    nzi = findall(!iszero, Tarr)
    s = Dict(idx=>T[idx] for idx in nzi)
end

function generatequbitlattices(T::ITensor, a::MetaGraph, q::MetaGraph)
    qgraphs = []

    # remove 0 amplitude states
    states = getnzstates(T)

    # create mapping from Index to index in vals
    Tidx2i_map = Dict(idx=>i for (i, idx) in enumerate(inds(T)))

    # for each state
    for (vals, amp) in states
        # for each edge in the graph
        for e in collect(edge_labels(q))
            # find the pindex and orientation of the qubit within that index 
            p, o = edge2physicalindexandorientation(a, e...)
            # find the value of that index, and thus the value of the qubit
            pval = vals[Tidx2i_map[p]]
            qval = p2ijk(pval)[o]
            q[e...] = qval == FibonacciAnyon(:I) ? 0 : 1
        end
        push!(qgraphs, deepcopy(q))
    end

    qgraphs
end

function qubitlatticeplot(qgraphs::Any)
    commonargs = Dict(:curves=>false, :nodeshape=>:circle, :fillcolor=>:lightgray, :names=>collect(map(string, labels(q))))

    qgraphplots = []
    for q in qgraphs
        lc = Dict(code_for.((q,),t) => q[t...] ? :red : :black for t in collect(edge_labels(q)))
        gp = GraphRecipes.graphplot(q, curves=false, edgecolor=lc; commonargs...)
        push!(qgraphplots, gp)
    end
    plot(qgraphplots...)
end

