using Graphs
using MetaGraphsNext
using GraphRecipes, Plots

using ITensors

function metagraphplot(mg::MetaGraph, vlabels::Bool=true, elabels::Bool=true)
    # graphplot indices labels by Graphs.jl codes, so we create an array of
    # vlabels in code order, and a dict of (code1, code2) => elabel
    labelargs = []
    if vlabels push!(labelargs, :names=>collect(map(string, labels(mg)))) end
    if elabels push!(labelargs, :edgelabel=>Dict(code_for.((mg,), t) => mg[t...] for t in collect(edge_labels(mg)))) end
    GraphRecipes.graphplot(mg, curves=false; labelargs...)
end

mutable struct VertexData
    type::Symbol
    tensor::ITensor
    VertexData(type) = new(type)
    VertexData(type, tensor) = new(type, tensor)
end

mutable struct EdgeData
    indices::Vector{Index}
    tensor::ITensor
    EdgeData(indices) = new(indices)
    EdgeData(indices, tensor) = new(indices, tensor)
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
        g[startindex + i] = VertexData(:GSTriangle)
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
        g[startindex + i] = VertexData(:GSTriangle)
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
    g[nextindex] = VertexData(:Trivial)

    # add new edge
    g[v, nextindex] = EdgeData([])
end

function cap_remaining!(g::MetaGraph)
    for v in collect(labels(g))
        if degree(g, v) == 2 add_cap!(g, v) end
    end
end

function initvertex(g::MetaGraph, v::Int)
    # create tensor for this vertex
    if g[v].type == :GSTriangle
        g[v].tensor = GSTriangle()
    elseif g[v].type == :Trivial
        g[v].tensor = StringTripletVector(1)
    end

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
    g[v1, v2].tensor = StringTripletReflector(g[v1, v2].indices...)
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

function contractgraph(g::MetaGraph)
    edgetensors = [g[e...].tensor for e in collect(edge_labels(g))]
    vertextensors = [g[v].tensor for v in collect(labels(g))]
    tensors = append(edgetensors, vertextensors)
    T = *(tenors...)
end
