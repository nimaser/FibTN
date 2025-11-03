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

struct VertexData
    indices::Vector{Index}
end

struct EdgeData
    indices::Vector{Index}
end

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
        g[startindex + i] = VertexData([])
    end

    # add new edges
    for i in 1:order-1
        g[startindex + i, startindex + i + 1] = EdgeData([])
    end
    g[startindex + order, startindex + 1] = EdgeData([])
end

function add_chain!(g::MetaGraph, links::Int, v1::Int, v2::Int)
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
        g[startindex + i] = VertexData([])
    end

    # add new edges
    g[v1, startindex + 1] = EdgeData([])
    for i in 1:links-2
        g[startindex + i, startindex + i + 1] = EdgeData([])
    end
    g[startindex + links - 1, v2] = EdgeData([])
end


