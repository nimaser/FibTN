using Graphs
using MetaGraphsNext
using GraphRecipes, Plots

using ITensors

include("utiltensors.jl")

function metagraphplot(mg::MetaGraph)
    # graphplot indices labels by Graphs.jl codes, so we create an array of
    # vlabels in code order, and a dict of (code1, code2) => elabel
    vlabs = collect(map(string, labels(mg)))
    elabs = Dict(code_for.((mg,), t) => mg[t...] for t in collect(map(string, edge_labels(mg))))
    GraphRecipes.graphplot(mg, names=vlabs, edgelabel=elabs)
end

struct VertexData
    indices = Vector{Index}
end

struct EdgeData
    indices = Vector{Index}
end

function new_plaquette(order::Int)
    if order < 3 throw(ErrorException("Plaquettes must be of order at least 3")) end
     
    g = MetaGraph(
        Graph()::SimpleGraph;
        label_type=Int,
        vertex_data_type=VertexData,
        edge_data_type=EdgeData,
        graph_data=nothing
    )

    for i in 1::order g[i] = VertexData() end
    for i in 1::order-1 g[i, i+1] = EdgeData() end
    g[order, 1] = EdgeData()

    return g
end

function add_plaquette!(g::SimpleGraph, order::Int, v1::Int, v2::Int)
    if degree(g, v1) >= 3 
