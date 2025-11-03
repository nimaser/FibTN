using Graphs
using MetaGraphsNext
using GraphRecipes, Plots
using StringAlgorithms

include("tensor_init.jl")
include("lattice_builder.jl")

function test1()
    D = makeD([1], [6], [])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test2()
    D = makeD([1, 2], [3, 4], [])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test3()
    D = makeD([1, 2], [3, 4], [1=>2])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test4()
    D = makeD([1, 2, 3, 4], [3, 4, 5, 3], [1=>2, 2=>3, 3=>4])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test5()
    D = makeD([1, 2, 3], [3, 4, 5], [1=>2, 2=>3, 3=>1])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test6()
    D = makeD([1, 2, 3, 4], [6, 6, 6, 6], [1=>2, 2=>3, 3=>4, 4=>1, 1=>3])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test7()
    D = makeD([1, 2, 3, 4, 5], [4, 4, 4, 4, 4], [1=>2, 1=>3, 1=>5, 4=>2, 4=>3, 4=>5, 5=>2, 2=>3])
    makeLvertices(D)
    makeLedges(D)
    D
end

function completescycle(D::MetaGraph, p::Int)
    #
end

function attachvertices_cycle(D::MetaGraph, p::Int)
    # get codes in L for vertices belonging to p
    # then get subgraph of just the vertices of p in L
    pvertices = collect(map(v->code_for(D[], v), D[p].vertices))
    sg, vmap = induced_subgraph(D[], pvertices)
    # find floating vertices and chain them
    floating::Vector{Tuple{Vararg{Int}}} = [l for (i, l) in enumerate(D[p].vertices) if degree(sg, i) == 0]
    attached::Vector{Tuple{Vararg{Int}}} = [l for (i, l) in enumerate(D[p].vertices) if degree(sg, i) == 1]
    if length(floating) > 0
        tail = floating[1]
        head = floating[1]
        for v in floating[2:end]
            D[][head, v] = false # not shared
            head = v
        end
        # attach single floating vertex to some other vertex
        if head == tail
            D[][head, attached[1]] = false
        end
    end
    # regenerate subgraph to take into account newly added edges
    pvertices = collect(map(v->code_for(D[], v), D[p].vertices))
    sg, vmap = induced_subgraph(D[], pvertices)
    # find attached and complete vertices
    floating = [l for (i, l) in enumerate(D[p].vertices) if degree(sg, i) == 0]
    attached = [l for (i, l) in enumerate(D[p].vertices) if degree(sg, i) == 1]
    complete::Vector{Tuple{Vararg{Int}}} = [l for (i, l) in enumerate(D[p].vertices) if degree(sg, i) == 2]
    # connect pairs of attached vertices, ensuring that attaching them doesn't complete a cycle
    while length(attached) > 2
        # recompute subgraph again
        sg, vmap = induced_subgraph(D[], pvertices)
        v = attached[1]
        for (i, candidate) in enumerate(attached[2:end])
            if !has_path(sg, code_for(sg, v), code_for(sg, candidate))
                D[][v, candidate] = false
                deleteat!(attached, [1, i+1])
                append!(complete, [v, candidate])
                break
            end
        end
    end
    # every vertex is complete except for the last two attached vertices; attach them
    D[][attached[1], attached[2]] = false
    append!(complete, attached)
    # check that we have the right number of vertices
    if length(complete) != D[p].order throw(ErrorException("plaquette $p: order != complete")) end
end

function generateTN(L::MetaGraph)
    # we will use these labels as index names
end







