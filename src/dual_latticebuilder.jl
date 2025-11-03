using Graphs
using MetaGraphsNext
using GraphRecipes, Plots

#=
# Consider an undirected graph L = (VL, EL) such that
# - every vertex in VL has a degree of 2 or 3
# - L is planar
#
# Furthermore, let a face f of L be defined by a chordless cycle c in L, and let the
# edges and order of f be the set of edges in and cardinality of c respectively.
#
# Next, let FL be the set of all faces in L, and restrict L such that no two faces in FL
# share more than one edge. Two faces are said to be connected if they share an edge.
#
# Finally, pick some subset PL of FL such that every edge in L belongs to some element of
# PL and call the elements of PL the "plaquettes" of L.
#
# L can represent a lattice realization of a ribbon graph using the Fibonacci input
# category. This file handles generating such a graph L from a graph D which contains
# information about the plaquettes of L and their connectivity, up to shuffling degree
# 2 vertices around a plaquette.
#
# Let D = (VD, ED) be an undirected graph such that
# - D is planar
# - There is an exact correspondence between elements of VD and elements of PL; that is,
#   every plaquette in L has a corresponding node in D, so we can consider elements of VD
#   to be plaquettes
# - There is an exact correspondence between shared edges between plaquettes of PL and
#   elements of ED; that is, every edge shared between two plaquettes in L has a
#   corresponding edge in D, and in particular if plaquettes p1, p2 ∈ PL share an edge
#   eL ∈ EL, the nodes vD1, vD2 ∈ VD which correspond to p1 and p2 will be connected
#   by an edge eD ∈ ED; thus, the connectivity of elements of VD mirrors that of the
#   elements of PL
#
# D must satisfy a certain condition to ensure that the L that is generated has only
# degree 2 and 3 vertices:
# - given any vD ∈ VD, let the order of vD be n
# - given any vD ∈ VD, let the number of plaquettes connected to vD be cp
# - given any vD ∈ VD, let the number of pairs of plaquettes both connected to vD
#   and also connected to each other be t
# - Then any vD ∈ VD must satisfy n ≥ 2*cp - t; this comes because for two plaquettes
#   to be connected, they must share exactly one edge, and thus two vertices, so each
#   plaquette needs to have two vertices per connected plaquette. However, if two
#   plaquettes p1 and p2 both connect to the same third plaquette p3, and they also
#   connect to each other, all three will share one vertex, meaning that to connect
#   p1 and p2 to p3, only 4 - 1 = 3 vertices are required.
#
# We generate L by doing two passes: the first pass will go through the vertices in D and
# generate all vertices in L, along with metadata about which plaquettes they belong to;
# The second pass will iterate through the vertices in D and generate the edges of L by
# connecting the previously generated vertices of L. The main subtlety is making sure
# that the second pass does not produce a nonplanar output: every time the last plaquette
# in a cycle in D is being processed, care has to be taken not to cross the edges that
# are drawn.
#
# There is also an ambiguity in where the vertices of degree 2 are placed: unlike the
# true dual graph of L, D does not have information about every edge between every
# face, only those faces that are plaquettes. However, because of the fusion rules
# of Fibonacci anyons, a degree two vertex can be eliminated just by joining the two
# edges that make it up, as they will always have the same value. Therefore the positions
# of degree 2 vertices don't matter. If the position is important, users can shuffle the
# positions of degree 2 vertices in a third pass.
#
# Finally, there is an ambiguity in whether degree three cycles in D result in one vertex
# shared between three plaquettes or three vertices each shared between two plaquettes. We
# always assume the first case, with the user able to change it in the third pass.
=#

function metagraphplot(mg::MetaGraph)
    # graphplot indices labels by Graphs.jl codes, so we create an array of
    # vlabels in code order, and a dict of (code1, code2) => elabel
    vlabs = collect(map(string, labels(mg)))
    elabs = Dict(code_for.((mg,), t) => mg[t...] for t in collect(edge_labels(mg)))
    GraphRecipes.graphplot(mg, names=vlabs, edgelabel=elabs)
end

### VERTEX PASS ###

mutable struct Plaquette
    order::Int
    vertices::Array{Tuple{Vararg{Int}}}
    edgesprocessed::Bool
end

function findmutualneighbors(mg::MetaGraph, p::Int, p_nb::Int)
    p_nbs = collect(neighbor_labels(mg, p))
    p_nb_nbs = collect(neighbor_labels(mg, p_nb))
    common_nbs = intersect(p_nbs, p_nb_nbs)
end

function emptyL()
    MetaGraph(
        Graph()::SimpleGraph;
        # ordered list of plaquettes this vertex is shared by, first element is an index
        # of vertices shared by those plaquettes
        label_type=Tuple{Vararg{Int}},
        vertex_data_type=Int, # persistent vertex labeling in order they were created
        edge_data_type=Bool, # whether the edge is shared between two plaquettes
        graph_data=nothing,
    )
end

# should accept a list of labels, a list of ints, and a list of pairs
function makeD(vertices::Any, orders::Any, edges::Any)
    if length(vertices) != length(orders) throw(ArgumentError("should have one order per vertex")) end
    if 2 ∈ orders throw(ArgumentError("order-2 plaquettes not supported")) end
    D = MetaGraph(
        Graph();
        label_type=Int, # this label is different than the underlying Graph.jl vertex code
        vertex_data_type=Plaquette,
        edge_data_type=Tuple{Vararg{Int}}, # mutually shared neighbors
        graph_data=emptyL(),
    )
    # add vertices
    for (v, o) in zip(vertices, orders)
        D[v] = Plaquette(o, [], false)
    end
    # add edges
    labs = collect(labels(D))
    for e in edges
        if (e.first ∉ labs || e.second ∉ labs) throw(ArgumentError("invalid vertex label in edge")) end
        D[collect(e)...] = () # empty by default, fill later 
    end
    # label edges with mutual plaquettes - most convenient to do this after the graph is built
    for e in edges
        mutuals = findmutualneighbors(D, e.first, e.second)
        D[collect(e)...] = tuple(sort(mutuals)...)
    end
    D
end

function makeLvertices(D::MetaGraph)
    # counter/index for all vertices created in L
    vertex_counter = 1
    
    # iterate through L's plaquettes (D's vertice)
    for p in collect(labels(D))
        # for each adjacent plaquette, attempt to create two shared Lvertices
        p_nbs = collect(neighbor_labels(D, p))
        for p_nb in p_nbs
            # get the neighbors we've got in common 
            common_nbs = D[p, p_nb]
    
            # create vertices shared between three plaquettes
            p_nb_vertices_created = 0
            for common_nb in common_nbs
                # if the vertex already exists, we will get false
                L_vertex_label = tuple(1, sort([p, p_nb, common_nb])...)
                if add_vertex!(D[], L_vertex_label, vertex_counter)
                    vertex_counter += 1
                    # add the vertex to the three plaquettes' collections
                    push!(D[p].vertices, L_vertex_label)
                    push!(D[p_nb].vertices, L_vertex_label)
                    push!(D[common_nb].vertices, L_vertex_label)
                end
                # whether or not we created this vertex now or previously,
                # it is still shared with p_nb so we shouldn't remake it
                p_nb_vertices_created += 1
            end
    
            # create vertices shared between just two plaquettes
            for i in 1:(2-p_nb_vertices_created)
                L_vertex_label = tuple(i, sort([p, p_nb])...)
                if add_vertex!(D[], L_vertex_label, vertex_counter)
                    vertex_counter += 1
                    push!(D[p].vertices, L_vertex_label)
                    push!(D[p_nb].vertices, L_vertex_label)
                end
            end
        end
        
        vertices_left = D[p].order - length(D[p].vertices)
        # check that we haven't used too many vertices
        if vertices_left < 0 throw(ErrorException("plaquette $p ran out of vertices")) end
        
        # make any remaining unshared vertices
        for i in 1:vertices_left
            L_vertex_label = (i, p,)
            add_vertex!(D[], L_vertex_label, vertex_counter)
            vertex_counter += 1
            push!(D[p].vertices, L_vertex_label)
        end
    end
end

### EDGE PASS ###

function getplaquettesubgraph(D::MetaGraph, p::Int)
    # get codes in L for vertices belonging to p
    pvertices = collect(map(v->code_for(D[], v), D[p].vertices))
    # then get subgraph in L of just the vertices of p
    sg, _ = induced_subgraph(D[], pvertices)
    sg
end

function getunsharedLsubgraph(D::MetaGraph)
    unshared = [e for e in edges(D[]) if !D[][label_for(D[], e.src), label_for(D[], e.dst)]]
    sg, _ = induced_subgraph(D[], unshared)
    sg
end

function completescycle(D::MetaGraph, p::Int)
    # get subgraph of vertices in D which have already been processed
    processed = filter(p->D[p].edgesprocessed, collect(labels(D)))
    @show processed
    if length(processed) == 0 return false end
    sg, vmap = induced_subgraph(D, processed)
    # take all pairs of neighbors, then check if there is a path in sg
    nbs = collect(neighbor_labels(sg, p))
    for nb_pair in Base.product(nbs, nbs)
        if first(nb_pair) == last(nb_pair) continue end
        if has_path(sg, first(nb_pair), last(nb_pair)) return true end
    end
    false
end

function createsharedneighbors(D::MetaGraph, p::Int)
    p_nbs = collect(neighbor_labels(D, p))
    for p_nb in p_nbs
        # v is a tuple of plaquettes the Lvertex belongs to, where the first entry is an index, so it is
        # only shared if entries after the index match
        vs = [v for v in D[p].vertices if length(v) > 1 && p_nb ∈ v[2:end]]
        if length(vs) != 2 throw(ErrorException("more than two vertices shared between plaquettes")) end
        D[][vs...] = true # set label to true as this edge is shared
        # ^ don't worry about overwriting edges that already exist, as the shared edge Bool won't change
    end
end

function attachvertices_floating(D::MetaGraph, p::Int)
    sg = getplaquettesubgraph(D, p)
    floating::Vector{Tuple{Vararg{Int}}} = [l for (i, l) in enumerate(D[p].vertices) if degree(sg, i) == 0]
    if length(floating) == 0 return end
    # if there's only one, attach it to some already attached vertex
    head = floating[1]
    if length(floating) == 1
        attached::Vector{Tuple{Vararg{Int}}} = [l for (i, l) in enumerate(D[p].vertices) if degree(sg, i) == 1]
        D[][head, attached[1]] = false # with only one floating vertex, length(attached) always > 0
        return
    end
    # if there's multiple, chain them together
    for v in floating[2:end]
        D[][head, v] = false # not a shared edge
        head = v
    end
end

function attachvertices_finaltwo(D::MetaGraph, p::Int)
    sg = getplaquettesubgraph(D, p)
    attached::Vector{Tuple{Vararg{Int}}} = [l for (i, l) in enumerate(D[p].vertices) if degree(sg, i) == 1]
    if length(attached) != 2 throw(ErrorException("more than two vertices remain to be attached")) end
    D[][attached[1], attached[2]] = false
    D[p].edgesprocessed = true # indicate that this plaquette is complete
end

function attachvertices_acyclic(D::MetaGraph, p::Int)
    sg = getplaquettesubgraph(D, p)
    attached::Vector{Tuple{Vararg{Int}}} = [l for (i, l) in enumerate(D[p].vertices) if degree(sg, i) == 1]
    # connect pairs of attached vertices, ensuring that attaching them doesn't complete a cycle
    while length(attached) > 2
        sg = getplaquettesubgraph(D, p)
        v = attached[1]
        for (i, candidate) in enumerate(attached[2:end])
            if !has_path(sg, code_for(sg, v), code_for(sg, candidate))
                D[][v, candidate] = false
                deleteat!(attached, [1, i+1])
                break
            end
        end
    end
    attachvertices_finaltwo(D, p)
end

function attachvertices_cyclic(D::MetaGraph, p::Int)
    attached::Vector{Tuple{Vararg{Int}}} = [l for (i, l) in enumerate(D[p].vertices) if degree(sg, i) == 1]
    if length(attached) > 2
        # attempt to make a single connection in this plaquette
        unshared_sg = getunsharedLsubgraph(D)
        plaquette_sg = getplaquettesubgraph(D, p)
        for (i, j) in Base.product(attached, attached)
            # if the vertices are the same, or are reachable via a path along the border of the plaquette,
            # or aren't reachable by a path only traversing unshared edges, don't connect
            if i == j continue end
            if has_path(plaquette_sg, code_for(plaquette_sg, i), code_for(plaquette_sg, j)) continue end
            if !has_path(unshared_sg, code_for(unshared_sg, i), code_for(unshared_sg, j)) continue end
            # otherwise, connect the two
            D[][i, j] = false
            return
        end
    end
    if length(attached) == 2 attachvertices_finaltwo(D, p) end
end

function makeLedges(D::MetaGraph)
    # first pass: process all plaquettes which would not create cycles by doing so
    for p in collect(labels(D))
        @show p
        createsharedneighbors(D, p)
        @show collect(edge_labels(D[]))
        attachvertices_floating(D, p)
        @show collect(edge_labels(D[]))
        if completescycle(D, p) continue end
        attachvertices_acyclic(D, p)
        @show collect(edge_labels(D[]))
    end
    # second pass: process all plaquettes which would have created cycles, trying until all plaquettes have
    # been completed; each plaquette which is unable to be filled in is waiting on others which can be, so
    # there shouldn't be any deadlock
    while true
        unprocessed = filter(p->!D[p].edgesprocessed, collect(labels(D)))
        @show unprocessed
        if length(unprocessed) == 0 break end
        for p in unprocessed
            attachvertices_cyclic(D, p)
            @show collect(edge_labels(D[]))
        end
    end
end
