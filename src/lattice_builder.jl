using Graphs
using MetaGraphsNext
using GraphRecipes, Plots
using StringAlgorithms

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
=#

struct Plaquette
    order::Int
    vertices::Array{Tuple{Vararg{Int}}}
    edgesprocessed::Bool
end

function metagraphplot(mg::MetaGraph)
    # graphplot indices labels by Graphs.jl codes, so we create an array of
    # vlabels in code order, and a dict of (code1, code2) => elabel
    vlabs = collect(labels(mg))
    elabs = Dict(code_for.((mg,), t) => mg[t...] for t in collect(edge_labels(mg)))
    GraphRecipes.graphplot(mg, names=vlabs, edgelabel=elabs)
end

### VERTEX PASS ###

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

function makeD(vertices::Vector{Int}, orders::Vector{Int}, edges::Vector{Pair{Int, Int}})
    if length(vertices) != length(orders) throw(ArgumentError("should have one order per vertex")) end
    mg = MetaGraph(
        Graph();
        label_type=Int, # this label is different than the underlying Graph.jl vertex code
        vertex_data_type=Plaquette,
        edge_data_type=Nothing,
        graph_data=emptyL(),
    )
    # add vertices
    for (v, o) in zip(vertices, orders)
        mg[v] = o
    end
    # add edges
    labs = collect(labels(mg))
    for e in edges
        if (e.first ∉ labs || e.second ∉ labs) throw(ArgumentError("invalid vertex label in edge")) end
        mg[[e...]...] = nothing
    end
    mg
end

function makeLvertices(D::MetaGraph)
    # counter/index for all vertices created in L
    vertex_counter = 1
    
    # iterate through L's plaquettes (D's vertice)
    for p in collect(labels(D))
        # max number of vertices we have to work with based on order of p
        vertices_left = D[p].order
        # for each adjacent plaquette, attempt to create two shared Lvertices
        p_nbs = collect(neighbor_labels(D, p))
        for p_nb in p_nbs
            # determine whether our neighbors have neighbors in common with us
            p_nb_nbs = collect(neighbor_labels(D, p_nb))
            common_nbs = intersect(p_nbs, p_nb_nbs)
    
            # create vertices shared between three plaquettes
            p_nb_vertices_created = 0
            for common_nb in common_nbs
                # if the vertex already exists, we will get false
                L_vertex_label = tuple(1, sort([p, p_nb, common_nb])...)
                if add_vertex!(D[], L_vertex_label, vertex_counter)
                    p_nb_vertices_created += 1
                    global vertex_counter += 1
                    # add the vertex to the three plaquettes' collections
                    push!(D[p].vertices, L_vertex_label)
                    push!(D[p_nb].vertices, L_vertex_label)
                    push!(D[common_nb].vertices, L_vertex_label)
                end
                # whether or not we created this vertex, it still has been
                # added to our collection, so we have one less available
                vertices_left -= 1
            end
    
            # create vertices shared between just two plaquettes
            for i in 1:(2-p_nb_vertices_created)
                L_vertex_label = tuple(i, sort([p, p_nb])...)
                if add_vertex!(L, L_vertex_label, vertex_counter)
                    global vertex_counter += 1
                    push!(D[p].vertices, L_vertex_label)
                    push!(D[p_nb].vertices, L_vertex_label)
                end
                vertices_left -= 1
            end
        end
    
        # check that we haven't used too many vertices
        if vertices_left < 0 throw(ErrorException("plaquette $p ran out of vertices")) end
    
        # make any remaining unshared vertices
        for i in 1:vertices_left
            L_vertex_label = (i, p,)
            add_vertex!(L, L_vertex_label, vertex_counter)
            global vertex_counter += 1
        end
    end
end

### EDGE PASS ###

function getdegreeonevertices(mg::MetaGraph)
    [l for l in collect(labels(mg)) if degree(mg, code_for(mg, l)) == 1]
end                                                                                  






