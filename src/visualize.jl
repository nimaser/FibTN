using GLMakie

struct FibTensorNetworkLayoutSpec
    groups::Vector{Int}
    positions::Vector{NTuple{2, Float64}}
    colors::Vector{Symbol}
    markers::Vector{Symbol}
end

function visualize(ftn::FibTensorNetwork, ftnls::FibTensorNetworkLayoutSpec, ax::Axis)
    scatter!(ax, ftnls.positions, color=ftnls.colors, marker=ftnls.markers)
    for c in ftn.contractions
        pos1 = ftnls.positions[c.a.group]
        pos2 = ftnls.positions[c.b.group]
        lines!(ax, [pos1, pos2])
    end
    f
end

struct QubitLatticeLayoutSpec
    node_positions::Dict{Int, Tuple{Int, Int}}
    node_colors::Dict{Int, Symbol}
    edge_colors::Dict{Int, Symbol}
end

function visualize(ql::QubitLattice, qlls::QubitLatticeLayoutSpec)
    # plot the lattice connectivity graph in ql, using qlls for the styling and to
    # change the color of the graph edges to denote the qubit states |0> and |1>
end
