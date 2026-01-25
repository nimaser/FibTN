struct QubitLattice
    adjacent_qubits::Dict{Int, Vector{int}}
    adjacent_tensors::Dict{Int, NTuple{2, Int}}
    function QubitLattice(ftn::FibTensorNetwork)
        
    end
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
