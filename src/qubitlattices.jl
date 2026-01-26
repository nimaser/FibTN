module QubitLattices

struct QubitLattice

    qubits_from_index::Dict{IndexLabel, NTuple{3, Int}}
    adjacent_qubits::Dict{Int, Vector{int}}
    adjacent_tensors::Dict{Int, NTuple{2, Int}}
    function QubitLattice(ftn::FibTensorNetwork)
        
    end
end

; qubit_index_map=Dict{IndexLabel, NTuple{3, Int}}()
    merge!(tn.qubits_from_index, qubit_index_map)
        # check that all indices in qubit index map are in this tensor
    for (k, v) in qubit_index_map
        if k.group != tl.group error("got qubit index map for index not associated with this tensor") end
    end

end # module QubitLattices
