using FibTN.Indices
using FibTN.QubitLattices
using Graphs

@testset "QubitLattice construction" begin
    ql = QubitLattice()
    @test isempty(ql.qubits_from_index)
    @test isempty(ql._node_from_index)
    @test isempty(ql.indices_from_qubit)
    @test isempty(ql._edge_from_qubit)
    @test isempty(ql._unpaired_qubits)
    @test nv(ql.graph) == 0
    @test ne(ql.graph) == 0
end

@testset "QubitLattice index basics" begin
    ql = QubitLattice()
    a1 = IndexLabel(1, :a)
    qubits = [1, 2, 3]
    
    # test that adding duplicate qubits causes an error
    @test_throws "duplicate" add_index!(ql, a1, [1, 1, 2])
    @test nv(ql.graph) == 0
    
    # index and qubits are added
    add_index!(ql, a1, qubits)
    @test ql.qubits_from_index[a1] == qubits
    @test nv(ql.graph) == 1
    @test ne(ql.graph) == 0
    
    @test ql._node_from_index[a1] == 1
    
    # check qubit -> index mapping
    @test ql.indices_from_qubit[1] == [a1]
    @test ql.indices_from_qubit[2] == [a1]
    @test ql.indices_from_qubit[3] == [a1]
    
    # check unpaired
    @test Set(ql._unpaired_qubits) == Set(qubits)
    
    # check that duplicating an index causes an error
    @test_throws ErrorException add_index!(ql, a1, qubits)
    @test nv(ql.graph) == 1
end

@testset "QubitLattice pairing basics" begin
    ql = QubitLattice()
    a1 = IndexLabel(1, :a)
    b2 = IndexLabel(2, :b)
    
    # index with qubits [1, 2, 3]
    add_index!(ql, a1, [1, 2, 3])
    @test ql.qubits_from_index[a1] == [1, 2, 3]
    @test nv(ql.graph) == 1
    @test ne(ql.graph) == 0
    
    # index with qubits [3, 4, 5]
    add_index!(ql, b2, [3, 4, 5])
    @test ql.qubits_from_index[b2] == [3, 4, 5]
    @test nv(ql.graph) == 2
    @test ne(ql.graph) == 1
    
    @test ql._node_from_index[a1] == 1
    @test ql._node_from_index[b2] == 2
    
    # check edges are correct
    @test has_edge(ql.graph, 1, 2)
    
    # check qubit -> edge mapping
    @test ql._edge_from_qubit[3] == Edge(1, 2) || ql._edge_from_qubit[3] == Edge(2, 1)
    
    # check qubit -> index mapping
    @test Set(ql.indices_from_qubit[3]) == Set([a1, b2])
    @test Set(ql.indices_from_qubit[1]) == Set([a1])
    @test Set(ql.indices_from_qubit[5]) == Set([b2])
    
    # check unpaired
    @test Set(ql._unpaired_qubits) == Set([1, 2, 4, 5])
end

@testset "QubitLattice multiple qubits" begin
    ql = QubitLattice()
    i1 = IndexLabel(1, :q)
    i2 = IndexLabel(2, :q)
    i3 = IndexLabel(3, :q)
    
    # create a triangle
    add_index!(ql, i1, [1, 2, 4])
    add_index!(ql, i2, [2, 3, 5])
    add_index!(ql, i3, [3, 1, 6])
    
    @test ql.qubits_from_index[i1] == [1, 2, 4]
    @test ql.qubits_from_index[i2] == [2, 3, 5]
    @test ql.qubits_from_index[i3] == [3, 1, 6]
    
    @test ql._node_from_index[i1] == 1
    @test ql._node_from_index[i2] == 2
    @test ql._node_from_index[i3] == 3
    
    @test nv(ql.graph) == 3
    @test ne(ql.graph) == 3
    
    # check edges are correct
    @test has_edge(ql.graph, 1, 2)
    @test has_edge(ql.graph, 2, 3)
    @test has_edge(ql.graph, 3, 1)
    
    # check qubit -> edge mapping
    @test ql._edge_from_qubit[2] == Edge(1, 2) || ql._edge_from_qubit[2] == Edge(2, 1)
    @test ql._edge_from_qubit[3] == Edge(2, 3) || ql._edge_from_qubit[3] == Edge(3, 2)
    @test ql._edge_from_qubit[1] == Edge(1, 3) || ql._edge_from_qubit[1] == Edge(3, 1)
    
    # check qubit -> index mapping
    @test Set(ql.indices_from_qubit[2]) == Set([i1, i2])
    @test Set(ql.indices_from_qubit[3]) == Set([i2, i3])
    @test Set(ql.indices_from_qubit[1]) == Set([i3, i1])
    
    # check unpaired
    @test Set(ql._unpaired_qubits) == Set([4, 5, 6])
end
