using FibTN.TensorNetworks
using FibTN.IndexTriplets
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

    # check indices
    @test Set(QubitLattices.indices(ql)) == Set([a1])
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

    # check indices
    @test Set(QubitLattices.indices(ql)) == Set([a1, b2])
end

@testset "QubitLattice multiple indices" begin
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

    # check indices
    @test Set(QubitLattices.indices(ql)) == Set([i1, i2, i3])
end

@testset "QubitLattice extraction basics" begin
    ql = QubitLattice()
    idx = IndexLabel(1, :p)
    add_index!(ql, idx, [7, 9, 1])

    # check that correct values are returned and in the right order
    for val in 1:5
        qs = idxval2qubitvals(ql, idx, val)
        @test (qs[7], qs[9], qs[1]) == split_index(val)
    end

    # add index with shared qubit 7, check that inconsistency errors
    idx2 = IndexLabel(2, :p)
    add_index!(ql, idx2, [7, 4, 2])
    @test_throws ErrorException idxvals2qubitvals(ql, [idx, idx2], [1, 5])
end

@testset "QubitLattice extraction no unpaired" begin
    ql = QubitLattice()
    # tetrahedron projected onto plane
    add_index!(ql, IndexLabel(1, :p), [1, 2, 3])
    add_index!(ql, IndexLabel(4, :p), [2, 4, 6])
    add_index!(ql, IndexLabel(3, :p), [5, 6, 1])
    add_index!(ql, IndexLabel(2, :p), [3, 4, 5])
    inds = [
            IndexLabel(1, :p),
            IndexLabel(4, :p),
            IndexLabel(3, :p),
            IndexLabel(2, :p),
           ]
    test_indices = [
                    [1, 1, 1, 1], # all 0
                    [3, 1, 3, 3], # big triangle
                    [5, 5, 5, 5], # all 1
                    [4, 3, 2, 1], # small triangle
                    [4, 5, 5, 2], # adjacent triangles
                    [4, 4, 3, 2], # quadrilateral
                    [5, 4, 3, 5], # nested triangles
                   ]
    ref_qubitvals = [
                     Dict( # all 0
                          1=>0,
                          2=>0,
                          3=>0,
                          4=>0,
                          5=>0,
                          6=>0,
                         ),
                     Dict( # big triangle
                          1=>1,
                          2=>0,
                          3=>1,
                          4=>0,
                          5=>1,
                          6=>0,
                         ),
                     Dict( # all 1
                          1=>1,
                          2=>1,
                          3=>1,
                          4=>1,
                          5=>1,
                          6=>1,
                         ),
                     Dict( # small triangle
                          1=>1,
                          2=>1,
                          3=>0,
                          4=>0,
                          5=>0,
                          6=>1,
                         ),
                     Dict( # adjacent triangles
                          1=>1,
                          2=>1,
                          3=>0,
                          4=>1,
                          5=>1,
                          6=>1,
                         ),
                     Dict( # quadrilateral
                          1=>1,
                          2=>1,
                          3=>0,
                          4=>1,
                          5=>1,
                          6=>0,
                         ),
                     Dict( # nested triangles
                          1=>1,
                          2=>1,
                          3=>1,
                          4=>1,
                          5=>1,
                          6=>0,
                         ),
                    ]
    for (vals, ref) in zip(test_indices, ref_qubitvals)
        qubitvals = idxvals2qubitvals(ql, inds, vals)
        @test qubitvals == ref
    end
end

@testset "QubitLattice extraction unpaired" begin
    ql = QubitLattice()
    # triangle
    add_index!(ql, IndexLabel(2, :p), [3, 5, 4])
    add_index!(ql, IndexLabel(3, :p), [5, 1, 6])
    add_index!(ql, IndexLabel(1, :p), [1, 3, 2])
    inds = [
            IndexLabel(2, :p),
            IndexLabel(3, :p),
            IndexLabel(1, :p),
           ]
    test_indices = [
                    [1, 1, 1], # all 0
                    [5, 5, 5], # all 1
                   ]
    ref_qubitvals = [
                     Dict( # all 0
                          1=>0,
                          2=>0,
                          3=>0,
                          4=>0,
                          5=>0,
                          6=>0,
                         ),
                     Dict( # all 1
                          1=>1,
                          2=>1,
                          3=>1,
                          4=>1,
                          5=>1,
                          6=>1,
                         ),
                    ]
    for (vals, ref) in zip(test_indices, ref_qubitvals)
        qubitvals = idxvals2qubitvals(ql, inds, vals)
        @test qubitvals == ref
    end
end
