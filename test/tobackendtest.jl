using FibTN.TensorNetworks
using FibTN.TOBackend

using SparseArrayKit, TensorOperations

@testset "ExecutionState basics" begin
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    a2 = IndexLabel(2, :a)
    b2 = IndexLabel(2, :b)

    tl1 = TensorLabel(1, [a1, b1])
    tl2 = TensorLabel(2, [a2, b2])

    tn = TensorNetwork()
    add_tensor!(tn, tl1)
    add_tensor!(tn, tl2)
    
    # check construction
    A1 = [1.0 2.0; 3.0 4.0]
    A2 = [5.0 6.0; 7.0 8.0]
    es = ExecutionState(tn, Dict(1 => A1, 2 => A2))
    @test length(es.tensor_from_id) == 2
    @test en._next_id == 3

    # check indices and data
    et1, et2 = es.tensor_from_id[1], es.tensor_from_id[2]
    @test et1.indices == [a1, b1]
    @test et2.indices == [a2, b2]
    @test et1.data == A1
    @test et2.data == A2
    
    # get
    @test Set(get_ids(es)) == Set([1, 2])
    @test Set(get_indices(es)) == Set([a1, b1, a2, b2])
    @test get_tensors(es, 1) == [et1]
    @test get_tensors(es, 2) == [et2]
    @test get_tensor(es, IndexLabel(1, :a)) == et1
    @test get_tensor(es, IndexLabel(2, :a)) == et2
end

@testset "ExecutionState single contraction" begin
    # A[i,j] * B[j,k] -> C[i,k]
    i = IndexLabel(1, :i)
    j1 = IndexLabel(1, :j)
    j2 = IndexLabel(2, :j)
    k = IndexLabel(2, :k)

    tlA = TensorLabel(1, [i, j1])
    tlB = TensorLabel(2, [j2, k])

    tn = TensorNetwork()
    add_tensor!(tn, tlA)
    add_tensor!(tn, tlB)
    add_contraction!(tn, IndexPair(j1, j2))

    A = [1.0 2.0; 3.0 4.0]
    B = [5.0 6.0; 7.0 8.0]

    es = ExecutionState(tn, Dict(1 => A, 2 => B))
    @test length(es.tensor_from_id) == 2
    @test es._next_id == 3

    # execute contraction
    execute_step!(es, ContractionStep(IndexContraction(j1, j2)))
    @test length(es.tensor_from_id) == 1
    @test es._next_id == 4

    # check result
    et = es.tensor_from_id[only(get_ids(es))]
    @test et.id == 3
    @test et.groups == Set([1, 2])
    @test et.indices == [i, k]
    
    # check calculation via @tensor
    @tensor C[a,c] := A[a,b] * B[b,c]
    @test Array(et.data) ≈ C
end

@testset "ExecutionState multiple contraction" begin
    # A[i,j] * B[j,k] * C[k,l] -> D[i,l]
    i = IndexLabel(1, :i)
    j1 = IndexLabel(1, :j)
    j2 = IndexLabel(2, :j)
    k1 = IndexLabel(2, :k)
    k2 = IndexLabel(3, :k)
    l = IndexLabel(3, :l)

    tlA = TensorLabel(1, [i, j1])
    tlB = TensorLabel(2, [j2, k1])
    tlC = TensorLabel(3, [k2, l])

    tn = TensorNetwork()
    add_tensor!(tn, tlA)
    add_tensor!(tn, tlB)
    add_tensor!(tn, tlC)

    add_contraction!(tn, IndexContraction(j1, j2))
    add_contraction!(tn, IndexContraction(k1, k2))

    A = [1.0 2.0; 3.0 4.0]
    B = [5.0 6.0; 7.0 8.0]
    C = [9.0 0.0; 1.0 2.0]

    es = ExecutionStsate(tn, Dict(1 => A, 2 => B, 3 => C))
    @test length(es.tensor_from_id) == 3
    @test es._next_id == 4
    # first contraction
    execute_step!(es, ContractionStep(IndexContraction(j1, j2)))
    @test length(es.tensor_from_id) == 2
    @test es._next_id == 5
    # second contraction
    execute_step!(es, ContractionStep(IndexContraction, k2))
    @test length(es.tensor_from_id) == 1
    @test es._next_id == 6
    # check result
    et = es.tensor_from_id[only(get_ids(es))]
    @test et.id == 5
    @test et.groups == Set([1, 2, 3])
    @test et.indices == [i, l]
    # check calculation via @tensor
    @tensor D[a,d] := A[a,b] * B[b,c] * C[c,d]
    @test Array(et.data) ≈ D
end
