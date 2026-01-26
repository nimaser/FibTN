using FibTN.Indices
using FibTN.TensorNetworks

@testset "TensorLabel basics" begin
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    c1 = IndexLabel(1, :c)
    a2 = IndexLabel(2, :a)

    # TensorLabel construction
    tl1 = TensorLabel(1, [a1, b1, c1])
    @test tl1.group == 1
    @test length(tl1.indices) == 3
    
    # TensorLabel group consistency
    @test_throws "group" TensorLabel(1, [a1, a2])
end

@testset "TensorNetwork construction" begin
    tn = TensorNetwork()
    @test length(tn.tensors) == 0
    @test length(tn.contractions) == 0
end

@testset "TensorNetwork basics" begin
    # A[a1, b1, c1] * B[2a] with (a1, a2) and (b1, c1) contracted
    tn = TensorNetwork()
    
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    c1 = IndexLabel(1, :c)
    a2 = IndexLabel(2, :a)
    
    tl1 = TensorLabel(1, [a1, b1, c1])
    tl2 = TensorLabel(2, [a2])

    # add_tensor
    add_tensor!(tn, tl1)
    @test length(tn.tensors) == 1
    @test tn.tensor_with_index[a1] === tl1
    @test tn.tensor_with_index[b1] === tl1
    @test tn.tensor_with_index[c1] === tl1
    @test tn._index_use_count[a1] == 1
    @test tn._index_use_count[b1] == 1
    @test tn._index_use_count[c1] == 1
    
    # IndexLabels should act by value not by object identity
    @test tn.tensor_with_index[IndexLabel(1, :a)] == tn.tensor_with_index[a1]

    # duplicate IndexLabel should error
    @test_throws ErrorException add_tensor!(tn, TensorLabel(1, [a1]))

    # add internal contraction
    ip = IndexPair(a1, b1)
    add_contraction!(tn, ip)

    @test length(tn.contractions) == 1
    @test tn._index_use_count[a1] == 2
    @test tn._index_use_count[b1] == 2
    @test tn.contraction_with_index[a1] === ip
    @test tn.contraction_with_index[b1] == ip

    # cannot contract same indices twice
    @test_throws ErrorException add_contraction!(tn, ip)
    
    # add external contraction
    add_tensor!(tn, tl2)
    ip2 = IndexPair(c1, a2)
    add_contraction!(tn, ip2)
    
    @test length(tn.contractions) == 2
    @test tn._index_use_count[a2] == 2
    @test tn._index_use_count[c1] == 2
    @test tn.contraction_with_index[a2] == ip2
    @test tn.contraction_with_index[c1] == ip2
end

@testset "TensorNetwork multiple contractions" begin
    # star topology: one central tensor connected to three others
    # C[center1, a1, b1, c1] * A[a2] * B[b3] * C[c4]
    tn = TensorNetwork()
    
    center1 = IndexLabel(1, :center)
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    c1 = IndexLabel(1, :c)
    a2 = IndexLabel(2, :a)
    b3 = IndexLabel(3, :b)
    c4 = IndexLabel(4, :c)
    
    tl1 = TensorLabel(1, [center1, a1, b1, c1])
    tl2 = TensorLabel(2, [a2])
    tl3 = TensorLabel(3, [b3])
    tl4 = TensorLabel(4, [c4])
    
    add_tensor!(tn, tl1)
    add_tensor!(tn, tl2)
    add_tensor!(tn, tl3)
    add_tensor!(tn, tl4)
    
    @test length(tn.tensors) == 4
    
    add_contraction!(tn, IndexPair(a1, a2))
    add_contraction!(tn, IndexPair(b1, b3))
    add_contraction!(tn, IndexPair(c1, c4))
    
    @test length(tn.contractions) == 3
    @test tn._index_use_count[a1] == 2
    @test tn._index_use_count[center1] == 1
end

@testset "TensorNetwork find" begin
    tn = TensorNetwork()
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    c1 = IndexLabel(1, :c)
    a2 = IndexLabel(2, :a)
    b3 = IndexLabel(3, :b)
    
    tl1 = TensorLabel(1, [a1, b1, c1])
    tl2 = TensorLabel(2, [a2])
    tl3 = TensorLabel(3, [b3])
    
    add_tensor!(tn, tl1)
    add_tensor!(tn, tl2)
    add_tensor!(tn, tl3)
    
    @test Set(find_indices(tn) do idx idx.group == 2 || idx.group == 3 end) == Set([a2, b3])
    
    @test Set(find_indices_by_group(tn, 1)) == Set([a1, b1, c1])
    @test Set(find_indices_by_group(tn, 2)) == Set([a2])
    @test Set(find_indices_by_group(tn, 3)) == Set([b3])
    
    @test Set(find_indices_by_port(tn, :a)) == Set([a1, a2])
    @test Set(find_indices_by_port(tn, :b)) == Set([b1, b3])
    @test Set(find_indices_by_port(tn, :c)) == Set([c1])
    
    @test find_index(tn, 1, :a) == IndexLabel(1, :a) == a1
    @test_throws ErrorException find_index(tn, 2, :b)
end
