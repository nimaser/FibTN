@testset "TensorLabel and symbolic network" begin
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    a2 = IndexLabel(2, :a)

    # TensorLabel construction
    tl = TensorLabel(1, [a1, b1])
    @test tl.group == 1
    @test length(tl.indices) == 2
    
    # TensorLabel group consistency
    @test_throws TensorLabel(1, [a1, a2])
    
    # FibTensorNetwork construction
    tn = FibTensorNetwork()

    # add tensor
    add_tensor!(tn, tl)
    @test length(tn.tensors) == 1
    @test tn.tensor_with_index[i1] === tl
    @test tn._index_use_count[i1] == 1

    # duplicate index should error
    tl2 = TensorLabel(2, [i1])
    @test_throws ErrorException add_tensor!(tn, tl2)

    # add contraction
    ip = IndexPair(i1, i2)
    add_contraction!(tn, ip)

    @test length(tn.contractions) == 1
    @test tn._index_use_count[i1] == 2
    @test tn.contraction_with_index[i2] === ip

    # cannot contract twice
    @test_throws ErrorException add_contraction!(tn, ip)
end
