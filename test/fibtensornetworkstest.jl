@testset "TensorLabel and fibtensor" begin
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    c1 = Indexlabel(1, :c)
    a2 = IndexLabel(2, :a)

    # TensorLabel construction
    tl1 = TensorLabel(1, [a1, b1, c1])
    @test tl1.group == 1
    @test length(tl1.indices) == 3
    
    # TensorLabel group consistency
    @test_throws "group" TensorLabel(1, [a1, a2])
    
    # FibTensorNetwork construction
    ftn = FibTensorNetwork()

    # add_tensor
    add_tensor!(ftn, tl1)
    @test length(ftn.tensors) == 1
    @test ftn.tensor_with_index[a1] === tl1
    @test ftn.tensor_with_index[b1] === tl1
    @test ftn.tensor_with_index[c1] === tl1
    @test ftn._index_use_count[a1] == 1
    @test ftn._index_use_count[b1] == 1
    @test ftn._index_use_count[c1] == 1

    # duplicate IndexLabel should error
    @test_throws ErrorException add_tensor!(ftn, TensorLabel(1, [a1]))

    # add internal contraction
    ip = IndexPair(a1, b1)
    add_contraction!(ftn, ip)

    @test length(ftn.contractions) == 1
    @test ftn._index_use_count[a1] == 2
    @test ftn._index_use_count[b1] == 2
    @test ftn.contraction_with_index[a1] === ip
    @test ftn.contraction_with_index[b1] == ip

    # cannot contract same indices twice
    @test_throws ErrorException add_contraction!(tn, ip)
    
    # add external contraction
    tl2 = TensorLabel(2, [a2])
    add_tensor!(ftn, tl2)
    ip2 = IndexPair(c1, a2)
    add_contraction!(ftn, ip2)
    
    @test length(ftn.contractions) == 2
    @test ftn._index_use_count[a2] == 2
    @test ftn._index_use_count[c1] == 2
    @test ftn.contraction_with_index[a2] == ip2
    @test ftn.contraction_with_index[c1] == ip2
end
