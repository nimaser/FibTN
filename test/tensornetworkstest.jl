using FibTN.Indices
using FibTN.TensorNetworks

@testset "TensorLabel and TensorNetwork" begin
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
    
    # TensorNetwork construction
    tn = TensorNetwork()

    # add_tensor
    add_tensor!(tn, tl1)
    @test length(tn.tensors) == 1
    @test tn.tensor_with_index[a1] === tl1
    @test tn.tensor_with_index[b1] === tl1
    @test tn.tensor_with_index[c1] === tl1
    @test tn._index_use_count[a1] == 1
    @test tn._index_use_count[b1] == 1
    @test tn._index_use_count[c1] == 1

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
    tl2 = TensorLabel(2, [a2])
    add_tensor!(tn, tl2)
    ip2 = IndexPair(c1, a2)
    add_contraction!(tn, ip2)
    
    @test length(tn.contractions) == 2
    @test tn._index_use_count[a2] == 2
    @test tn._index_use_count[c1] == 2
    @test tn.contraction_with_index[a2] == ip2
    @test tn.contraction_with_index[c1] == ip2
end
