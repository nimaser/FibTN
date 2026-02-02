using FibTN.TensorNetworks

@testset "IndexLabel basics" begin
    # IndexLabel construction
    a1 = IndexLabel(1, :a)
    a2 = IndexLabel(2, :a)
    b1 = IndexLabel(1, :b)
    b2 = IndexLabel(2, :b)
    a1_2 = IndexLabel(1, :a)
    
    # check equality comparisons
    @test a1 != a2
    @test a1 != b1
    @test a1 == a1_2
    
    # check ordering
    @test a1 < b1 < a2 < b2
    
    # check that they are hashable
    d = Dict(a1 => :val)
    @test d[a1]     == :val
    @test d[a1_2]   == :val
end

@testset "IndexContraction basics" begin
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    a1_2 = IndexLabel(1, :a)
    
    # errors on identical indices
    @test_throws ArgumentError IndexContraction(a1, a1_2)
    
    # IndexPair construction, with ordering enforced
    ic = IndexContraction(a1, b1)
    ci = IndexContraction(b1, a1)
    @test ic.a == ci.a
    @test ic.b == ci.b
    
    # check that they are hashable
    d = Dict(ic => :val)
    @test d[ic] == :val
    @test d[ci] == :val
end

@testset "TensorLabel basics" begin
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    c1 = IndexLabel(1, :c)
    a2 = IndexLabel(2, :a)

    # TensorLabel construction
    tl1 = TensorLabel(1, [a1, b1, c1])
    @test tl1.group == 1
    @test length(tl1.indices) == 3
    
    # TensorLabel duplicate indices
    @test_throws ArgumentError TensorLabel(1, [a1, b1, c1, b1])
    
    # TensorLabel group consistency
    @test_throws ArgumentError TensorLabel(1, [a1, a2])
end

@testset "IndexLabel TensorLabel regroup" begin
    # IndexLabel regroup
    a1 = IndexLabel(1, :a)
    a2 = IndexLabel(2, :a)
    @test regroup(a1, 2) == a2
    
    # TensorLabel regroup
    b1 = IndexLabel(1, :b)
    b2 = IndexLabel(2, :b)
    tl1 = TensorLabel(1, [a1, b1])
    tl2 = regroup(tl1, 2)
    @test tl2.group == 2
    @test Set(tl2.indices) == Set([a2, b2])
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

    # add_tensor!
    add_tensor!(tn, tl1)
    @test length(tn.tensors) == 1
    @test tn._tensor_with_index[a1] === tl1
    @test tn._tensor_with_index[b1] === tl1
    @test tn._tensor_with_index[c1] === tl1
    
    # sanity check: IndexLabels act by value not by object identity
    @test tn._tensor_with_index[IndexLabel(1, :a)] == tn._tensor_with_index[a1]

    # duplicate IndexLabel should error
    @test_throws ArgumentError add_tensor!(tn, TensorLabel(1, [a1]))

    # same tensor add_contraction!
    ic = IndexContraction(a1, b1)
    add_contraction!(tn, ic)

    @test length(tn.contractions) == 1
    @test tn._contraction_with_index[a1] === ic
    @test tn._contraction_with_index[b1] == ic
    
    # cannot contract index not in network
    @test_throws ArgumentError add_contraction!(tn, IndexContraction(a1, a2))

    # cannot contract same indices twice
    @test_throws ArgumentError add_contraction!(tn, ic)
    
    # add_tensor! again
    add_tensor!(tn, tl2)
    @test length(tn.tensors) == 2
    @test tn._tensor_with_index[a1] === tl1
    @test tn._tensor_with_index[b1] === tl1
    @test tn._tensor_with_index[c1] === tl1
    @test tn._tensor_with_index[a2] === tl2
    
    # different tensors add_contraction!
    ic2 = IndexContraction(c1, a2)
    add_contraction!(tn, ic2)
    
    @test length(tn.contractions) == 2
    @test tn._contraction_with_index[a2] == ic2
    @test tn._contraction_with_index[c1] == ic2

    # check getters
    @test Set(get_groups(tn)) == Set([1, 2])
    @test Set(get_indices(tn)) == Set([a1, b1, c1, a2])
    @test get_tensor(tn, 1) == tl1
    @test get_tensor(tn, 2) == tl2
    @test get_tensor(tn, a1) == tl1
    @test get_tensor(tn, a2) == tl2
    @test get_contraction
    @test get_contraction(tn, a2) == ic2
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
    
    # find generic
    @test Set(find_indices(tn) do idx idx.group == 2 || idx.group == 3 end) == Set([a2, b3])
    
    # find by group
    @test Set(find_indices(tn, 1)) == Set([a1, b1, c1])
    @test Set(find_indices(tn, 2)) == Set([a2])
    @test Set(find_indices(tn, 3)) == Set([b3])
    
    # find by port
    @test Set(find_indices(tn, :a)) == Set([a1, a2])
    @test Set(find_indices(tn, :b)) == Set([b1, b3])
    @test Set(find_indices(tn, :c)) == Set([c1])
    
    # has_index
    @test has_index(tn, 1, :a) == true
    @test has_index(tn, 2, :b) == false
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
    
    add_contraction!(tn, IndexContraction(a1, a2))
    add_contraction!(tn, IndexContraction(b1, b3))
    add_contraction!(tn, IndexContraction(c1, c4))
    
    @test length(tn.contractions) == 3
    # no specific tests here, just another construction check
end
