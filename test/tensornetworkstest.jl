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
end

@testset "TensorNetwork get" begin
    # A[a1, b1, c1] * B[a3] * C[b5] with (a1, a3) and (b1, b5) contracted
    tn = TensorNetwork()
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    c1 = IndexLabel(1, :c)
    a3 = IndexLabel(3, :a)
    b5 = IndexLabel(5, :b)
    
    tl1 = TensorLabel(1, [a1, b1, c1])
    tl3 = TensorLabel(3, [a3])
    tl5 = TensorLabel(5, [b5])
    
    add_tensor!(tn, tl1)
    add_tensor!(tn, tl3)
    add_tensor!(tn, tl5)
    
    ic1 = IndexContraction(a1, a3)
    ic2 = IndexContraction(b1, b5)
    
    add_contraction!(tn, ic1)
    add_contraction!(tn, ic2)
    
    # get groups and indices
    @test Set(get_groups(tn)) == Set([1, 3, 5])
    @test Set(get_indices(tn)) == Set([a1, b1, c1, a3, b5])
    
    # get tensor
    @test get_tensor(tn, 1) == tl1
    @test get_tensor(tn, 3) == tl3
    @test get_tensor(tn, 5) == tl5
    @test get_tensor(tn, a1) == tl1
    @test get_tensor(tn, a3) == tl3
    @test get_tensor(tn, b5) == tl5
    
    # get contraction
    @test get_contraction(tn, a1) == ic1
    @test get_contraction(tn, a3) == ic1
    @test get_contraction(tn, b1) == ic2
    @test get_contraction(tn, b5) == ic2
end

@testset "TensorNetwork find" begin
    # A[a1, b1, c1] * B[a3] * C[b5] with (a1, a3) and (b1, b5) contracted
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
    
    ic1 = IndexContraction(a1, a3)
    ic2 = IndexContraction(b1, b5)
    
    add_contraction!(tn, ic1)
    add_contraction!(tn, ic2)
    
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

@testset "TensorNetwork remove" begin
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
    
    add_contraction!(tn, ic1 = IndexContraction(a1, a2))
    add_contraction!(tn, ic2 = IndexContraction(b1, b3))
    add_contraction!(tn, ic3 = IndexContraction(c1, c4))
    @test length(tn.contractions) == 3
    
    # can't remove tensor with contractions
    @test_throws ArgumentError remove_tensor!(tn, tl1)
    
    # can remove contractions on a tensor
    remove_contraction!(tn, ic1)
    @test_throws get_contraction(tn, a2)
    
    # can remove uncontracted tensors
    remove_tensor!(tn, tl2)
    @test_throws get_tensor(tn, a2)
    
    # can remove all contractions on a tensor
    remove_contractions!(tn, tl1)
    @test_throws get_contraction(tn, b1)
    @test_throws get_contraction(tn, c1)
    remove_tensor!(tn, tl1)
end

@testset "TensorNetwork combine!" begin
    # length 4 MPS-like chain
    a1 = IndexLabel(1, :a)
    a2 = IndexLabel(2, :a)
    a3 = IndexLabel(3, :a)
    a4 = IndexLabel(4, :a)
    
    r1 = IndexLabel(1, :r)
    l2 = IndexLabel(2, :l)
    r2 = IndexLabel(2, :r)
    l3 = IndexLabel(3, :l)
    r3 = IndexLabel(3, :r)
    l4 = IndexLabel(4, :l)
    
    tl1 = TensorLabel(1, [a1, r1])
    tl2 = TensorLabel(2, [a2, l2, r2])
    tl3 = TensorLabel(3, [a3, l3, r3])
    tl4 = TensorLabel(4, [a4, l4])
    
    tn1 = TensorNetwork()
    add_tensor!(tn1, tl1)
    add_tensor!(tn1, tl2)
    add_tensor!(tn1, tl3)
    add_tensor!(tn1, tl4)
    
    add_contraction!(tn1, ic1 = IndexContraction(r1, l2))
    add_contraction!(tn1, ic2 = IndexContraction(r2, l3))
    add_contraction!(tn1, ic3 = IndexContraction(r3, l4))
    
    # combine!
    tn2 = deepcopy(tn1)
    gmap = combine!(tn2, tn1)
    @test length(tn2.tensors) == 2 * length(tn1.tensors)
    @test length(tn2.contractions) == 2 * length(tn1.contractions)
    
    # matchcombine!
    tn3 = deepcopy(tn1)
    gmap = matchcombine!(tn3, tn2)
    @test length(tn3.tensors) == 3 * length(tn1.tensors)
    @test length(tn3.contractions) == 3 * length(tn1.contractions) + 4
end
