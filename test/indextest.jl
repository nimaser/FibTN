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

@testset "IndexPair basics" begin
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    a1_2 = IndexLabel(1, :a)
    
    # errors on identical indices
    @test_throws "labels" IndexPair(a1, a1_2)
    
    # IndexPair construction, with ordering enforced
    ip = IndexPair(a1, b1)
    pi = IndexPair(b1, a1)
    @test ip.a == pi.a
    @test ip.b == pi.b
    
    # check that they are hashable
    d = Dict(ip => :val)
    @test d[ip] == :val
    @test d[pi] == :val
end
