using FibTN.TensorHandles

@testset "IndexLabel basics" begin
    # check construction
    a1 = IndexLabel(1, :a)
    a2 = IndexLabel(2, :a)
    b1 = IndexLabel(1, :b)
    
    a1_2 = IndexLabel(1, :a)
    
    # check equality comparisons
    @test a1 != a2
    @test a1 != b1
    @test a1 == a1_2
    
    # check that they are hashable
    d = Dict(a1 => :val)
    @test d[a1]     = :val
    @test d[a1_2]   = :val
end

@testset "IndexData basics" begin
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)

    # check construction
    ida1_v_1 = IndexData(a1, VIRT, 1)
    idb1_v_1 = IndexData(b1, VIRT, 1)
    ida1_p_1 = IndexData(a1, PHYS, 1)
    ida1_v_2 = IndexData(a1, VIRT, 2)
    
    ida1_v_1_2 = IndexData(a1, VIRT, 1)

    # check equality comparisons
    @test ida1_v_1 != idb1_v_1
    @test ida1_v_1 != ida1_p_1
    @test ida1_v_1 != ida1_v_2
    @test ida1_v_1 == ida1_v_1_2

    # check that they're hashable
    d = Dict(ida1_v_1 => :val)
    @test d[ida1_v_1]   = :val
    @test d[ida1_v_1_2] = :val
end

@testset "IndexPair basics" begin
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    ida1_v_1 = IndexData(a1, VIRT, 1)
    idb1_v_1 = IndexData(b1, VIRT, 1)
    idb1_v_2 = IndexData(b1, VIRT, 2)
    ida1_v_1_2 = IndexData(a1, VIRT, 1)
    
    # check construction
    @test_throws "dimensions" IndexPair(ida1_v_1, idb1_v_2)
    @test_throws "labels" IndexPair(ida1_v_1, ida1_v_1_2)
    ip = IndexPair(ida1_v_1, idb1_v_1)
end

@testset "DummyBackend construction" begin
    struct DummyBackend <: AbstractBackend end
    
    SIZE = 2
    tensor = rand(SIZE, SIZE)
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    index_map = Dict(
                     IndexData(a1, VIRT, SIZE) => 1,
                     IndexData(b1, VIRT, SIZE) => 2,
                    )

    # check that construction works and the values aren't modified
    th = TensorHandle{DummyBackend, typeof(tensor), Int}(tensor, index_map)
    @test th.tensor === tensor
    @test th.index_map === index_map
end

@testset "unimplemented functions" begin
    struct DummyBackend <: AbstractBackend end
    
    a1 = IndexLabel(1, :a)
    b1 = IndexLabel(1, :b)
    ida1 = IndexData(a1, VIRT, 1)
    idb1 = IndexData(b1, VIRT, 1)
    ip = IndexPair(ida1, idb1)

    arr1 = rand(2, 2)
    arr2 = rand(2, 2)
    th1 = TensorHandle{DummyBackend, typeof(arr1), Int}(arr1, Dict())
    th2 = TensorHandle{DummyBackend, typeof(arr2), Int}(arr2, Dict())
    
    # check that if contract isn't implemented for an AbstractBackend subtype, it errors
    @test_throws "DummyBackend" contract(th1, th2, ip)

    # check that if trace isn't implemented it errors
    @test_throws "DummyBackend" trace(th1, ip)
end

