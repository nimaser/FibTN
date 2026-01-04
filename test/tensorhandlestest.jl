using ..TensorHandles

@testset "IndexData basics" begin
    # check that construction works
    id1 = IndexData(1, :a, VIRT, 1)
    id2 = IndexData(1, :a, VIRT, 1)
    id3 = IndexData(2, :a, VIRT, 1)
    id4 = IndexData(1, :b, VIRT, 1)
    id5 = IndexData(1, :a, PHYS, 1)
    id6 = IndexData(1, :a, VIRT, 2)

    # check that these things compare equal as expected
    @test id1 == id2
    @test id1 != id3
    @test id1 != id4
    @test id1 != id5
    @test id1 != id6

    # check that they're hashable and play nice with Dicts
    d = Dict(id1 => :testval)
    @test d[id2] == :testval
end

@testset "ContractionSpec basics" begin
    id1 = IndexData(1, :a, VIRT, 3)
    id2 = IndexData(2, :a, VIRT, 3)
    id3 = IndexData(1, :b, VIRT, 2)
    id4 = IndexData(2, :c, VIRT, 2)
    id5 = IndexData(2, :a, VIRT, 2)
    
    # check that construction works and dimension constraints are checked
    ContractionSpec((id1=>id2, id3=>id4))
    @test_throws "dimension mismatch" ContractionSpec((id1=>id5, id3=>id4))
end

@testset "DummyBackend construction" begin
    struct DummyBackend <: AbstractBackend end
    
    SIZE = 2
    tensor = rand(SIZE, SIZE)
    ids = [
        IndexData(1, :a, VIRT, SIZE),
        IndexData(1, :b, VIRT, SIZE),
    ]
    index_map = Dict(ids[1] => 1, ids[2] => 2)

    # check that construction works and the values aren't modified
    th = TensorHandle{DummyBackend}(tensor, index_map)
    @test th.tensor === tensor
    @test th.index_map === index_map
end

@testset "unimplemented contract" begin
    struct DummyBackend <: AbstractBackend end

    th1 = TensorHandle{DummyBackend}(rand(2,2), Dict())
    th2 = TensorHandle{DummyBackend}(rand(2,2), Dict())
    cs = ContractionSpec(())
    
    # check that if contract isn't implemented for an AbstractBackend subtype, it errors
    @test_throws "DummyBackend" contract(th1, th2, cs)
end

