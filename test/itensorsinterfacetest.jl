using ..TensorHandles
using ..ITensorsInterface
using ITensors

@testset "ITensorsBackend constructor" begin
    data = rand(2, 3)
    ids = [
        IndexData(1, :a, VIRT, 2),
        IndexData(1, :b, VIRT, 3),
    ]

    th = TensorHandle(ITensorsBackend, data, ids)

    @test th isa TensorHandle{ITensorsBackend}
    @test length(th.index_map) == length(ids)

    for id in ids
        @test haskey(th.index_map, id)
        @test dim(th.index_map[id]) == id.dim
    end
end

@testset "ITensorsBackend index bookkeeping after contraction" begin
    ids1 = [
        IndexData(1, :a, VIRT, 2),
        IndexData(1, :b, VIRT, 2),
    ]
    ids2 = [
        IndexData(2, :a, VIRT, 2),
        IndexData(2, :c, VIRT, 2),
    ]

    t1 = TensorHandle(ITensorsBackend, rand(2,2), ids1)
    t2 = TensorHandle(ITensorsBackend, rand(2,2), ids2)

    cs = ContractionSpec([
        ids1[1] => ids2[1],
    ])

    tnew = contract(t1, t2, cs)

    @test !haskey(tnew.index_map, ids1[1])
    @test !haskey(tnew.index_map, ids2[1])
    @test haskey(tnew.index_map, ids1[2])
    @test haskey(tnew.index_map, ids2[2])
end
