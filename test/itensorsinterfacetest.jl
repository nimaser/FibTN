using FibTN.TensorHandles
using FibTN.ITensorsInterface
using ITensors

@testset "ITensorsBackend constructor" begin
    data = rand(2, 3)
    ids = [
        IndexData(IndexLabel(1, :a), VIRT, 2),
        IndexData(IndexLabel(1, :b), VIRT, 3),
    ]
    th = TensorHandle(ITensorsBackend, data, ids)

    # check construction worked and index map has the correct length
    @test th isa TensorHandle{ITensorsBackend}
    @test length(ids) == length(th.index_map) == length(inds(th.tensor))

    # check that each index has the correct dims and was put into the dictionary properly
    for id in ids
        @test haskey(th.index_map, id)
        @test dim(th.index_map[id]) == id.dim
    end
    
    # check that constructor fails if the wrong number of indices are provided
    @test_throws DimensionMismatch TensorHandle(ITensorsBackend, data, [ids[1]])
end

@testset "mismatched backends" begin
    struct OtherBackend <: AbstractBackend end
    
    arr1 = rand(2, 2)
    arr2 = rand(2, 2)
    th1 = TensorHandle{ITensorsBackend, typeof(arr1), Dict{IndexData, Any}}(arr1, Dict{IndexData, Any}())
    th2 = TensorHandle{OtherBackend, typeof(arr2), Dict{IndexData, Any}}(arr2, Dict{IndexData, Any}())
    
    ip = IndexPair(IndexData(IndexLabel(1, :a), VIRT, 2), IndexData(IndexLabel(1, :b), VIRT, 2))
    
    # make sure that using mismatched backends causes a problem
    @test_throws UndefVarError contract(th1, th2, ip)
end

@testset "ITensorsBackend single contract" begin
    inds1 = [
        IndexData(IndexLabel(1, :a), VIRT, 2),
        IndexData(IndexLabel(1, :b), VIRT, 2),
    ]
    inds2 = [
        IndexData(IndexLabel(2, :a), VIRT, 2),
        IndexData(IndexLabel(2, :c), VIRT, 2),
    ]
    arr1 = [1 2; 1 2]
    arr2 = [2 1; 4 2]
    res = [10 5; 10 5]
    th1 = TensorHandle(ITensorsBackend, arr1, inds1)
    th2 = TensorHandle(ITensorsBackend, arr2, inds2)
    th3 = ITensorsInterface.contract(th1, th2, IndexPair(inds1[2], inds2[1]))

    # check that the new tensor has the proper indices and data
    @test length(th3.index_map) == 2
    @test !haskey(th3.index_map, inds1[2])
    @test !haskey(th3.index_map, inds2[1])
    @test haskey(th3.index_map, inds1[1])
    @test haskey(th3.index_map, inds2[2])
    @test Array(th3.tensor, th3.index_map[inds1[1]], th3.index_map[inds2[2]]) == res
    
    # check that th1 and th2 were not mutated:
    @test length(th1.index_map) == length(th2.index_map) == 2
    @test Array(th1.tensor, th1.index_map[inds1[1]], th1.index_map[inds1[2]]) == arr1
    @test Array(th2.tensor, th2.index_map[inds2[1]], th2.index_map[inds2[2]]) == arr2
end

@testset "tensor trace" begin
    inds = [
        IndexData(IndexLabel(1, :a), VIRT, 2),
        IndexData(IndexLabel(1, :b), VIRT, 2),
    ]
    arr = [1 2; 2 3]
    res = 4
    th1 = TensorHandle(ITensorsBackend, arr, inds)
    th2 = ITensorsInterface.trace(th1, IndexPair(inds[1], inds[2]))
    
    # make sure the trace occurred properly
    @test length(th2.index_map) == 0
    @test Array(th2.tensor)[] == res
end

@testset "ITensorsBackend multiple contract by trace" begin
    inds1 = [
        IndexData(IndexLabel(1, :a), VIRT, 2),
        IndexData(IndexLabel(1, :b), VIRT, 2),
        IndexData(IndexLabel(1, :c), VIRT, 2),
    ]
    inds2 = [
        IndexData(IndexLabel(2, :a), VIRT, 2),
        IndexData(IndexLabel(2, :b), VIRT, 2),
    ]
    arr1 = cat([1 1; 1 1], [2 2; 2 2], dims=3)
    arr2 = [1 2; 3 4]
    res = [10, 20]
    th1 = TensorHandle(ITensorsBackend, arr1, inds1)
    th2 = TensorHandle(ITensorsBackend, arr2, inds2)
    
    th3 = ITensorsInterface.contract(th1, th2, IndexPair(inds1[1], inds2[1]))
    th4 = ITensorsInterface.trace(th3, IndexPair(inds1[2], inds2[2]))
    
    # check that the new tensor has the proper indices and data
    @test length(th4.index_map) == 1
    @test !haskey(th4.index_map, inds1[1])
    @test !haskey(th4.index_map, inds1[2])
    @test !haskey(th4.index_map, inds2[1])
    @test !haskey(th4.index_map, inds2[2])
    @test haskey(th4.index_map, inds1[3])
    @test Array(th4.tensor, th4.index_map[inds1[3]]) == res
    
    # check that th1 and th2 were not mutated:
    @test length(th1.index_map) == 3
    @test length(th2.index_map) == 2
    @test Array(th1.tensor, th1.index_map[inds1[1]], th1.index_map[inds1[2]], th1.index_map[inds1[3]]) == arr1
    @test Array(th2.tensor, th2.index_map[inds2[1]], th2.index_map[inds2[2]]) == arr2
end

@testet "ITensorsBackend multiple contract" begin
# TODO
end

@testset "invalid contractions" begin
    inds1 = [
        IndexData(IndexLabel(1, :a), VIRT, 2),
        IndexData(IndexLabel(1, :b), VIRT, 2),
    ]
    inds2 = [
        IndexData(IndexLabel(2, :a), VIRT, 2),
        IndexData(IndexLabel(2, :c), VIRT, 2),
    ]
    th1 = TensorHandle(ITensorsBackend, rand(2, 2), inds1)
    th2 = TensorHandle(ITensorsBackend, rand(2, 2), inds2)
    
    # contraction with nonexistent index
    @test_throws KeyError ITensorsInterface.contract(th1, th2, IndexPair(IndexData(IndexLabel(3, :a), VIRT, 2), inds2[1]))
end


