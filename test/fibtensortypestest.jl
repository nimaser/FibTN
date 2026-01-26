using FibTN.FibTensorTypes

@testset "fibtensortypes basics" begin
    # verify that multiple calls return the same object (caching)
    # and that the index and tensor data are returned without errors
    ri = index_labels(Reflector, 1)
    r1 = tensor_data(Reflector)
    r2 = tensor_data(Reflector)
    @test r1 === r2
    
    li = index_labels(LoopAmplitude, 1)
    l1 = tensor_data(LoopAmplitude)
    l2 = tensor_data(LoopAmplitude)
    @test l1 === l2
    
    vi = index_labels(Vertex, 1)
    v1 = tensor_data(Vertex)
    v2 = tensor_data(Vertex)
    @test v1 === v2
    
    ti = index_labels(Tail, 1)
    t1 = tensor_data(Tail)
    t2 = tensor_data(Tail)
    @test t1 === t2
end
