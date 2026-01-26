using FibTN.FibTensorTypes

@testset "fibtensortypes caching" begin
    # verify that multiple calls return the same object
    r1 = tensor_data(Reflector)
    r2 = tensor_data(Reflector)
    @test r1 === r2
    
    l1 = tensor_data(LoopAmplitude)
    l2 = tensor_data(LoopAmplitude)
    @test l1 === l2
    
    v1 = tensor_data(Vertex)
    v2 = tensor_data(Vertex)
    @test v1 === v2
    
    t1 = tensor_data(Tail)
    t2 = tensor_data(Tail)
    @test t1 === t2
end
