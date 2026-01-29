using FibTN.FibTensorTypes

@testset "fibtensortypes basics" begin
    # verify that multiple calls return the same object (caching)
    # and that the index and tensor data are returned without errors
    # and that the dimensions match the number of indices
    
    # for some tensors also check the data
    tensortypes = [Reflector, Boundary, VacuumLoop, Tail, Vertex, Crossing, Fusion]

    ϕ = (1 + √5) / 2
    
    reflectordata = [1 0 0 0 0;
                     0 0 0 1 0;
                     0 0 1 0 0;
                     0 1 0 0 0;
                     0 0 0 0 1]
                     
    boundarydata = [1 0 0 0 0;
                    0 0 0 0 0;
                    0 0 0 0 0;
                    0 1 0 0 0;
                    0 0 0 0 0]

    vacuumloopdata = [1 0 0 0 0;
                      0 0 0 1 0;
                      0 0 ϕ 0 0;
                      0 ϕ 0 0 0;
                      0 0 0 0 ϕ]

    taildata = zeros(Float64, 5, 5, 5)
    taildata[1, 1, 1] = 1
    taildata[2, 4, 3] = 1
    taildata[3, 3, 1] = 1
    taildata[4, 2, 3] = 1
    taildata[5, 5, 3] = 1

    tensordata = Dict(Reflector => reflectordata,
                      Boundary => boundarydata,
                      VacuumLoop => vacuumloopdata,
                      Tail => taildata,
                      )
    
    for tt in tensortypes
        ips = tensor_ports(tt)
        td1 = tensor_data(tt)
        td2 = tensor_data(tt)
        @test length(ips) == ndims(td1)
        @test td1 === td2
        if haskey(tensordata, tt) @test td1 == tensordata[tt] end
    end
end
