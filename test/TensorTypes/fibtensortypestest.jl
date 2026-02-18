using FibErrThresh.FibTensorTypes

@testset "FibTensorTypes basics" begin
    # verify that port and tensor data are returned without errors

    tensortypes = [
        REFLECTOR,
        BOUNDARY,
        VACUUMLOOP,
        ELBOW_T1,
        ELBOW_T2,
        ELBOW_T3,
        TAIL,
        VERTEX,
        CROSSING,
        FUSION,
        STRINGEND,
        EXCITATION,
        FUSIONTREEROOT,
        DOUBLEDFUSION,
    ]

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
        0 0 0 ϕ 0;
        0 0 ϕ 0 0;
        0 1 0 0 0;
        0 0 0 0 ϕ]

    elbow_t2_data = zeros(Float64, 5, 5, 5)
    elbow_t2_data[1, 1, 1] = 1
    elbow_t2_data[2, 4, 3] = 1
    elbow_t2_data[3, 3, 1] = 1
    elbow_t2_data[4, 2, 3] = 1
    elbow_t2_data[5, 5, 3] = 1

    elbow_t1_data = zeros(Float64, 5, 5, 5)
    elbow_t1_data[1, 1, 1] = 1
    elbow_t1_data[2, 4, 2] = 1
    elbow_t1_data[3, 3, 1] = 1
    elbow_t1_data[4, 2, 2] = 1
    elbow_t1_data[5, 5, 2] = 1

    elbow_t3_data = zeros(Float64, 5, 5, 5)
    elbow_t3_data[1, 1, 1] = 1
    elbow_t3_data[2, 4, 4] = 1
    elbow_t3_data[3, 3, 1] = 1
    elbow_t3_data[4, 2, 4] = 1
    elbow_t3_data[5, 5, 4] = 1

    # for some tensors also check the data matches expected matrix
    tensordata = Dict(REFLECTOR => reflectordata,
        BOUNDARY => boundarydata,
        VACUUMLOOP => vacuumloopdata,
        ELBOW_T1 => elbow_t1_data,
        ELBOW_T2 => elbow_t2_data,
        ELBOW_T3 => elbow_t3_data,
        TAIL => elbow_t2_data,
    )

    for tt in tensortypes
        ports = tensor_ports(tt)
        td1 = tensor_data(tt)
        td2 = tensor_data(tt)
        @test length(ports) == ndims(td1) # check port/data consistency
        @test td1 === td2 # check caching
        if haskey(tensordata, tt)
            @test collect(td1) == tensordata[tt]
        end
    end
end
