using FibTN.FibTensorTypes

@testset "FibTensorTypes basics" begin
    # verify that port and tensor data are returned without errors

    tensortypes = [REFLECTOR,
        BOUNDARY,
        VACUUMLOOP,
        TAIL,
        T_ELBOW,
        ELBOW_T,
        VERTEX,
        CROSSING,
        FUSION,
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

    taildata = zeros(Float64, 5, 5, 5)
    taildata[1, 1, 1] = 1
    taildata[2, 4, 3] = 1
    taildata[3, 3, 1] = 1
    taildata[4, 2, 3] = 1
    taildata[5, 5, 3] = 1

    t_elbowdata = zeros(Float64, 5, 5, 5)
    t_elbowdata[1, 1, 1] = 1
    t_elbowdata[2, 4, 2] = 1
    t_elbowdata[3, 3, 1] = 1
    t_elbowdata[4, 2, 2] = 1
    t_elbowdata[5, 5, 2] = 1

    elbow_tdata = zeros(Float64, 5, 5, 5)
    elbow_tdata[1, 1, 1] = 1
    elbow_tdata[2, 4, 4] = 1
    elbow_tdata[3, 3, 1] = 1
    elbow_tdata[4, 2, 4] = 1
    elbow_tdata[5, 5, 4] = 1

    # for some tensors also check the data matches expected matrix
    tensordata = Dict(REFLECTOR => reflectordata,
        BOUNDARY => boundarydata,
        VACUUMLOOP => vacuumloopdata,
        TAIL => taildata,
        T_ELBOW => t_elbowdata,
        ELBOW_T => elbow_tdata,
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
