using FibTN.SegmentTensorTypes

@testset "SegmentTensorTypes basics" begin
    # verify that port and tensor data are returned without errors

    tensortypes = [
        BELBOW,
        TELBOW,
        VERTEX_TVL_VERTEX,
        VERTEX_TVL_RELBOW,
        RELBOW_TVL_VERTEX,
        RELBOW_TVL_RELBOW,
        VERTEX_LELBOW,
        LELBOW_VERTEX,
        LELBOW_LELBOW,
    ]

    @tensor telbow_data[U, R, tp] := tensor_data(REFLECTOR)[U, a] * tensor_data(ELBOW_T)[a, b, tp] * tensor_data(REFLECTOR)[b, c] * tensor_data(BOUNDARY)[d, c] * tensor_data(REFLECTOR)[d, R]
    # VERTEX_TVL_VERTEX
    # VERTEX_TVL_RELBOW
    # RELBOW_TVL_VERTEX
    # RELBOW_TVL_RELBOW
    @tensor vertex_lelbow_data[U, R, L, tp, bp] := tensor_data(REFLECTOR)[U, a] * tensor_data(VERTEX)[a, b, c, tp] * tensor_data(REFLECTOR)[c, d] * tensor_data(T_ELBOW)[L, d, bp] * tensor_data(REFLECTOR)[b, R]
    @tensor lelbow_vertex_data[U, D, L, tp, bp] := tensor_data(REFLECTOR)[U, a] * tensor_data(TAIL)[a, c, tp] * tensor_data(REFLECTOR)[c, d] * tensor_data(VERTEX)[D, L, d, bp]
    @tensor lelbow_lelbow_data[U, L, tp, bp] := tensor_data(REFLECTOR)[U, a] * tensor_data(TAIL)[a, c, tp] * tensor_data(REFLECTOR)[c, d] * tensor_data(T_ELBOW)[d, L, bp]

    # for some tensors also check the data matches expected matrix
    tensordata = Dict(
        BELBOW => tensor_data(ELBOW_T),
        TELBOW => telbow_data,
        VERTEX_LELBOW => vertex_lelbow_data,
        LELBOW_VERTEX => lelbow_vertex_data,
        LELBOW_LELBOW => lelbow_lelbow_data,
    )

    for tt in tensortypes
        ports = tensor_ports(tt)
        td1 = tensor_data(tt)
        td2 = tensor_data(tt)
        @test length(ports) == ndims(td1) # check port/data consistency
        @test td1 === td2 # check caching
        if haskey(tensordata, tt)
            @test td1 == tensordata[tt]
        end
    end
end
