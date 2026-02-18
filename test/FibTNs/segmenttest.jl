using FibErrThresh.FibTNs
using FibErrThresh.SegmentTensorTypes

@testset "Segment basics" begin
    T = validatemask(SegmentTensorType{STT_U | STT_R | STT_M | STT_T | STT_D | STT_L})
    s = FibTNs.Segment(T, 2, 3, 5)

    # test everything except positions: those only matter for display
    # so just see if it displays properly
    @test s.group == 5
    @test s.gpos == (2, 3)
    @test FibTNs.get_segmenttensortype(s) === T
    @test isempty(s.qubits)  # qubits not yet assigned
end
