using FibErrThresh.FibTNs
using FibErrThresh.SegmentTensorTypes

@testset "segmentmaskgrid basics" begin
    # nonperiodic case
    smg = segmentmaskgrid(2, 2; middle=STT_T)

    # all positions occupied
    @test length(smg) == 4
    for x in 1:2, y in 1:2
        @test haskey(smg, x, y)
    end

    # interior ports have been properly inferred
    @test hasR(smg[1, 1]) && hasL(smg[2, 1])
    @test hasR(smg[1, 2]) && hasL(smg[2, 2])
    @test hasU(smg[1, 1]) && hasD(smg[1, 2])
    @test hasU(smg[2, 1]) && hasD(smg[2, 2])

    # no wrap-around edges for this nonperiodic case
    @test !hasL(smg[1, 1]) && !hasR(smg[2, 1])
    @test !hasL(smg[1, 2]) && !hasR(smg[2, 2])
    @test !hasD(smg[1, 1]) && !hasU(smg[1, 2])
    @test !hasD(smg[2, 1]) && !hasU(smg[2, 2])

    # now check fixmiddles!
    smg = segmentmaskgrid(3, 3; middle=STT_T)
    @test hasT(smg[2, 2])
    @test hasM(smg[2, 2])
end
