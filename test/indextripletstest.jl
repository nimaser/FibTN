using FibTN.IndexTriplets

@testset "ijk2p basics" begin 
    @test ijk2p(0, 0, 0) == 1
    @test ijk2p(0, 1, 1) == 2
    @test ijk2p(1, 0, 1) == 3
    @test ijk2p(1, 1, 0) == 4
    @test ijk2p(1, 1, 1) == 5
end

@testset "p2ijk basics" begin
    @test p2ijk(1) == (0, 0, 0)
    @test p2ijk(2) == (0, 1, 1)
    @test p2ijk(3) == (1, 0, 1)
    @test p2ijk(4) == (1, 1, 0)
    @test p2ijk(5) == (1, 1, 1)
end

@testset "p2ijk2p roundtrip" begin
    for p in 1:5
        i, j, k = p2ijk(p)
        @test ijk2p(i, j, k) == p
    end
end
