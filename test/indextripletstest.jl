using FibTN.IndexTriplets

@testset "combine_indices basics" begin 
    @test combine_indices(0, 0, 0) == 1
    @test combine_indices(0, 1, 1) == 2
    @test combine_indices(1, 0, 1) == 3
    @test combine_indices(1, 1, 0) == 4
    @test combine_indices(1, 1, 1) == 5
end

@testset "split_index basics" begin
    @test split_index(1) == (0, 0, 0)
    @test split_index(2) == (0, 1, 1)
    @test split_index(3) == (1, 0, 1)
    @test split_index(4) == (1, 1, 0)
    @test split_index(5) == (1, 1, 1)
end

@testset "p2ijk2p roundtrip" begin
    for a in 1:5
        i, j, k = split_index(a)
        @test combine_indices(i, j, k) == a
    end
end
