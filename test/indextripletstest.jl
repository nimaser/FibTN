using FibTN.IndexTriplets

@testset "combine_indices basics" begin
    @test combine_indices(0, 0, 0) == 1
    @test combine_indices(0, 1, 1) == 2
    @test combine_indices(1, 0, 1) == 3
    @test combine_indices(1, 1, 0) == 4
    @test combine_indices(1, 1, 1) == 5
    @test combine_indices(1, 0, 0) == 6
    @test combine_indices(0, 1, 0) == 7
    @test combine_indices(0, 0, 1) == 8
end

@testset "split_index basics" begin
    @test split_index(1) == (0, 0, 0)
    @test split_index(2) == (0, 1, 1)
    @test split_index(3) == (1, 0, 1)
    @test split_index(4) == (1, 1, 0)
    @test split_index(5) == (1, 1, 1)
    @test split_index(6) == (1, 0, 0)
    @test split_index(7) == (0, 1, 0)
    @test split_index(8) == (0, 0, 1)
end

@testset "split and combine roundtrip" begin
    for a in 1:8
        i, j, k = split_index(a)
        @test combine_indices(i, j, k) == a
    end
end

@testset "permute_index_mapping basics" begin
    # just test all possible inputs
    for a in 1:8
        i, j, k = split_index(a)
        @test permute_index_mapping(a, (1, 2, 3)) == a
        @test permute_index_mapping(a, (1, 3, 2)) == combine_indices(i, k, j)
        @test permute_index_mapping(a, (2, 3, 1)) == combine_indices(j, k, i)
        @test permute_index_mapping(a, (2, 1, 3)) == combine_indices(j, i, k)
        @test permute_index_mapping(a, (3, 2, 1)) == combine_indices(k, j, i)
        @test permute_index_mapping(a, (3, 1, 2)) == combine_indices(k, i, j)
    end
end
