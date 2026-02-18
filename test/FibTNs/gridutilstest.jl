using FibErrThresh.FibTNs

@testset "BoundaryConditionsDict nonperiodic basics" begin
    # construct empty
    bcd = BoundaryConditionsDict{false, Unsigned}(3, 3)
    @test ingrid(bcd, 2, 2)
    @test !ingrid(bcd, 4, 4)

    # doesn't wrap
    bcd[1, 1] = 2
    @test haskey(bcd, 1, 1)
    bcd[1, 1] = 3
    @test bcd[1, 1] == 3
    @test_throws KeyError bcd[4, 1]
    @test_throws KeyError bcd[1, 4]
    @test_throws KeyError bcd[4, 4]
    delete!(bcd, 1, 1)
    @test !haskey(bcd, 1, 1)
end

@testset "BoundaryConditionsDict perioidic basics" begin
    # construct empty
    bcd = BoundaryConditionsDict{true, Unsigned}(3, 3)
    @test ingrid(bcd, 2, 2)
    @test ingrid(bcd, 4, 4)

    # wraps correctly
    bcd[1, 1] = 2
    bcd[4, 1] = 5
    @test haskey(bcd, 4, 1)
    @test haskey(bcd, 4, 4)
    @test bcd[1, 1] == bcd[4, 1] == 5
    @test_throws KeyError bcd[1, 2]
    delete!(bcd, 4, 4)
    @test !haskey(bcd, 1, 1)
end
