using FibErrThresh: new_plaquette, add_plaquette!, add_chain!, add_cap!
using Graphs: nv, ne

@testset "separate plaquettes; order 6" begin
    g = new_plaquette(6)
    @test nv(g) == 6
    @test ne(g) == 6
end

@testset "separate plaquettes; order 6; capped" begin
    g = new_plaquette(6)
    for i in 1:6 add_cap!(g, i) end
    @test nv(g) == 12
    @test ne(g) == 12

    g = new_plaquette(6)
    cap_remaining!(g)
    @test nv(g) == 12
    @test ne(g) == 12
end

@testset "separate plaquettes; order 6, 6" begin
    g = new_plaquette(6)
    add_plaquette!(g, 6)
    @test nv(g) == 12
    @test ne(g) == 12
end

@testset "separate plaquettes; order 6, 6; capped" begin
    g = new_plaquette(6)
    add_plaquette!(g, 6)
    cap_remaining!(g)
    @test nv(g) == 24
    @test ne(g) == 24
end

@testset "connected plaquettes; order 6, 6" begin
    g = new_plaquette(6)
    add_chain!(g, 5, 1, 6)
    @test nv(g) == 10
    @test ne(g) == 11
end

@testset "connected plaquettes; order 6, 6; capped" begin
    g = new_plaquette(6)
    add_chain!(g, 5, 1, 6)
    cap_remaining!(g)
    @test nv(g) == 18
    @test ne(g) == 19
end

@testset "connected plaquettes; order 6, 6, 6" begin
    g = new_plaquette(6)
    add_chain!(g, 5, 6, 1)
    add_chain!(g, 4, 5, 7)
    @test nv(g) == 13
    @test ne(g) == 15
end

@testset "no order 2 plaquettes" begin
    @test_throws ArgumentError new_plaquette(2)

    g = new_plaquette(6)
    @test_throws ArgumentError add_plaquette!(g, 2)
   
    g = new_plaquette(6)
    @test_throws ArgumentError add_chain!(g, 2, 1, 1)
end

@testset "no degree >3 vertices" begin
    g = new_plaquette(6)
    add_chain!(g, 5, 6, 1)
    @test_throws ErrorException add_chain!(g, 4, 5, 6)

    g = new_plaquette(6)
    add_chain!(g, 5, 6, 1)
    @test_throws ErrorException add_chain!(g, 4, 1, 2)

    g = new_plaquette(6)
    @test_throws ErrorException add_chain!(g, 1, 1, 1)

    g = new_plaquette(6)
    add_chain!(g, 5, 6, 1)
    @test_throws ErrorException add_cap!(g, 1)
end

@testset "no multigraphs" begin
    g = new_plaquette(6)
    @test_throws ErrorException add_chain!(g, 1, 1, 6)
end

# TODO: testset to demonstrate that in the case of the above errors, the input is not mutated

