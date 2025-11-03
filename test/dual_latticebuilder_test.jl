function D2Ltest(Dvertices, Dorders, Dedges, Lvertexcount, Ledgecount, output)
    D = makeD(Dvertices, Dorders, Dedges)
    @test Set(Dvertices) = Set((labels(D)))
    @test length(edges) = length(collect(edge_labels(D)))
    for (p, o) in zip(vertices, orders)
        @test D[p].order == o
        @test D[p].edgesprocessed = false
    end
    if output @show collect(labels(D)), collect(edge_labels(D)) end

    makeLvertices(D)
    @test Lvertexcount == length(collect(labels(D[])))
    if output @show collect(labels(D[])), collect(edge_labels(D[])) end

    makeLedges(D)
    @test Ledgecount == length(collect(edge_labels(D[])))
    if output @show collect(labels(D[])), collect(edge_labels(D[])) end

    if output metagraphplot(D[]) end
end

@testset "LBFD separated; order 6" begin
    D2Ltest([1], [6], [], 6, 1, true)
end;

@testset "LBFD separated; order 3, 4" begin
    D2Ltest(1:2, [3, 4], [], 7, 7, true)
end;

@testset "LBFD 2-chain; 1, 2 floating" begin
    D2Ltest(1:2, [3, 4], [1=>2], 5, 6, true)
end;

@testset "LBFD 4-chain; 1, 0, 2, 3 floating" begin
    D2Ltest(1:4, [3, 4, 6, 5], [1=>2, 2=>3, 3=>4], 10, 13, true)
end;

@testset "LBFD 3-cycle; 0 floating" begin
    D2Ltest(1:3, [3, 3, 3], [1=>2, 2=>3, 3=>1], 4, 6, true)
end;

@testset "LBFD 3-cycle; 0, 1, 2 floating" begin
    D2Ltest(1:3, [3, 4, 5], [1=>2, 2=>3, 3=>1], 7, 9, true)
end;

@testset "LBFD 4-cycle; 0 floating" begin
    D2Ltest(1:4, [4, 4, 4, 4], [1=>2, 2=>3, 3=>4, 4=>1], 8, 12, true)
end;

@testset "LBFD 4-cycle; 0, 1, 2, 3 floating" begin
    D2Ltest(1:4, [4, 5, 6, 7], [1=>2, 2=>3, 3=>4, 4=>1], 14, 18, true)
end;

@testset "LBFD 4-cycle; dangling acyclic" begin
    D2Ltest(1:6, [7, 6, 6, 5, 3, 4], [1=>2, 2=>3, 3=>4, 4=>1, 1=>5, 2=>6, 6=>3], 18, 24, true)
end

@testset "LBFD 6-cycle; hexagons" begin
    D2Ltest(1:6, [6, 6, 6, 6, 6, 6], [1=>2, 2=>3, 3=>4, 4=>5, 5=>6, 6=>1], 24, 30, true)
end;

@testset "LBFD 4-tile; hexagons" begin
    D2Ltest(1:4, [6, 6, 6, 6], [1=>2, 2=>3, 3=>4, 4=>1, 1=>3], 16, 19, true)
end;

@testset "LBFD 5-tile; quadrilaterals" begin
    D2Ltest(1:5, [4, 4, 4, 4, 4], [1=>2, 1=>3, 1=>5, 4=>2, 4=>3, 4=>5, 5=>2, 2=>3], 8, 12, true)
end;

#@testset "LBFD nested cycle 1" begin
#    D2Ltest], [], [], ., ., true)
#end;
#
#@testset "LBFD nested cycle 2" begin
#    D2Ltest([], [], [], ., ., true)
#end;
