

# @testset "Segment group_from_gridposition" begin
#     # column-major: group = (j-1)*w + x
#     @test FibTNs.group_from_gridposition(1, 1, 3) == 1
#     @test FibTNs.group_from_gridposition(2, 1, 3) == 2
#     @test FibTNs.group_from_gridposition(3, 1, 3) == 3
#     @test FibTNs.group_from_gridposition(1, 2, 3) == 4
#     @test FibTNs.group_from_gridposition(2, 2, 3) == 5
#     @test FibTNs.group_from_gridposition(3, 2, 3) == 6
# end
