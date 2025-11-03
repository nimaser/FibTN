using Graphs
using MetaGraphsNext
using GraphRecipes, Plots
using StringAlgorithms

include("tensor_init.jl")
include("lattice_builder.jl")

function test1()
    D = makeD([1], [6], [])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test2()
    D = makeD([1, 2], [3, 4], [])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test3()
    D = makeD([1, 2], [3, 4], [1=>2])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test4()
    D = makeD([1, 2, 3, 4], [3, 4, 5, 3], [1=>2, 2=>3, 3=>4])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test5()
    D = makeD([1, 2, 3], [3, 4, 5], [1=>2, 2=>3, 3=>1])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test6()
    D = makeD([1, 2, 3, 4], [6, 6, 6, 6], [1=>2, 2=>3, 3=>4, 4=>1, 1=>3])
    makeLvertices(D)
    makeLedges(D)
    D
end

function test7()
    D = makeD([1, 2, 3, 4, 5], [4, 4, 4, 4, 4], [1=>2, 1=>3, 1=>5, 4=>2, 4=>3, 4=>5, 5=>2, 2=>3])
    makeLvertices(D)
    makeLedges(D)
    D
end

