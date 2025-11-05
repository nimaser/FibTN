include("chain_latticebuilder.jl")
include("utiltensors.jl")

g = new_plaquette(6)
add_chain!(g, 5, 6, 1)
add_chain!(g, 4, 5, 7)
add_chain!(g, 4, 13, 8)
cap_remaining!(g)

populatetensors(g)
a = alignmentlattice(g)
q = qubitlattice(a)

#contractcaps!(g)
#T = contractgraph(g)
#qgraphs = generatequbitlattices(T, a, q)
#qubitlatticeplot(qgraphs)
