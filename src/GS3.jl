include("tensorbuilder.jl")
include("latticebuilder.jl")

rsg = new_plaquette(4)
add_plaquette!(rsg, 4, 1, 4)
add_plaquette!(rsg, 6, 2, 4)
cap_all!(rsg)

ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

display(tgplot(tg))
readline()

contractcaps!(tg, true)
contractsequence!(tg, [2, 3, 4, 5, 6, 7, 1], true)

T = contractionresult(tg)
s = tensor2states(T)
p = statesplot(qg, s)
