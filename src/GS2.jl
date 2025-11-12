include("tensorbuilder.jl")
include("latticebuilder.jl")

rsg = new_plaquette(3)
add_plaquette!(rsg, 3, 1, 3)
cap_all!(rsg)

ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

display(tgplot(tg))
readline()

contractcaps!(tg, true)
contractsequence!(tg, [3, 2, 1, 4], true)

T = contractionresult(tg)
s = tensor2states(T)
p = statesplot(qg, s)
