include("tensorbuilder.jl")
include("latticebuilder.jl")

rsg = new_plaquette(6)
cap_all!(rsg)

ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

display(tgplot(tg))
readline()

contractcaps!(tg, true)
contractsequence!(tg, [1, 2, 3, 4, 5, 6], true)

T = contractionresult(tg)
s = tensor2states(T)
p = statesplot(qg, s)
