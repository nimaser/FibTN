include("tensorbuilder.jl")
include("latticebuilder.jl")

rsg = new_plaquette(4)
add_plaquette!(rsg, 4, 1, 3)
add_plaquette!(rsg, 5, 2, 4)
add_plaquette!(rsg, 6, 3, 3)
cap_all!(rsg)

ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

display(tgplot(tg))
readline()

# contractcaps!(tg, true) # this line is unnecessary because there are no caps
contractsequence!(tg, [1, 4, 5, 2, 3, 6], true)

T = contractionresult(tg)
s = tensor2states(T)
p = statesplot(qg, s)
