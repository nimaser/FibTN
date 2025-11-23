using NetworkLayout

include("tensorbuilder.jl")
include("latticebuilder.jl")
include("latticevisualizer.jl")

rsg = new_plaquette(6)
cap_all!(rsg)

ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

l = NetworkLayout.Spring(pin=Dict(1=>(0, 0), 2=>(0, 2), 3=>(sqrt(3)/2, 3), 4=>(sqrt(3), 2), 5=>(sqrt(3), 0), 6=>(sqrt(3)/2, -1)))

f, ax, p = tgplot(tg, layout=l)
f
readline()

contractcaps!(tg, true)
contractsequence!(tg, [1, 2, 3, 4, 5, 6], true)

T = contractionresult(tg)
s = tensor2states(T)

p = statesplot(qg, s, layout=l)
