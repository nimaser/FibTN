include("tensorbuilder.jl")
include("latticebuilder.jl")
include("latticevisualizer.jl")

### GS PARAMS ###

rsg = new_plaquette(4)
add_plaquette!(rsg, 4, 1, 3)
add_plaquette!(rsg, 5, 2, 4)
add_plaquette!(rsg, 6, 3, 3)

contractionsequence = [1, 4, 5, 2, 3, 6]

pindict = Dict(1=>(-1, 0),
               2=>(1, 0),
               3=>(2, 1),
               4=>(-2, 1),
               5=>(-2, -1),
               6=>(2, -1))
offset = (0, -1)
scale = 2
nlabeloffsetscale = 0.3

### THIS STAYS THE SAME FOR EVERY GS ###

cap_all!(rsg)
ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

addqdim!(tg, 1, 2)
addqdim!(tg, 4, 5)
addqdim!(tg, 5, 6)
addqdim!(tg, 6, 3)

for (k, v) in pindict
    pindict[k] = scale .* v .+ offset
end
l = NetworkLayout.Spring(pin=pindict)

f = Figure()
_, _, axs = getaxisgrid(f, 1)
p = tgplot!(axs[1], tg, layout=l, nlabeloffsetscale=nlabeloffsetscale*scale)
finalize(f, axs)
display(f)

contractcaps!(tg)
contractsequence!(tg, contractionsequence)
T = contractionresult(tg)
s = tensor2states(T)

f = Figure()
w, h, axs = getaxisgrid(f, length(s))
plots = statesplot!(axs, qg, s, layout=l, nlabeloffsetscale=nlabeloffsetscale*scale)
finalize(f, axs)
display(f)
