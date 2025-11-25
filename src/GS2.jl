include("tensorbuilder.jl")
include("latticebuilder.jl")
include("latticevisualizer.jl")

### GS PARAMS ###

rsg = new_plaquette(3)
add_plaquette!(rsg, 3, 1, 3)

contractionsequence = [3, 2, 1, 4]

pindict = Dict(1=>(-1, -1),
               2=>(-1, 1),
               3=>(1, 1),
               4=>(1, -1))
offset = (1, -1)
scale = 2
nlabeloffsetscale = 0.15

### THIS STAYS THE SAME FOR EVERY GS ###

cap_all!(rsg)
ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

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
