include("tensorbuilder.jl")
include("latticebuilder.jl")
include("latticevisualizer.jl")

### GS PARAMS ###

rsg = new_plaquette(6)

contractionsequence = [1, 2, 3, 4, 5, 6]

pindict = Dict(1=>(-sqrt(3), -1),
               2=>(-sqrt(3), 1),
               3=>(0, 2),
               4=>(sqrt(3), 1),
               5=>(sqrt(3), -1),
               6=>(0, -2))
offset = (0, 0)
scale = 1
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
p = tgplot!(axs[1], tg, layout=l)
finalize(f, axs)
display(f)

contractcaps!(tg)
contractsequence!(tg, contractionsequence)
T = contractionresult(tg)
s = tensor2states(T)

f = Figure()
w, h, axs = getaxisgrid(f, length(s))
plots = statesplot!(axs, qg, s, layout=l)
finalize(f, axs)
display(f)
