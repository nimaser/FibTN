include("tensorbuilder.jl")
include("latticebuilder.jl")
include("latticevisualizer.jl")

### GS PARAMS ###

rsg = new_plaquette(5)
add_plaquette!(rsg, 5, 1, 4)
add_plaquette!(rsg, 7, 2, 5)
add_plaquette!(rsg, 9, 3, 4)

contractionsequences = [[1, 5, 6, 7], [2, 9, 10, 3], [1, 8, 2, 4]]

pindict = Dict(1=>(-1, 0),
               2=>(1, 0),
               3=>(2, 1),
               4=>(0, 1),
               5=>(-2, 1),
               6=>(-3, 0),
               7=>(-2, -1),
               8=>(0, -1),
               9=>(2, -1),
               10=>(3, 0))
offset = (0, -1)
scale = 2
nlabeloffsetscale = 0.3

### THIS STAYS THE SAME FOR EVERY GS ###

cap_all!(rsg)
ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

addqdim!(tg, 1, 2)
addqdim!(tg, 5, 6)
addqdim!(tg, 7, 8)
addqdim!(tg, 9, 10)

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
for cs in contractionsequences
    contractsequence!(tg, cs)
end
T = contractionresult(tg)
s = tensor2states(T)

f = Figure()
w, h, axs = getaxisgrid(f, length(s))
plots = statesplot!(axs, qg, s, layout=l, nlabeloffsetscale=nlabeloffsetscale*scale)
finalize(f, axs)
display(f)
