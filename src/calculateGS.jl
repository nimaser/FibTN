# before this script is run, rsg, contractionsequences, pindict, offset, scale, and
# nlabeloffsetscale should've been added to the environment

# cap the rsg
cap_all!(rsg)

# generate the other graphs
ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

# generate the networklayout to display the qg
for (k, v) in pindict
    pindict[k] = scale .* v .+ offset
end
l = NetworkLayout.Spring(pin=pindict)

# display the tg before contraction
f = Figure()
_, _, axs = getaxisgrid(f, 1)
p = tgplot(axs[1], tg, layout=l)
finalize(f, axs)
display(f)

# contract the tg
contractcaps!(tg)
for cs in contractionsequences
    contractsequence!(tg, cs)
end
T = contractionresult(tg)
s = tensor2states(T)

# display the result of contracting the tg
f = Figure()
w, h, axs = getaxisgrid(f, length(s))
plots = statesplot(axs, qg, s, layout=l)
finalize(f, axs)
display(f)
