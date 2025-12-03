# before this script is run, rsg should be instantiated, contractionsequences should be defined,
# and pindict, offset, scale, and nlabeloffsetscale should be added to the environment

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
