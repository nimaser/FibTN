rsg = new_plaquette(6)
add_plaquette!(rsg, 1, 2, 4)
add_plaquette!(rsg, 3, 4, 4)
add_plaquette!(rsg, 5, 6, 4)
cap_all!(rsg)

ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

#contractcaps!(tg)
#
#contractedge!(tg, 19, 20)
#contractedge!(tg, 21, 22)
#contractedge!(tg, 23, 24)
#
#contractedge!(tg, 1, 2)
#contractedge!(tg, 3, 4)
#contractedge!(tg, 5, 6)
#
#contractedge!(tg, 25, 28)
#contractedge!(tg, 27, 30)
#contractedge!(tg, 26, 29)
#
#contractedge!(tg, 31, 32)
#contractedge!(tg, 32, 33)
#
#T = tg[collect(labels(tg))[1]].tensor
#s = tensor2states(T)
#statesplot(qg, s)
