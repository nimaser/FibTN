rsg = new_plaquette(6)
add_plaquette!(rsg, 1, 2, 3)
add_plaquette!(rsg, 3, 4, 3)
add_plaquette!(rsg, 5, 6, 3)
cap_all!(rsg)

ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

contractcaps!(tg)

contractedge!(tg, 1, 2)
contractedge!(tg, 3, 4)
contractedge!(tg, 5, 6)

contractedge!(tg, 13, 16)
contractedge!(tg, 14, 17)
contractedge!(tg, 15, 18)

contractedge!(tg, 19, 20)
#contractedge!(tg, 21, 22)
#
#T = tg[collect(labels(tg))[1]].tensor
#s = tensor2states(T)
#statesplot(qg, s)
