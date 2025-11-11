rsg = new_plaquette(6)
add_plaquette!(rsg, 6, 1, 6)
cap_all!(rsg)

ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

#contractedge!(tg, 23, 24)
#contractedge!(tg, 25, 26)
#
#contractedge!(tg, 19, 20)
#contractedge!(tg, 21, 22)
#
#contractedge!(tg, 29, 30)
#contractedge!(tg, 27, 28)
#
#contractedge!(tg, 1, 31)
#contractedge!(tg, 6, 32)
#
#contractedge!(tg, 33, 34)
