rsg = new_plaquette(4)
add_plaquette!(rsg, 4, 1, 4)
cap_all!(rsg)

ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)

#contractedge!(tg, 11, 12)
#contractedge!(tg, 13, 14)
#contractedge!(tg, 1, 4)
#
#contractedge!(tg, 15, 17)
#contractedge!(tg, 18, 16)
