rsg = new_plaquette(6)
add_plaquette!(rsg, 6, 1, 6)
cap_all!(rsg)

ig = rsg2ig(rsg)
qg = ig2qg(ig)
tg = ig2tg(ig)
