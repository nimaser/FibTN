# this script must define rsg, contractionsequences, pindict, offset, scale, and
# nlabeloffsetscale before calculateGS.jl is run
rsg = new_plaquette(4)
add_plaquette!(rsg, 4, 1, 3)
add_plaquette!(rsg, 5, 2, 4)
add_plaquette!(rsg, 6, 3, 3)

contractionsequences = [[1, 4, 5, 2, 3, 6]]

pindict = Dict(1=>(-1, 0),
               2=>(1, 0),
               3=>(2, 1),
               4=>(-2, 1),
               5=>(-2, -1),
               6=>(2, -1))
offset = (0, -1)
scale = 2
nlabeloffsetscale = 0.3

addqdim!(tg, 1, 2)
addqdim!(tg, 4, 5)
addqdim!(tg, 5, 6)
addqdim!(tg, 6, 3)
