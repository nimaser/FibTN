# this script must define rsg, contractionsequences, pindict, offset, scale, and
# nlabeloffsetscale before calculateGS.jl is run
rsg = new_plaquette(3)
add_plaquette!(rsg, 3, 1, 3)

contractionsequences = [[3, 2, 1, 4]]

pindict = Dict(1=>(-1, -1),
               2=>(-1, 1),
               3=>(1, 1),
               4=>(1, -1))
offset = (1, -1)
scale = 2
nlabeloffsetscale = 0.15

addqdim!(tg, 1, 2)
addqdim!(tg, 3, 4)
