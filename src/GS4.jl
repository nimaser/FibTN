# this script must define rsg, contractionsequences, pindict, offset, scale, and
# nlabeloffsetscale before calculateGS.jl is run
rsg = new_plaquette(4)
add_plaquette!(rsg, 4, 1, 4)
add_plaquette!(rsg, 6, 2, 4)

contractionsequences = [[2, 3, 4, 5, 6, 7, 1]]

pindict = Dict(1=>(0, 0),
               2=>(0, 1),
               3=>(1, 1),
               4=>(1, 0),
               5=>(1, -1),
               6=>(0, -1),
               7=>(-1, 0))
offset = (-1, -1)
scale = 2
nlabeloffsetscale = 0.15
