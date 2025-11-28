# this script must define rsg, contractionsequences, pindict, offset, scale, and
# nlabeloffsetscale before calculateGS.jl is run
rsg = new_plaquette(3)
add_plaquette!(rsg, 3, 1, 3)
add_plaquette!(rsg, 4, 2, 3)

contractionsequences = [collect(1:4)]

pindict = Dict(1=>(0, 0),
               2=>(0, 1),
               3=>( sqrt(3)/2, -0.5),
               4=>(-sqrt(3)/2, -0.5))
offset = (1, 0)
scale = 2
nlabeloffsetscale = 0.15

make_boundary_trivial!(rsg)
