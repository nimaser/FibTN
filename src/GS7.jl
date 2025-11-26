# this script must define rsg, contractionsequences, pindict, offset, scale, and
# nlabeloffsetscale before calculateGS.jl is run
rsg = new_plaquette(5)
add_plaquette!(rsg, 5, 1, 4)
add_plaquette!(rsg, 7, 2, 5)
add_plaquette!(rsg, 9, 3, 4)
add_plaquette!(rsg, 10, 4, 5)

contractionsequences = [[1, 5, 6, 7], [3, 4, 12, 11, 10], [2, 9, 8, 1, 3]]

pindict = Dict(1=>(-1, 0),
               2=>(1, 0),
               3=>(2, 1),
               4=>(0, 1),
               5=>(-2, 1),
               6=>(-3, 0),
               7=>(-2, -1),
               8=>(0, -1),
               9=>(2, -1),
               10=>(3, 0),
               11=>(3, 1),
               12=>(2.5, 1))
offset = (0, -1)
scale = 2
nlabeloffsetscale = 0.3
