# this script must define rsg, contractionsequences, pindict, offset, scale, and
# nlabeloffsetscale before calculateGS.jl is run
rsg = new_plaquette(6)

contractionsequences = [collect(1:6)]

pindict = Dict(1=>(-sqrt(3), -1),
               2=>(-sqrt(3), 1),
               3=>(0, 2),
               4=>(sqrt(3), 1),
               5=>(sqrt(3), -1),
               6=>(0, -2)
              )
offset = (0, 0)
scale = 1
nlabeloffsetscale = 0.15
