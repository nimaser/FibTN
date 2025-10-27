# @author Nikhil Maserang
# @date 2025/10/24

include("tensor_init.jl")

# so we can see our tensors
using ITensorUnicodePlots

# plaquette 1 top/bottom right/center/left
P1_TR = GSTriangle()
P1_TC = GSTriangle()
P1_TL = GSTriangle()
P1_BL = GSTriangle()
P1_BC = GSTriangle()
P1_BR = GSTriangle()

@visualize P1_TR

@show inds(P1_TR)


