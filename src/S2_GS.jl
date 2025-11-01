# @author Nikhil Maserang
# @date 2025/10/24

include("tensor_init.jl")

# so we can see our tensors
using ITensorUnicodePlots

# plaquette 1 top/bottom right/center/left

P1_TC = GSTriangle()
P1_TL = GSTriangle()
P1_BL = GSTriangle()
P1_BC = GSTriangle()

# plaquette 1, 2 

# generate a vacuum vector for each vertex
P1_TR_0 = StringTripletVector(1)
P1_TC_0 = StringTripletVector(1)
P1_TL_0 = StringTripletVector(1)
P1_BL_0 = StringTripletVector(1)
P1_BC_0 = StringTripletVector(1)
P1_BR_0 = StringTripletVector(1)

# contract the "a" indices with the vacuum vector
P1_TR = P1_TR * δ(inds(P1_TR)[1], inds(P1_TR_0)[1]) * P1_TR_0
P1_TC = P1_TC * δ(inds(P1_TC)[1], inds(P1_TC_0)[1]) * P1_TC_0
P1_TL = P1_TL * δ(inds(P1_TL)[1], inds(P1_TL_0)[1]) * P1_TL_0
P1_BL = P1_BL * δ(inds(P1_BL)[1], inds(P1_BL_0)[1]) * P1_BL_0
P1_BC = P1_BC * δ(inds(P1_BC)[1], inds(P1_BC_0)[1]) * P1_BC_0
P1_BR = P1_BR * δ(inds(P1_BR)[1], inds(P1_BR_0)[1]) * P1_BR_0

@show findall(!iszero, array(P1_TR))
@show array(P1_TR)[findall(!iszero, array(P1_TR))]

# generate a reflection tensor for each edge (b c pairing)
R1 = StringTripletReflector()
R2 = StringTripletReflector()
R3 = StringTripletReflector()
R4 = StringTripletReflector()
R5 = StringTripletReflector()
R6 = StringTripletReflector()


# match the indices of the tensors so they can be contracted;
# specifically, convert all c's to the clockwise neighbor's b
# after reflecting c so it can match with b
P1_TR = @visualize P1_TR * δ(inds(P1_TR)[2], inds(R1)[1]) * R1 * δ(inds(R1)[2], inds(P1_BR)[1])
P1_TC = P1_TC * δ(inds(P1_TC)[2], inds(R2)[1]) * R2 * δ(inds(R2)[2], inds(P1_TR)[1])
P1_TL = P1_TL * δ(inds(P1_TL)[2], inds(R3)[1]) * R3 * δ(inds(R3)[2], inds(P1_TC)[1])
P1_BL = P1_BL * δ(inds(P1_BL)[2], inds(R4)[1]) * R4 * δ(inds(R4)[2], inds(P1_TL)[1])
P1_BC = P1_BC * δ(inds(P1_BC)[2], inds(R5)[1]) * R5 * δ(inds(R5)[2], inds(P1_BL)[1])
P1_BR = P1_BR * δ(inds(P1_BR)[2], inds(R6)[1]) * R6 * δ(inds(R6)[2], inds(P1_BC)[1])

@show findall(!iszero, array(P1_TR))
@show array(P1_TR)[findall(!iszero, array(P1_TR))]

P1_temp = @visualize P1_TR * P1_TC
@show findall(!iszero, array(P1_temp))
@show array(P1_temp)[findall(!iszero, array(P1_temp))]

# contract the plaquette
P1 = @visualize P1_TR * P1_TC * P1_TL * P1_BL * P1_BC * P1_BR

P1_data = array(P1)
@show findall(!iszero, array(P1_data))
@show array(P1_data)[findall(!iszero, array(P1_data))]

@show inds(P1)
