# @author Nikhil Maserang
# @date 2025/10/24

include("utiltensors.jl")

# so we can see our tensors
using ITensorUnicodePlots

# plaquette 1 top/bottom right/center/left
P1_1 = GSTriangle()
P1_2 = GSTriangle()
P1_3 = GSTriangle()
P1_4 = GSTriangle()
P1_5 = GSTriangle()
P1_6 = GSTriangle()

# generate a vacuum vector for each vertex
P1_1_0 = StringTripletVector(1)
P1_2_0 = StringTripletVector(1)
P1_3_0 = StringTripletVector(1)
P1_4_0 = StringTripletVector(1)
P1_5_0 = StringTripletVector(1)
P1_6_0 = StringTripletVector(1)

# contract the "a" indices with the vacuum vector
P1_1 = P1_1 * ITensors.δ(inds(P1_1)[1], inds(P1_1_0)[1]) * P1_1_0
P1_2 = P1_2 * ITensors.δ(inds(P1_2)[1], inds(P1_2_0)[1]) * P1_2_0
P1_3 = P1_3 * ITensors.δ(inds(P1_3)[1], inds(P1_3_0)[1]) * P1_3_0
P1_4 = P1_4 * ITensors.δ(inds(P1_4)[1], inds(P1_4_0)[1]) * P1_4_0
P1_5 = P1_5 * ITensors.δ(inds(P1_5)[1], inds(P1_5_0)[1]) * P1_5_0
P1_6 = P1_6 * ITensors.δ(inds(P1_6)[1], inds(P1_6_0)[1]) * P1_6_0

@show findall(!iszero, array(P1_1))
@show array(P1_1)[findall(!iszero, array(P1_1))]

# generate a reflection tensor for each edge (b c pairing)
R1 = StringTripletReflector(inds(P1_1)[2], inds(P1_6)[1])
R2 = StringTripletReflector(inds(P1_2)[2], inds(P1_1)[1])
R3 = StringTripletReflector(inds(P1_3)[2], inds(P1_2)[1])
R4 = StringTripletReflector(inds(P1_4)[2], inds(P1_3)[1])
R5 = StringTripletReflector(inds(P1_5)[2], inds(P1_4)[1])
R6 = StringTripletReflector(inds(P1_6)[2], inds(P1_5)[1])
                                                   
# match the indices of the tensors so they can be contracted;
# specifically, convert all c's to the clockwise neighbor's b
P1_1 = @visualize P1_1 * R1
P1_2 = P1_2 * R2
P1_3 = P1_3 * R3
P1_4 = P1_4 * R4
P1_5 = P1_5 * R5
P1_6 = P1_6 * R6

@show findall(!iszero, array(P1_1))
@show array(P1_1)[findall(!iszero, array(P1_1))]

P1_temp = @visualize P1_1 * P1_2
P1_temp_data = array(P1_temp)
@show findall(!iszero, P1_temp_data)
@show P1_temp_data[findall(!iszero, P1_temp_data)]

# contract the plaquette
T = @visualize P1_1 * P1_2 * P1_3 * P1_4 * P1_5 * P1_6

T_data = array(T)
@show findall(!iszero, T_data)
@show T_data[findall(!iszero, T_data)]

@show inds(T)

# janky temp display
q = new_plaquette(6)
cap_remaining!(q)
q1 = qubitlattice(q)
q2 = qubitlattice(q)

q2[1, 2] = true
q2[2, 3] = true
q2[3, 4] = true
q2[4, 5] = true
q2[5, 6] = true
q2[6, 1] = true

qubitlatticeplot([q1, q2], T_data[findall(!iszero, T_data)])
