module QubitOperatorTensorTypes

using ..IndexTriplets
using ..TensorNetworks
import ..TensorNetworks: tensor_ports, tensor_data, tensor_color, tensor_marker


export QubitOperatorTensorType
# export types

### TENSOR TYPES ###

abstract type QubitOperatorTensorType <: TensorType end

struct PAULI_X{N} <: QubitOperatorTensorType end
struct PAULI_Z{N} <: QubitOperatorTensorType end
struct PAULI_Y{N} <: QubitOperatorTensorType end

### TENSOR PORTS ###





### TENSOR DATA ###


function _generate_1_qubit_operator_data(U::<:AbstractArray{<:Number,2}, qubit_idx::Int;
    arr::<:AbstractArray{<:Number, 2}=zeros(Float64, 5, 5)
)
    size(U) = (2, 2) || throw(ArgumentError("U must have size (2, 2), got $(size(U))"))
    0 < qubit_idx < 4 || throw(ArgumentError("qubit index must be in 1:3, got $qubit_idx"))
    for a in 1:5
        # get matrix elements corresponding to this input qubit value
        vals = split_index(a)
        qval = vals[qubit_idx]
        coeffs = U[qval, :]
        # set matrix elements for the two branches of the superposition
        vals[qubit_idx] = 0
        arr[a, combine_qubits(vals)] = coeffs[0]
        vals[qubit_idx] = 1
        arr[a, combine_qubits(vals)] = coeffs[1]
    end
    arr
end

using SparseArrayKit

function _generate_2_qubit_operator_data(
    U::AbstractArray{<:Number,4},
    qubit_map::Dict{Int, Vector{Tuple{Int,Int}}}
)
    size(U) == (2,2,2,2) || throw(ArgumentError("U must have size (2,2,2,2), got $(size(U))"))
    for (q, idxposvect) in qubit_map
        for (idxnum, qubitpos) in idxposvect

    # Output tensor dimensions: 8×8×8×8
    # (phys1_in, phys2_in, phys1_out, phys2_out)
    arr = SparseArray{Float64}(8,8,8,8)

    # Track whether dims ever use values >5
    used_high = falses(4)  # one flag per dimension

    for a in 1:8, b in 1:8, c in 1:8, d in 1:8

        # Convert physical indices to qubit triples
        in1 = split_index(a)
        in2 = split_index(b)
        out1 = split_index(c)
        out2 = split_index(d)

        consistent = true

        # canonical qubit values
        qin  = Dict{Int,Int}()
        qout = Dict{Int,Int}()

        # check input consistency
        for (q, locs) in qubit_map
            vals = Int[]
            for (phys_idx, pos) in locs
                if phys_idx == 1
                    push!(vals, in1[pos])
                elseif phys_idx == 2
                    push!(vals, in2[pos])
                else
                    error("Invalid physical index number")
                end
            end
            if !all(v -> v == vals[1], vals)
                consistent = false
                break
            end
            qin[q] = vals[1]
        end

        consistent || continue

        # check output consistency
        for (q, locs) in qubit_map
            vals = Int[]
            for (phys_idx, pos) in locs
                if phys_idx == 1
                    push!(vals, out1[pos])
                elseif phys_idx == 2
                    push!(vals, out2[pos])
                else
                    error("Invalid physical index number")
                end
            end
            if !all(v -> v == vals[1], vals)
                consistent = false
                break
            end
            qout[q] = vals[1]
        end

        consistent || continue

        # Extract operator value
        # (assuming qubits labeled 1 and 2)
        val = U[
            qin[1] + 1,
            qin[2] + 1,
            qout[1] + 1,
            qout[2] + 1
        ]

        iszero(val) && continue

        arr[a,b,c,d] = val

        # Track high index usage
        if a > 5; used_high[1] = true; end
        if b > 5; used_high[2] = true; end
        if c > 5; used_high[3] = true; end
        if d > 5; used_high[4] = true; end
    end

    return arr, used_high
end

### TENSOR DISPLAY PROPERTIES ###




end # module QubitOperatorTensorTypes
