module QubitOperators

using ..IndexTriplets
using ..TensorNetworks
import ..TensorNetworks: tensor_ports, tensor_data, tensor_color, tensor_marker


export QubitOperatorTensorType
# export types

### TENSOR TYPES ###

abstract type QubitOperatorTensorType <: TensorType end

struct REFLECTOR <: FibTensorType end
struct BOUNDARY <: FibTensorType end

### TENSOR PORTS ###





### TENSOR DATA ###

function _generate_1_qubit_unitary_data(U::<:AbstractArray, qubit_idx::Int)
    size(U) = (2, 2) || throw(ArgumentError("U must have size (2, 2), got $(size(U))"))
    arr = zeros(Float64, 5, 5)
    for a in 1:5
        # get matrix elements corresponding to this input qubit value
        vals = split_index(a)
        qval = vals[qubit_idx]
        coeffs = U[qval, :]
        # set matrix elements for the two branches of the superposition
        vals[qubit_idx] = 0
        arr[a, combine_indices(vals)] = coeffs[0]
        vals[qubit_idx] = 1
        arr[a, combine_indices(vals)] = coeffs[1]
    end
    arr
end








    end
end


### TENSOR DISPLAY PROPERTIES ###




end # module QubitOperators
