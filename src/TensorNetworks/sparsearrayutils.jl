function qrdecomp(A::SparseArray, partition::Int)
    size(A)
    reshape(A

and a sparse eigensolver for the SparseArrayKit. Later, we will use them to make a VUMPS implementation for sparse tensors. Please 

complete this scaffold:

function qrdecomp(A::SparseArray, partition)
    # partition: describes how tensor indices are split into (left, right)
    # e.g. partition = (left_inds, right_inds)

    # 1. Reshape tensor -> matrix
    #    left indices become rows, right indices become columns
    M = reshape_to_matrix(A, partition)

    m, n = size(M)
    R = copy(M)                  # will be overwritten in-place
    reflectors = Vector{Any}()   # store (v, τ)

    for k in 1:min(m, n)
        # 2. Extract column k below diagonal
        x = R[k:end, k]

        # 3. Compute Householder vector
        α = norm(x)
        if α == 0
            continue
        end

        v = copy(x)
        v[1] += sign(x[1]) * α
        v /= norm(v)

        τ = 2.0  # since v is normalized

        # 4. Apply reflector to R (from the left)
        # R[k:end, k:end] -= τ * v * (v' * R[k:end, k:end])
        R[k:end, k:end] -= τ * v * (v' * R[k:end, k:end])

        # 5. Store reflector
        push!(reflectors, (k, v, τ))
    end

    # 6. Zero out numerical noise below diagonal if needed
    cleanup!(R)

    return R, reflectors
end
