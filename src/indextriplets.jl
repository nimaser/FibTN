module IndexTriplets

export combine_indices, split_index, permute_index_mapping

"""
Takes three qubit values (anyon types) `i`, `j`, and `k` which
can take values 0 or 1 each and maps them to a single combined index.

Combinations which do not violate the fusion rules are mapped to 1:5,
while combinations which do are mapped to 6:8.
"""
function combine_indices(i::Int, j::Int, k::Int)
    # fusion rule violating cases
    if i + j + k == 1
        if i == 1
            a = 6
        end
        if j == 1
            a = 7
        end
        if k == 1
            a = 8
        end
        return a
    end
    # nonviolating cases
    if (i == j == k)
        a = (i == 0) ? 1 : 5
    elseif (i == 0)
        a = 2
    elseif (j == 0)
        a = 3
    elseif (k == 0)
        a = 4
    end
    a
end

"""
Takes a combined index value `a` from 1:8 and maps it to three
individual qubit values. When `a` ∈ 1:5 it obeys the fusion
constraints, but when `a` ∈ 6:8 it violates them.
"""
function split_index(a::Int)
    # fusion rule violating cases
    if a > 5
        i = j = k = 0
        if a == 6
            i = 1
        end
        if a == 7
            j = 1
        end
        if a == 8
            k = 1
        end
        return i, j, k
    end
    # nonviolating cases
    i = j = k = 1
    if a == 1
        i = j = k = 0
    elseif a == 2
        i = 0
    elseif a == 3
        j = 0
    elseif a == 4
        k = 0
    end
    i, j, k
end

"""
Permutes the index mapping produced by `combine_indices`:
in other words, if three qubits are passed into it in some
order, and you wish to know the combined index if you were
to instead encode them in a different order, you can use
this function to safely permute them and obtain the new
combined index.

This function is useful when a physical index has encoded
qubits in a geometrically problematic way, and you wish to
change the encoding to a more convenient one.
"""
function permute_index_mapping(a::Int, p::NTuple{3,Int})
    # check permutation validity
    for i in p
        i ∈ collect(1:3) || throw(ArgumentError("invalid index $i in permutation $p"))
    end
    collect(p) == unique(p) || throw(ArgumentError("invalid permutation $p: duplicate index"))
    # permute
    vals = collect(split_index(a))
    permute!(vals, collect(p))
    combine_indices(vals...)
end

end # module IndexTriplets
