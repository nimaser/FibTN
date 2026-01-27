module IndexTriplets

export combine_indices, split_index

function combine_indices(i::Int, j::Int, k::Int)
    if (i == j == k) p = (i == 0) ? 1 : 5
    elseif (i == 0) p = 2
    elseif (j == 0) p = 3
    elseif (k == 0) p = 4
    end
    p
end

function split_indices(a::Int)
    i = j = k = 1
    if p == 1 i = j = k = 0
    elseif p == 2 i = 0
    elseif p == 3 j = 0
    elseif p == 4 k = 0
    end
    i, j, k
end

end # module IndexTriplets
