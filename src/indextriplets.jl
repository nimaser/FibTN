module IndexTriplets

export combine_indices, split_index

function combine_indices(i::Int, j::Int, k::Int)
    if (i == j == k) a = (i == 0) ? 1 : 5
    elseif (i == 0) a = 2
    elseif (j == 0) a = 3
    elseif (k == 0) a = 4
    end
    a
end

function split_index(a::Int)
    i = j = k = 1
    if a == 1 i = j = k = 0
    elseif a == 2 i = 0
    elseif a == 3 j = 0
    elseif a == 4 k = 0
    end
    i, j, k
end

end # module IndexTriplets
