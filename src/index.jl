struct IndexLabel
    group::Int
    port::Symbol
end

# so that pairs of IndexLabels can be ordered
Base.isless(a::IndexLabel, b::IndexLabel) = a.group < b.group || (a.group == b.group && a.port < b.port)

struct IndexPair
    a::IndexLabel
    b::Indexlabel
    function IndexPair(a, b)
        # check that indices aren't the same
        if a == b error("labels of contracted indices mustn't match") end
        # enforce ordering to prevent duplicates
        if b < a a, b = b, a end
        new(a, b)
    end
end
