module IndexLabels

export IndexLabel IndexLevel, VIRT, PHYS

@enum IndexLevel VIRT PHYS

struct IndexLabel
    name::Symbol
    level::IndexLevel
end

end # module IndexData
