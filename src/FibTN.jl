module FibTN

end # module FibTN

include("tensorbuilder.jl")
include("networkbuilder.jl")
include("networkvisualizer.jl")
include("latticebuilder.jl")

case = length(ARGS) > 0 ? ARGS[1] : throw(ErrorException("case number must be supplied as the first argument"))
genTN = length(ARGS) > 1 && ARGS[2] == "1" ? true : false
contractTN = length(ARGS) > 2 && ARGS[3] == "1" ? true : false

@show case, genTN, contractTN

include("GS$(case).jl")
@show rsg
if genTN
    include("genTN.jl")
    if contractTN
        include("contractTN.jl")
    end
end
