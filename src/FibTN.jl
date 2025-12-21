module FibTN
end # module FibTN

# define case and mode before running this script    

using Serialization
using NetworkLayout

include("tensorbuilder.jl")
include("networkbuilder.jl")
if mode == :V || mode == :D include("networkvisualizer.jl") end
include("latticebuilder.jl")

mutable struct GSCalculation
    rsg::MetaGraph
    ig::MetaGraph
    qg::MetaGraph
    tg::Union{MetaGraph, Nothing}
    contractionsequences::Vector
    pindict::Dict
    offset::Tuple
    scale::Float64
    nlabeloffsetscale::Float64
    l::NetworkLayout.AbstractLayout
    T::Union{ITensor, Nothing}
    s::Dict
end

if mode == :D
    gscalc = deserialize(pwd() * "/out/case$(case)")

    displayrsg(gscalc.rsg, gscalc.l)
    displayig(gscalc.ig, gscalc.l)
    displayqg(gscalc.qg, gscalc.l)
    displays(gscalc.s, gscalc.qg, gscalc.l)
else
    # get the rsg and display information
    include("cases.jl")
    f = getfield(Main, Symbol("case$(case)"))
    rsg, contractionsequences, pindict, offset, scale, nlabeloffsetscale = f()
    
    # generate the other graphs
    ig = rsg2ig(rsg)
    qg = ig2qg(ig)
    tg = ig2tg(ig)
    
    # generate the networklayout to display the qg
    for (k, v) in pindict
        pindict[k] = scale .* v .+ offset
    end
    l = NetworkLayout.Spring(pin=pindict)

    #displayqg(qg, l)
    #throw(ErrorException())
    
    # contract the tg
    contractcaps!(tg)
    for cs in contractionsequences
        contractsequence!(tg, cs)
    end
    
    # get the results
    T = contractionresult(tg)
    s = tensor2states(T)
    
    gscalc = GSCalculation(rsg, ig, qg, tg,
                           contractionsequences,
                           pindict, offset, scale, nlabeloffsetscale,
                           l, T, s
                          )

    if mode == :S
        # gotta discard the memory-intensive parts
        gscalc.tg = nothing
        gscalc.T = nothing
        serialize(pwd() * "/out/case$(case)", gscalc)
    end

    if mode == :V
        displayrsg(gscalc.rsg, gscalc.l)
        displayig(gscalc.ig, gscalc.l)
        displayqg(gscalc.qg, gscalc.l)
        displays(gscalc.s, gscalc.qg, gscalc.l)
    end
end
