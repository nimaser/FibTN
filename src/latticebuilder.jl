function new_hex_chain(length::Int)
    if length < 1 throw(ArgumentError("chain must have length of at least one")) end
    rsg = new_plaquette(6)
    for i in 2:length
        # add plaquettes sharing an edge parallel to edge 1, 2
        s = 4*(i-1)
        e = s + 1
        add_plaquette!(rsg, s, e, 6)
    end
    rsg
end

function add_hex_chain!(rsg::MetaGraph, length::Int, tips::Vector{Int})
    if length < 1 throw(ArgumentError("chain must have length of at least one")) end
    # if we're only adding one plaquette
    if length == 1
        s = tips[1]
        e = tips[1] + 1
        add_plaquette!(rsg, s, e, 6)
        return
    end
    # add first plaquette to the new chain
    add_plaquette!(rsg, tips[1], tips[2], 6)
    # add other plaquettes, minus the last one
    for i in 2:length-1
        # share edge parallel to edge 1, 2, but on this chain
        s = nv(rsg)
        e = tips[i+1]
        add_plaquette!(rsg, s, e, 6)
    end
    # add last plaquette
    s = nv(rsg)
    e = tips[end] + 1
    add_plaquette!(rsg, s, e, 6)
end

function hex_lattice(length::Int, height::Int)
    if length < 1 throw(ArgumentError("chain must have length of at least one")) end
    if height < 1 throw(ArgumentError("chain must have height of at least one")) end
    # make first new chain
    rsg = new_hex_chain(length)
    # make pins for first chain
    pin = Dict(1=>(0,0), 2=>(0, 2), 3=>(√3, 3), 4=>(2*√3, 2), 5=>(2*√3, 0), 6=>(√3, -1))
    for i in 2:length
        j = 6+4*(i-2)
        pin = merge(pin, Dict(j+1=>((2*i-1)*√3, 3), j+2=>(2*i*√3, 2), j+3=>(2*i*√3, 0), j+4=>((2*i-1)*√3, -1)))
    end
    # make other chains
    tips = [-1 + 4*i for i in 1:length]
    for i in 2:height
        # make new plaquette chain
        s = nv(rsg) + 2
        add_hex_chain!(rsg, length, tips)
        tips = [s + 2*j - 2 for j in 1:length]
        # set new pinned positions
        for j in 1:length
            x = (i-1)*√3 + 2*(j-1)*√3
            y = (3*i-1)
            pin = merge(pin, Dict(s+2*(j-1)-1=>(x, y), s+2*(j-1)=>(x + √3, y+1)))
        end
        x = (i-1)*√3 + 2*(length)*√3
        y = (3*i-1)
        s = nv(rsg) 
        pin = merge(pin, Dict(s-1=>(x, y), s=>(x, y-2)))
    end
    rsg, pin
end


