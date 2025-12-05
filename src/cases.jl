CASES = [:case1,
         :case1h,
         :case2,
         :case2h,
         :case3a,
         :case3b,
         :case4,
         :case7,
         :case8,
         :case9,
         :case9h,
        ]

function case1()
    rsg = new_plaquette(3)
    cap_all!(rsg)
    
    contractionsequences = [collect(1:3)]
    
    pindict = Dict(1=>(-sqrt(3), -1),
                   2=>(-sqrt(3), 1),
                   3=>(0, 2),
                  )
    offset = (0, 0)
    scale = 1
    nlabeloffsetscale = 0.15

    rsg, contractionsequences, pindict, offset, scale, nlabeloffsetscale
end

function case1h()
    rsg = new_plaquette(6)
    cap_all!(rsg)
    
    contractionsequences = [collect(1:6)]
    
    pindict = Dict(1=>(-sqrt(3), -1),
                   2=>(-sqrt(3), 1),
                   3=>(0, 2),
                   4=>(sqrt(3), 1),
                   5=>(sqrt(3), -1),
                   6=>(0, -2)
                  )
    offset = (0, 0)
    scale = 1
    nlabeloffsetscale = 0.15
    
    rsg, contractionsequences, pindict, offset, scale, nlabeloffsetscale
end

function case2()
    rsg = new_plaquette(3)
    add_plaquette!(rsg, 3, 1, 3)
    cap_all!(rsg)
    
    contractionsequences = [[3, 2, 1, 4]]
    
    pindict = Dict(1=>(-1, -1),
                   2=>(-1, 1),
                   3=>(1, 1),
                   4=>(1, -1))
    offset = (1, -1)
    scale = 2
    nlabeloffsetscale = 0.15
    
    rsg, contractionsequences, pindict, offset, scale, nlabeloffsetscale
end

function case2h()
    len = 6
    rsg, pindict = hex_lattice(len, 6)
    cap_all!(rsg)
    
    contractionsequences = []
    for i in 1:len
        push!(contractionsequences, [2, 4*i-1, 4*i], [1, 4*i+2, 4*i+1])
    end
    push!(contractionsequences, [1, 2])

    offset = (1, -1)
    scale = 2
    nlabeloffsetscale = 0.15
    
    rsg, contractionsequences, pindict, offset, scale, nlabeloffsetscale
end

function case3a()
    rsg = new_plaquette(3)
    add_plaquette!(rsg, 3, 1, 3)
    add_plaquette!(rsg, 4, 2, 3)
    cap_all!(rsg)
    
    contractionsequences = [collect(1:4)]
    
    pindict = Dict(1=>(0, 0),
                   2=>(0, 1),
                   3=>( sqrt(3)/2, -0.5),
                   4=>(-sqrt(3)/2, -0.5))
    offset = (1, 0)
    scale = 2
    nlabeloffsetscale = 0.15
    
    rsg, contractionsequences, pindict, offset, scale, nlabeloffsetscale
end

function case3b()
    rsg, args... = case3a()
    make_boundary_trivial!(rsg)
    rsg, args...
end

function case4()
    rsg = new_plaquette(4)
    add_plaquette!(rsg, 4, 1, 3)
    add_plaquette!(rsg, 5, 2, 4)
    add_plaquette!(rsg, 6, 3, 3)
    cap_all!(rsg)
    
    contractionsequences = [[1, 4, 5, 2, 3, 6]]
    
    pindict = Dict(1=>(-1, 0),
                   2=>(1, 0),
                   3=>(2, 1),
                   4=>(-2, 1),
                   5=>(-2, -1),
                   6=>(2, -1))
    offset = (0, -1)
    scale = 2
    nlabeloffsetscale = 0.3
    
    make_boundary_trivial!(rsg)
    
    rsg, contractionsequences, pindict, offset, scale, nlabeloffsetscale
end

function case7()
    rsg = new_plaquette(6)
    add_plaquette!(rsg, 1, 2, 4)
    add_plaquette!(rsg, 8, 3, 4)
    add_plaquette!(rsg, 9, 4, 4)
    add_plaquette!(rsg, 10, 5, 4)
    add_plaquette!(rsg, 11, 6, 4)
    add_plaquette!(rsg, 12, 7, 4)
    cap_all!(rsg)
    
    contractionsequences = [[7, 8], [1, 2], [3, 9], [12, 6], [5, 4], [11, 10], [1, 7], [5, 11], [1, 3], [5, 12], [1, 5]]
    
    pindict = Dict(1=>(0, 0),
                   2=>(0, 2),
                   3=>(sqrt(3), 3),
                   4=>(2*sqrt(3), 2),
                   5=>(2*sqrt(3), 0),
                   6=>(sqrt(3), -1),
                   7=>(-sqrt(3), -1),
                   8=>(-sqrt(3), 3),
                   9=>(sqrt(3), 5),
                   10=>(3*sqrt(3), 3),
                   11=>(3*sqrt(3), -1),
                   12=>(sqrt(3), -3),
                  )
    offset = (0, -1)
    scale = 2
    nlabeloffsetscale = 0.3

    make_boundary_trivial!(rsg)
    
    rsg, contractionsequences, pindict, offset, scale, nlabeloffsetscale
end

function case8()
    len = 6
    rsg, pindict = hex_lattice(len, 1)
    cap_all!(rsg)
    
    contractionsequences = []
    for i in 1:len
        push!(contractionsequences, [2, 4*i-1, 4*i], [1, 4*i+2, 4*i+1])
    end
    push!(contractionsequences, [1, 2])
    
    offset = (0, -1)
    scale = 2
    nlabeloffsetscale = 0.3
    
    rsg, contractionsequences, pindict, offset, scale, nlabeloffsetscale
end

function case9()
    rsg = new_plaquette(6)
    add_plaquette!(rsg, 1, 2, 5)
    add_plaquette!(rsg, 9, 3, 4)
    add_plaquette!(rsg, 10, 4, 5)
    add_plaquette!(rsg, 12, 5, 5)
    add_plaquette!(rsg, 11, 13, 3)
    add_plaquette!(rsg, 14, 6, 4)
    add_plaquette!(rsg, 15, 7, 5)
    add_plaquette!(rsg, 16, 8, 3)
    cap_all!(rsg)

    contractionsequences = [[7, 8, 16], [11, 12, 13], [1, 2, 7, 9], [4, 5, 11, 14], [1, 3, 10, 4, 6, 15]]
    
    pindict = Dict(1=>(0, 0),
                   2=>(0, 2),
                   3=>(sqrt(3), 3),
                   4=>(2*sqrt(3), 2),
                   5=>(2*sqrt(3), 0),
                   6=>(sqrt(3), -1),
                   7=>(-sqrt(3), -1),
                   8=>(-2*sqrt(3), 0),
                   9=>(-sqrt(3), 3),
                  10=>(0, 5),
                  11=>(3*sqrt(3), 5),
                  12=>(3*sqrt(3), 3),
                  13=>(4*sqrt(3), 2),
                  14=>(3*sqrt(3), -1),
                  15=>(0, -3),
                  16=>(-sqrt(3), -3),
                 )

    offset = (0, -1)
    scale = 2
    nlabeloffsetscale = 0.3

end

function case9h()
    rsg, pindict = hex_lattice(3, 3)
    cap_all!(rsg)

    contractionsequences = [[1, 2], [3, 15], [5, 4], [7, 17], [9, 8], [11, 19], [13, 12],
                            [22, 21], [30, 29], [20, 27], [18, 25], [16, 23], [16, 24], [18, 26],
                            [20, 28], [5, 6], [9, 10], [13, 14], [1, 3, 5], [30, 20, 22], [1, 9, 7],
                            [30, 18, 11, 13, 16], [1, 30]
                           ]

    offset = (0, -1)
    scale = 2
    nlabeloffsetscale = 0.3
    
    rsg, contractionsequences, pindict, offset, scale, nlabeloffsetscale
end
