using FibTN.TensorNetworks
using FibTN.FibTensorTypes

const IC = IndexContraction
const IL = IndexLabel

const GridPosition = NTuple{2, Int}

function connect_segments!(ftn::FibTensorNetwork, pos1::GridPosition, pos2::GridPosition, group::Int; enable_adjacency_check::Bool=true, dir::Symbol=:x)
    if enable_adjacency_check
        # check that positions are either directly above or below
        xdiff = pos1[1] - pos2[1]
        ydiff = pos1[2] - pos2[2]
        xdiff == 0 || ydiff == 0 || throw(ArgumentError("positions are diagonal"))
        if xdiff < 0 pos1, pos2, xdiff, dir = pos2, pos1, -xdiff, :h end
        if ydiff < 0 pos1, pos2, ydiff, dir = pos2, pos1, -ydiff, :v end
        xdiff == 1 || ydiff == 1 || throw(ArgumentError("positions aren't directly adjacent"))
    else
        dir == :h || dir == :v || throw(ArgumentError("dir must be specified if adjacency check disabled"))
    end
    # fetch segments
    s1 = ftn.segment_from_position[pos1]
    s2 = ftn.segment_from_position[pos2]
    @show pos1, s1.start
    @show pos2, s2.start
    @show group, dir
    @show IC(IL(s1.finish, :a), IL(group, :a))
    @show IC(IL(group, :b), IL(s2.start, :a))
    @show IC(IL(s1.start, :b), IL(group, :a))
    @show IC(IL(group, :b), IL(s2.finish, :b))
    print("\n")
    # add reflector
    _add_tensor!(ftn, group, Reflector)
    # add contractions depending on relative segment orientations
    if dir == :v
        add_contraction!(ftn.tn, IC(IL(s1.finish, :a), IL(group, :a)))
        add_contraction!(ftn.tn, IC(IL(group, :b), IL(s2.start, :a)))
    else
        add_contraction!(ftn.tn, IC(IL(s1.finish, :b), IL(group, :a)))
        add_contraction!(ftn.tn, IC(IL(group, :b), IL(s2.start, :b)))
    end
end

function grid_periodic(w::Int, h::Int)
    ftn = FibTensorNetwork()
    positions::Dict{Int, Point2}()
    new_group = 1
    # make all of the segments
    for i in 1:w
        for j in 1:h
            add_segment!(ftn, new_group, i, j)
            s = ftn.segment_from_position[(i, j)]
            new_group = s.finish + 1
            # insert points
            points = insert_midpoints([(i - 0.1, j - 0.1), (i + 0.1, j + 0.1)]; counts=[s.finish - s.start - 1])
            for (g, p) in zip(s.start:s.finish, points)
                positions[g] = p
            end
        end
    end
    # connect internal rows
    for i in 1:w-1
        for j in 1:h
            connect_segments!(ftn, (i, j), (i+1, j), new_group)
            positions[new_group] = (i+0.5, j)
            new_group += 1
        end
    end
    # connect internal columns
    for j in 1:h-1
        for i in 1:w
            connect_segments!(ftn, (i, j), (i, j+1), new_group)
            positions[new_group] = (i, j+0.5)
            new_group += 1
        end
    end
    # connect row wraparound
    for j in 1:h
        connect_segments!(ftn, (w, j), (1, j), new_group; enable_adjacency_check=false, dir=:h)
        new_group += 1
    end
    # connect col wraparound
    for i in 1:w
        connect_segments!(ftn, (i, h), (i, 1), new_group; enable_adjacency_check=false, dir=:v)
        new_group += 1
    end
    # make qubit lattice
    # qubits are assigned by rows, then cols, then internals
    function get_qubit_mapping(il::IndexLabel)
        s = get_segment(ftn, il.group)
        i, j = s.pos
        if s.start == il.group
            q1 = w*h + (j-1)*w + i
            q2 = (j-1)*w + i
            q3 = 2*w*h + 3*((j-1)*w + i-1) + 1
            return [q1, q2, q3]
        elseif get_group(s, Tail) == il.group
            q1 = 2*w*h + 3*((j-1)*w + i-1) + 1
            q2 = 2*w*h + 3*((j-1)*w + i-1) + 2
            q3 = 2*w*h + 3*((j-1)*w + i-1) + 3
            return [q1, q2, q3]
        elseif s.finish == il.group
            q1 = w*h + ((j == h ? 1 : j+1)-1)*w + i
            q2 = (j-1)*w + (i == 1 ? w : i-1)
            q3 = 2*w*h + 3*((j-1)*w + i-1) + 3
            return [q1, q2, q3]
        else
            error("IndexLabel provided doesn't belong to a physical index")
        end
    end
    ql = QubitLattice()
    for il in TensorNetworks.get_indices(ftn.tn)
        if il.port == :p add_index!(ql, get_qubit_mapping(il)) end
    end
    positions = Dict(g => (4*p[1], 4*p[2]) for (g, p) in positions)
    ftn, ql, positions
end

function grid_bounded(w::Int, h::Int)
    ftn = FibTensorNetwork()
    positions = Dict{Int, Point2}()
    new_group = 1
    # make segments
    for i in 1:w, j in 1:h
        add_segment!(ftn, new_group, i, j)
        s = ftn.segment_from_position[(i,j)]
        new_group = s.finish + 1
        # insert points
        points = insert_midpoints([(i + 0.1, j + 0.1), (i - 0.1, j - 0.1)]; counts=[s.finish - s.start - 1])
        for (g, p) in zip(s.start:s.finish, points)
            positions[g] = p
        end
    end

    #plot(ftn.tn, [positions[g] for g in 1:new_group-1], ftn.tensortype_from_group)

    # connect internal rows and cols
    for i in 1:w-1, j in 1:h
        connect_segments!(ftn, (i,j), (i+1,j), new_group)
        positions[new_group] = (i+0.5, j)
        new_group += 1
    end
    for j in 1:h-1, i in 1:w
        connect_segments!(ftn, (i,j), (i,j+1), new_group)
        positions[new_group] = (i, j+0.5)
        new_group += 1
    end

    plot(ftn.tn, [positions[g] for g in 1:new_group-1], ftn.tensortype_from_group)

    # --- right boundary (i = w) ---
    for j in 1:h-1
        s1 = ftn.segment_from_position[(w,j+1)]
        s2 = ftn.segment_from_position[(w,j)]
        _add_tensor!(ftn, new_group, Reflector)
        @show s1.start, s2.start
        add_contraction!(ftn.tn, IC(IL(s1.start,:b), IL(new_group,:a)))
        add_contraction!(ftn.tn, IC(IL(new_group,:b), IL(s2.start,:b)))
        positions[g] = (w + 0.35, j + 0.5)
        new_group += 1
    end

    # --- left boundary (i = 1) ---
    for j in 1:h-1
        s1 = ftn.segment_from_position[(1,j+1)]
        s2 = ftn.segment_from_position[(1,j)]
        g = new_group; new_group += 1
        _add_tensor!(ftn, g, Reflector)
        add_contraction!(ftn.tn, IC(IL(s1.finish,:b), IL(g,:b)))
        add_contraction!(ftn.tn, IC(IL(g,:a), IL(s2.finish,:b)))
        positions[g] = (1 - 0.35, j + 0.5)
    end

    # --- top boundary (j = h) ---
    for i in 1:w-1
        s1 = ftn.segment_from_position[(i+1,h)]
        s2 = ftn.segment_from_position[(i,h)]
        g = new_group; new_group += 1
        _add_tensor!(ftn, g, Reflector)
        add_contraction!(ftn.tn, IC(IL(s1.start,:a), IL(g,:b)))
        add_contraction!(ftn.tn, IC(IL(g,:a), IL(s2.start,:a)))
        positions[g] = (i + 0.5, h + 0.35)
    end

    # --- bottom boundary (j = 1) ---
    for i in 1:w-1
        s1 = ftn.segment_from_position[(i+1,1)]
        s2 = ftn.segment_from_position[(i,1)]
        g = new_group; new_group += 1
        _add_tensor!(ftn, g, Reflector)
        add_contraction!(ftn.tn, IC(IL(s1.finish,:a), IL(g,:a)))
        add_contraction!(ftn.tn, IC(IL(g,:b), IL(s2.finish,:a)))
        positions[g] = (i + 0.5, 1 - 0.35)
    end

    # --- qubit lattice ---
    function get_qubit_mapping(il::IndexLabel)
        s = get_segment(ftn, il.group)
        i, j = s.pos
        if s.start == il.group
            q1 = w*h + (j-1)*w + i
            q2 = (j-1)*w + i
            q3 = 2*w*h + 3*((j-1)*w + i-1) + 1
            return [q1, q2, q3]
        elseif get_group(s, Tail) == il.group
            q1 = 2*w*h + 3*((j-1)*w + i-1) + 1
            q2 = 2*w*h + 3*((j-1)*w + i-1) + 2
            q3 = 2*w*h + 3*((j-1)*w + i-1) + 3
            return [q1, q2, q3]
        elseif s.finish == il.group
            q1 = w*h + ((j == h ? 1 : j+1)-1)*w + i
            q2 = (j-1)*w + (i == 1 ? w : i-1)
            q3 = 2*w*h + 3*((j-1)*w + i-1) + 3
            return [q1, q2, q3]
        else
            error("IndexLabel provided doesn't belong to a physical index")
        end
    end
    ql = QubitLattice()
    #for il in TensorNetworks.get_indices(ftn.tn)
    #    if il.port == :p add_index!(ql, get_qubit_mapping(il)) end
    #end
    #positions = Dict(g => (4*p[1], 4*p[2]) for (g, p) in positions)
    ftn, ql, positions
end

index_labels(::Type{T}, group::Int) where T <: AbstractFibTensorType = [IL(group, p) for p in tensor_ports(T)]

function lattice_calculation(w::Int, h::Int)
    ftn, ql, positions = grid(w, h)
    # tn construction
    g2tt::Dict{Int, Type{<:AbstractFibTensorType}} = Dict(g => tt for (tt, gs) in tt2gs for g in gs)
    tn = TensorNetwork()
    for (g, tt) in g2tt add_tensor!(tn, TensorLabel(g, index_labels(tt, g))) end
        for ic in contractions add_contraction!(tn, ic) end
            # en construction and execution
            inds, data = naive_contract(tn, g2tt)
            # ql and data extraction
            ql = build_ql(qubits_from_index)
            s, a = get_states_and_amps(ql, inds, data)
            # visualization
            plot(tn, positions, g2tt)
            for (state, amp) in zip(s, a)
                plot(ql, positions, state, amp)
            end
end
