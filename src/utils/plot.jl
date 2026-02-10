using GraphMakie
using Graphs
using Printf

tt2color(::Type{Reflector}) = :gray
tt2color(::Type{Boundary}) = :black
tt2color(::Type{VacuumLoop}) = :orange
tt2color(::Type{Vertex}) = :red
tt2color(::Type{Tail}) = :blue
tt2color(::Type{Crossing}) = :green
tt2color(::Type{Fusion}) = :teal

tt2marker(::Type{Reflector}) = :vline
tt2marker(::Type{Boundary}) = :xcross
tt2marker(::Type{VacuumLoop}) = :circle
tt2marker(::Type{Tail}) = :rect
tt2marker(::Type{Vertex}) = :star6
tt2marker(::Type{Crossing}) = :star4
tt2marker(::Type{Fusion}) = :star3

color_from_qubit(qubitvals::Dict{Int, Int}) = Dict(q => v == 1 ? :red : :gray90 for (q, v) in qubitvals)

position_from_index(indices::Vector{IL}, positions::Vector{Point2}) = Dict(i => positions[i.group] for i in indices)

function calculategridsidelengths(area::Int)
    width = height = floor(sqrt(area))
    if width == sqrt(area) return Int(width), Int(height) end
    width += 1
    while width * height < area
        width += 1
    end
    Int(width), Int(height)
end

function getaxisgrid(f, area::Int; args...)
    w, h = calculategridsidelengths(area)
    axs = [Axis(f[r, c]; aspect = DataAspect(), args...) for r in 1:h, c in 1:w]
    w, h, axs
end

function finalize!(f, axs)
    for ax in axs
        hidespines!(ax)
        hidedecorations!(ax)
    end
    resize_to_layout!(f)
end

function plot(tn::TensorNetwork, positions::Vector{Point2}, g2tt::Dict{Int, Type{<:AbstractFibTensorType}})
    groups = sort(collect(keys(g2tt)))
    colors = [tt2color(g2tt[g]) for g in groups]
    markers = [tt2marker(g2tt[g]) for g in groups]
    tnds = TensorNetworkDisplaySpec(groups, positions, colors, markers)

    f = Figure()
    ax = Axis(f[1, 1])
    TensorNetworkVisualizer.visualize(tn, tnds, ax)
    DataInspector(f, range=30)
    finalize!(f, [ax])
    display(GLMakie.Screen(), f)
    ax
end

plot(ftn::FibTensorNetwork, positions::Vector{Point2}) = plot(ftn.tn, positions, ftn.tensortype_from_group)

function plot(ql::QubitLattice, qlds::QubitLatticeDisplaySpec, qubitvals::Dict{Int, Int}, amp::Real)
    qlds.color_from_qubit = color_from_qubit(qubitvals)

    f = Figure()
    ax = Axis(f[1, 1])
    QubitLatticeVisualizer.visualize(ql, qlds, ax)
    ax.title = "$amp"
    DataInspector(f, range=30)
    finalize!(f, [ax])
    display(GLMakie.Screen(), f)
    ax
end

function plot_interactive(ql::QubitLattice, qlds::QubitLatticeDisplaySpec, states::Vector{Dict{Int, Int}}, amps::Vector{Real})
    f = Figure()
    ax = Axis(f[1, 1])
    p = QubitLatticeVisualizer.visualize(ql, qlds, ax)
    finalize!(f, [ax])
    function action(idx, event, axis)
        clickededge = collect(edges(ql.graph))[idx]
        if haskey(ql._qubit_from_edge, clickededge)
            qubit = ql._qubit_from_edge[clickededge]
            qubitval = qlds.color_from_qubit[qubit] == :black ? 0 : 1
            # this is gonna need some better datastructure thinking

        end

        #            p.edge_color[][idx] = rand(RGB)
        #            p.edge_color[] = p.edge_color[]
    end
    register_interaction!(ax, :edgeclick, EdgeClickHandler(action))
    display(GLMakie.Screen(), f)
end

function plot_all(ql::QubitLattice, qlds::QubitLatticeDisplaySpec, states::Vector{Dict{Int, Int}}, amps::Vector{Real})
    # maximum number of states we can have per pane, 5x8
    maxstatesperpane = 3
    # using exactly max states per pane
    numpanes = Int(ceil(length(states) / maxstatesperpane))
    # now move some states from the full panes to the last one
    # so that each pane has roughly the same amount of states
    statesperpane = Int(ceil(length(states) / numpanes))
    # now move some states back from the last one so that each
    # pane is perfectly filled
    w, h = calculategridsidelengths(statesperpane)
    statesperpane = w*h
    # update numpanes
    numpanes = Int(ceil(length(states) / statesperpane))

    # create the screen, figure, axes, etc
    screen = GLMakie.Screen()
    f = Figure()
    w, h, axs = getaxisgrid(f, statesperpane)
    for ax in axs ax.titlesize=32 end

    get_pane_states(pane::Int) = states[(pane-1)*statesperpane+1:min(pane*statesperpane, length(states))]
    get_pane_amps(pane::Int) = amps[(pane-1)*statesperpane+1:min(pane*statesperpane, length(amps))]

    # set up slilders
    sg = SliderGrid(f[h+1, :],
                    (label="Pane #", range = 1:numpanes, startvalue = 1, update_while_dragging=false)
                   )
    sl = sg.sliders[1]
    on(sl.value) do v
        # get the states and amps
        pane_states = get_pane_states(v[])
        pane_amps = get_pane_amps(v[])
        # prepare the axes and display the amps
        for ax in axs
            empty!(ax)
            ax.title=""
        end
        for i in 1:length(pane_amps)
            axs[i].title="$(pane_amps[i])"
        end
        # plot the states
        for i in 1:length(pane_states)
            qlds.color_from_qubit = color_from_qubit(pane_states[i])
            QubitLatticeVisualizer.visualize(ql, qlds, axs[i])
        end
    end

    # compute normalization factor and add to title
    square(x) = x^2
    N = sqrt(sum(square.(amps)))
    f[0, :] = Label(f, L"N = %$N", fontsize=24)

    # finalize slider interaction
    notify(sl.value)
    register_interaction!(axs[1], :change_pane) do event::KeysEvent, ax
        if Keyboard.left ∈ event.keys
            set_close_to!(sl, sl.value[] - 1)
        elseif Keyboard.right ∈ event.keys
            set_close_to!(sl, sl.value[] + 1)
        end
    end
    # hide axis splines, resize to fit content, and display
    finalize!(f, axs)
    display(screen, f)
end
