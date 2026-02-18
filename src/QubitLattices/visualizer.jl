using FibErrThresh.QubitLattices
using FibErrThresh.TensorNetworks

using GLMakie, GeometryBasics

export visualize

"""The default color, linewidth, etc for plotting qubit values."""
function default_qubitdisplayspec()
    colors = Dict{Int, Any}()
    linewidths = Dict{Int, Any}()
    colors[0] = :gray90
    colors[1] = :red
    linewidths[0] = 5
    linewidths[1] = 10
    Dict(:color => colors, :linewidth => linewidths)
end

"""
Plots a qubit lattice on the provided axis. The 0 qubit is not displayed.
All unpaired qubits are displayed as tails of length `tail_length`.

Custom display properties for qubit values 0 and 1 should be provided via
the `qubitdisplayspec` parameter.
"""
function FibErrThresh.visualize(
    ax::Axis,
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f},
    qubitvals::Dict{Int, Int};
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_length::Real=0.3,
)
    # get lists of edge endpoints and other edge display properties
    qubits = collect(get_qubits(ql)) # so we have a deterministic ordering
    filter!(!=(0), qubits) # we don't want to display the 0 qubit
    edge_endpoints = Vector{Tuple{Point2f, Point2f}}()
    colors = []
    linewidths = []
    for q in qubits
        inds = get_indices(ql, q)
        if length(inds) == 1
            # unpaired case
            pos1 = position_from_index[inds[1]]
            pos2 = (pos1[1] + tail_length, pos1[2])
        else
            # paired case
            length(inds) == 2 || error("more than two indices sharing qubit $q")
            pos1 = position_from_index[inds[1]]
            pos2 = position_from_index[inds[2]]
        end
        push!(edge_endpoints, (pos1, pos2))
        push!(colors, qubitdisplayspec[:color][qubitvals[q]])
        push!(linewidths, qubitdisplayspec[:linewidth][qubitvals[q]])
    end
    # plot linesegments
    segmentsresult = linesegments!(ax, edge_endpoints, color=Observable(colors), linewidth=5)
    segmentsresult.inspector_label = (plot, i, idx) -> begin
        i = i ÷ 2 # i counts up by endpoints, so by two per segment plotted
        q = qubits[i]
        "qubit $q with value $(qubitvals[q])"
    end
    autolimits!(ax) # compute limits based on the axis content
    qubits, segmentsresult
end

"""Plots `ql` on `ax` with all qubits set to 0, for lattice visualization purposes."""
function FibErrThresh.visualize(
    ax::Axis,
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f};
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_length::Real=0.3
)
    qubitvals = Dict(q => 0 for q in get_qubits(ql))
    visualize(ax, ql, position_from_index, qubitvals;
        qubitdisplayspec=qubitdisplayspec, tail_length=tail_length)
end


"""
Plots `states` and corresponding `amps` onto a figure with a scrollable set of 'panes',
each of which can contain at most `maxstatesperpane` states. Left and right buttons can
be used to change between panes.

Returns a Makie `Figure` and array of axes.

Forwards `qubitdisplayspec` and `tail_length` to `visualize`.
"""
function FibErrThresh.visualize(
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f},
    states::Vector{Dict{Int, Int}},
    amps::Vector{<:Real};
    maxstatesperpane::Int=3,
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_length::Real=0.5,
)
    # using exactly maxstatesperpane states per pane
    numpanes = Int(ceil(length(states) / maxstatesperpane))
    # the last pane might be mostly empty, in which case we
    # can keep the same number of panes while reducing the
    # number of states per pane by moving some states to
    # the last one from the others so they all have roughly
    # the same amount, then calculate the width and height
    # of each pane
    statesperpane = Int(ceil(length(states) / numpanes))
    w, h = _calculategridsidelengths(statesperpane)
    # w*h could be larger than stateperpane, so to make
    # sure all panes are fully filled we do
    statesperpane = w*h
    # and finally we update numpanes
    numpanes = Int(ceil(length(states) / statesperpane))

    # create the figure, axes, etc
    f = Figure()
    axs = [Axis(f[r, c]; aspect = DataAspect()) for r in 1:h, c in 1:w]
    for ax in axs ax.titlesize=32 end

    # compute normalization factor and add to title
    square(x) = x^2
    N = sqrt(sum(square.(amps)))
    f[0, :] = Label(f, L"N = %$N", fontsize=24, tellwidth=false)

    get_pane_states(pane::Int) = states[(pane-1)*statesperpane+1:min(pane*statesperpane, length(states))]
    get_pane_amps(pane::Int) = amps[(pane-1)*statesperpane+1:min(pane*statesperpane, length(amps))]

    # set up slilders
    sg = SliderGrid(f[h+1, :],
                    (
                        label="Pane #",
                        range = 1:numpanes,
                        startvalue = 1,
                        update_while_dragging=false
                    )
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
        # plot the states; hide unused axes so DataInspector doesn't crash on them
        for i in 1:length(pane_states)
            visualize(axs[i], ql, position_from_index, pane_states[i]; qubitdisplayspec=qubitdisplayspec, tail_length=tail_length)
            axs[i].scene.visible[] = true
        end
        for i in length(pane_states)+1:length(axs)
            axs[i].scene.visible[] = false
        end
    end
    # finalize slider interaction
    notify(sl.value)
    register_interaction!(axs[1], :change_pane) do event::KeysEvent, ax
        if Keyboard.left ∈ event.keys
            set_close_to!(sl, sl.value[] - 1)
        elseif Keyboard.right ∈ event.keys
            set_close_to!(sl, sl.value[] + 1)
        end
    end
    # cleanup and return
    DataInspector(f, range=30)
    for ax in axs
        hidespines!(ax)
        hidedecorations!(ax)
    end
    resize_to_layout!(f)
    f, axs
end

"""
Interactive view of the qubit lattice on `ax`: left-clicking an edge toggles its value,
allowing exploration of the amplitudes of different states. `f` must be the `Figure`
containing `ax`; it is needed for pixel-accurate picking.

Caller is responsible for cleanup (hiding decorations, `DataInspector`, etc.).

Forwards `qubitdisplayspec` and `tail_length` to `visualize`.
"""
function FibErrThresh.visualize(
    f::Figure,
    ax::Axis,
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f},
    inds::Vector{IndexLabel},
    data::AbstractArray;
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_length::Real=0.5,
)
    # helper function that calculates amplitude from qubitvals
    function get_amp(qubitvals::Dict{Int, Int})
        _, vals = qubitvals2idxvals(ql, qubitvals; inds=inds)
        try
            data[vals...]
        catch e
            # if we left the vertex-constraint subspace
            e isa BoundsError ? 0.0 : rethrow(e)
        end
    end
    # plot all 0 state and put amplitude
    qubitvals = Dict(q => 0 for q in get_qubits(ql))
    qubits, segmentsresult = visualize(ax, ql, position_from_index, qubitvals;
        qubitdisplayspec=qubitdisplayspec, tail_length=tail_length)
    ax.title="$(get_amp(qubitvals))"
    # click callback function
    deregister_interaction!(ax, :rectanglezoom)
    toggled = Set{Int}()
    register_interaction!(ax, :toggle_qubit) do event::MouseEvent, ax
        # discard non-leftdrag
        if event.type == MouseEventTypes.leftup
            empty!(toggled)
            return
        end
        if event.type != MouseEventTypes.leftdrag &&
           event.type != MouseEventTypes.leftdown return end
        # get the segment that was clicked, and check that it was our segments plot
        p, i = pick(f, 20)
        if p != segmentsresult return end
        # get selected qubit
        idx = i ÷ 2
        q = qubits[idx]
        if q ∈ toggled return end
        # toggle selected qubit
        qubitvals[q] = 1 - qubitvals[q]
        push!(toggled, q)
        # update the ax title and segment display properties
        ax.title="$(get_amp(qubitvals))"
        colors_observable = segmentsresult.kw[:color]
        colors_observable[][idx] = qubitdisplayspec[:color][qubitvals[q]]
        notify(colors_observable)
    end
    nothing
end

"""
Interactive view of the qubit lattice: left-clicking an edge toggles its value,
allowing exploration of the amplitudes of different states. Creates and returns a
new `Figure` and `Axis`.

Forwards `qubitdisplayspec` and `tail_length` to `visualize`.
"""
function FibErrThresh.visualize(
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f},
    inds::Vector{IndexLabel},
    data::AbstractArray;
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_length::Real=0.5,
)
    f = Figure()
    ax = Axis(f[1, 1]; aspect=DataAspect())
    visualize(f, ax, ql, position_from_index, inds, data;
        qubitdisplayspec=qubitdisplayspec, tail_length=tail_length)
    DataInspector(f, range=20)
    hidespines!(ax)
    hidedecorations!(ax)
    resize_to_layout!(f)
    f, ax
end

### UTILS ###

"""
Computes the width and height of a rectangle whose area is
at least `area` and whose sidelengths are integers that have
close to the same values.
"""
function _calculategridsidelengths(area::Int)
    width = height = floor(sqrt(area))
    if width == sqrt(area) return (Int(width), Int(height)) end
    width += 1
    while width * height < area
        width += 1
    end
    Int(width), Int(height)
end
