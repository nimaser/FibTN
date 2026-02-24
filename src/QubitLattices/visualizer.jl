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
Plots a qubit lattice on `ax`, with qubit values specified by `qubitvals_obs`.
The 0 qubit is not displayed. Unpaired qubits are shown as tails in the direction
of `tail_vector`.

Sets up an `on(qubitvals_obs)` callback so that pushing a new `Dict{Int,Int}` to the
observable automatically updates edge colors without replotting.

Custom display properties for qubit values 0 and 1 should be provided via
the `qubitdisplayspec` parameter.

Returns `(qubits, segmentsresult)` where `qubits` is the deterministic ordering used
internally (needed by callers that pick by index, e.g. the interactive toggle view).
"""
function FibErrThresh.visualize(
    ax::Axis,
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f},
    qubitvals_obs::Observable{Dict{Int, Int}};
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_vector::Point2f=Point2f(0.3, 0),
)
    # build deterministically ordered qubit list and static edge geometry
    qubits = collect(get_qubits(ql))
    filter!(!=(0), qubits)
    edge_endpoints = Vector{Tuple{Point2f, Point2f}}()
    for q in qubits
        inds = get_indices(ql, q)
        if length(inds) == 1
            pos1 = position_from_index[inds[1]]
            pos2 = pos1 + tail_vector
        else
            length(inds) == 2 || error("more than two indices sharing qubit $q")
            pos1 = position_from_index[inds[1]]
            pos2 = position_from_index[inds[2]]
        end
        push!(edge_endpoints, (pos1, pos2))
    end
    # build initial colors from the current observable value
    qv = qubitvals_obs[]
    colors = [qubitdisplayspec[:color][qv[q]] for q in qubits]
    colors_obs = Observable(colors)
    segmentsresult = linesegments!(ax, edge_endpoints, color=colors_obs, linewidth=5)
    # inspector label reads live qubitvals from the observable
    segmentsresult.inspector_label = (plot, i, idx) -> begin
        i = i ÷ 2 # i counts by endpoints (two per segment)
        q = qubits[i]
        "qubit $q\n|$(qubitvals_obs[][q])⟩"
    end
    # whenever qubitvals_obs is updated, recolor edges without replotting
    on(qubitvals_obs) do qv
        for (idx, q) in enumerate(qubits)
            colors_obs[][idx] = qubitdisplayspec[:color][get(qv, q, 0)]
        end
        notify(colors_obs)
    end
    autolimits!(ax) # set limits once based on geometry
    qubits, segmentsresult
end

"""
Plots `ql` on `ax` with the given `qubitvals` dict.

Wraps `qubitvals` in an `Observable` and calls the observable overload.
Returns `(qubitvals_obs, qubits, segmentsresult)` so the caller can push
updates to the observable later.
"""
function FibErrThresh.visualize(
    ax::Axis,
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f},
    qubitvals::Dict{Int, Int};
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_vector::Point2f=Point2f(0.3, 0),
)
    qubitvals_obs = Observable(qubitvals)
    qubits, segmentsresult = visualize(ax, ql, position_from_index, qubitvals_obs;
        qubitdisplayspec=qubitdisplayspec, tail_vector=tail_vector)
    qubitvals_obs, qubits, segmentsresult
end

"""
Plots `ql` on `ax` with all qubits set to 0.

Returns `(qubitvals_obs, qubits, segmentsresult)` so the caller can push
updates to the observable to highlight or change qubit states.
"""
function FibErrThresh.visualize(
    ax::Axis,
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f};
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_vector::Point2f=Point2f(0.3, 0),
)
    qubitvals_obs = Observable(Dict(q => 0 for q in get_qubits(ql)))
    qubits, segmentsresult = visualize(ax, ql, position_from_index, qubitvals_obs;
        qubitdisplayspec=qubitdisplayspec, tail_vector=tail_vector)
    qubitvals_obs, qubits, segmentsresult
end


"""
Plots `states` and corresponding `amps` onto a figure with a scrollable set of 'panes',
each of which can contain at most `maxstatesperpane` states. Left and right arrow keys
can be used to change between panes.

Each axis slot is plotted once; pane switching updates qubit colors via observables
without replotting, so axis limits and layout remain fixed.

Returns a Makie `Figure` and array of axes.

Forwards `qubitdisplayspec` and `tail_vector` to `visualize`.
"""
function FibErrThresh.visualize(
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f},
    states::Vector{Dict{Int, Int}},
    amps::Vector{<:Number};
    maxstatesperpane::Int=3,
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_vector::Point2f=Point2f(0.5, 0),
)
    # distribute states as evenly as possible across panes
    # using exactly maxstatesperpane states per pane
    numpanes = Int(ceil(length(states) / maxstatesperpane))
    # the last pane might be mostly empty, in which case we
    # can keep the same number of panes while reducing the
    # number of states per pane by moving some states to
    # the last one from the others so they all have roughly
    # the same amount, then calculate the width and height
    # of each pane
    statesperpane = Int(ceil(length(states) / numpanes))
    # w*h could be larger than stateperpane, so to make
    # sure all panes are fully filled we do
    w, h = _calculategridsidelengths(statesperpane)
    # and finally we update numpanes
    statesperpane = w * h
    numpanes = Int(ceil(length(states) / statesperpane))

    # create the figure and axes
    f = Figure()
    axs = [Axis(f[r, c]; aspect=DataAspect()) for r in 1:h, c in 1:w]
    for ax in axs ax.titlesize = 32 end
    linkaxes!(axs...)

    # compute normalization factor and add to title
    N = sqrt(sum(abs2.(amps)))
    f[0, :] = Label(f, L"N = %$N", fontsize=24, tellwidth=false)

    # plot the QL once per axis; store observable (for state updates) and plot (for visibility)
    slot_obs = Vector{Observable}(undef, length(axs))
    slot_plots = Vector{Any}(undef, length(axs))
    for i in eachindex(axs)
        obs, _, sr = visualize(axs[i], ql, position_from_index;
            qubitdisplayspec=qubitdisplayspec, tail_vector=tail_vector)
        slot_obs[i] = obs
        slot_plots[i] = sr
    end

    get_pane_states(pane::Int) = states[(pane-1)*statesperpane+1:min(pane*statesperpane, length(states))]
    get_pane_amps(pane::Int) = amps[(pane-1)*statesperpane+1:min(pane*statesperpane, length(amps))]

    # set up slider
    sg = SliderGrid(f[h+1, :],
        (label="Pane #", range=1:numpanes, startvalue=1, update_while_dragging=false))
    sl = sg.sliders[1]
    on(sl.value) do v
        pane_states = get_pane_states(v)
        pane_amps = get_pane_amps(v)
        for i in eachindex(axs)
            if i <= length(pane_states)
                axs[i].title = "$(pane_amps[i])"
                slot_obs[i][] = pane_states[i]
                slot_plots[i].visible = true
            else
                axs[i].title = ""
                slot_plots[i].visible = false
            end
        end
    end
    notify(sl.value)
    register_interaction!(axs[1], :change_pane) do event::KeysEvent, ax
        if Keyboard.left ∈ event.keys
            set_close_to!(sl, sl.value[] - 1)
        elseif Keyboard.right ∈ event.keys
            set_close_to!(sl, sl.value[] + 1)
        end
    end

    for ax in axs
        hidespines!(ax)
        hidedecorations!(ax)
    end
    add_inspector_lazily!(f)
    resize_to_layout!(f)
    f, axs
end

"""
Interactive view of the qubit lattice on `ax`: left-clicking an edge toggles its value,
allowing exploration of the amplitudes of different states. `f` must be the `Figure`
containing `ax`; it is needed for pixel-accurate picking.

Caller is responsible for cleanup (hiding decorations, `DataInspector`, etc.).

Forwards `qubitdisplayspec` and `tail_vector` to `visualize`.
"""
function FibErrThresh.visualize(
    f::Figure,
    ax::Axis,
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f},
    inds::Vector{IndexLabel},
    data::AbstractArray;
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_vector::Point2f=Point2f(0.5, 0),
)
    # helper: amplitude for a given qubitvals dict
    function get_amp(qubitvals::Dict{Int, Int})
        _, vals = qubitvals2idxvals(ql, qubitvals; inds=inds)
        try
            data[vals...]
        catch e
            e isa BoundsError ? 0.0 : rethrow(e) # out of vertex-constraint subspace
        end
    end
    # plot all-zero state; get observable and qubit ordering
    qubitvals_obs, qubits, segmentsresult = visualize(ax, ql, position_from_index;
        qubitdisplayspec=qubitdisplayspec, tail_vector=tail_vector)
    ax.title = "$(get_amp(qubitvals_obs[]))"
    # left-click toggles qubits; deregister default rectanglezoom first
    deregister_interaction!(ax, :rectanglezoom)
    toggled = Set{Int}()
    register_interaction!(ax, :toggle_qubit) do event::MouseEvent, ax
        if event.type == MouseEventTypes.leftup
            empty!(toggled)
            return
        end
        if event.type != MouseEventTypes.leftdrag &&
           event.type != MouseEventTypes.leftdown return end
        p, i = pick(f, 20)
        p == segmentsresult || return
        idx = i ÷ 2
        q = qubits[idx]
        q ∈ toggled && return
        # toggle qubit in-place in the observable's dict, then notify
        qubitvals_obs[][q] = 1 - qubitvals_obs[][q]
        push!(toggled, q)
        ax.title = "$(get_amp(qubitvals_obs[]))"
        notify(qubitvals_obs)
    end
    nothing
end

"""
Interactive view of the qubit lattice: left-clicking an edge toggles its value,
allowing exploration of the amplitudes of different states. Creates and returns a
new `Figure` and `Axis`.

Forwards `qubitdisplayspec` and `tail_vector` to `visualize`.
"""
function FibErrThresh.visualize(
    ql::QubitLattice,
    position_from_index::Dict{IndexLabel, Point2f},
    inds::Vector{IndexLabel},
    data::AbstractArray;
    qubitdisplayspec::Dict{Symbol, Dict{Int, Any}}=default_qubitdisplayspec(),
    tail_vector::Point2f=Point2f(0.5, 0),
)
    f = Figure()
    ax = Axis(f[1, 1]; aspect=DataAspect())
    visualize(f, ax, ql, position_from_index, inds, data;
        qubitdisplayspec=qubitdisplayspec, tail_vector=tail_vector)
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
