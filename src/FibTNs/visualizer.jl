using FibErrThresh.FibTNs
using FibErrThresh.SegmentTensorTypes
using FibErrThresh.TensorNetworks
const IL = IndexLabel

using GLMakie, GeometryBasics

export visualize

"""
Plots the tensor network of `ftn` on `tax` and sets an inspector label showing,
for each tensor: segments: `g`, `(x,y)`, and the mask string;
crossings/reflectors: `g`, `edge`, `1-based index in the chain`;
fusions: `g`, `gridpos`, `1-based index in the fusion list`.
Returns the scatter plot result.
"""
function _visualize_ttn(tax::Axis, ftn::FibTN)
    _, scatterresult = visualize(tax, ftn.ttn, ftn.tpos)

    # visualize sorts scatter points by group, so sorted_groups[i] == group of i-th point
    sorted_groups = sort(collect(get_groups(ftn.ttn.tn)))
    gpos_from_group = Dict(s.group => s.gpos for (_, s) in ftn.segments)

    scatterresult.inspector_label = (plot, i, idx) -> begin
        g = sorted_groups[i]
        if haskey(gpos_from_group, g)
            gpos = gpos_from_group[g]
            T = get_segmenttensortype(ftn.segments[gpos...])
            "$(g)\n$(gpos)\n$(hex2mask(getmask(T)))"
        elseif haskey(ftn.edgecrossing_from_group, g)
            edge, k = ftn.edgecrossing_from_group[g]
            "$(g)\ncrossing $(k) on $(edge)"
        elseif haskey(ftn.edgereflector_from_group, g)
            edge, k = ftn.edgereflector_from_group[g]
            "$(g)\nreflector $(k) on $(edge)"
        elseif haskey(ftn.fusion_from_group, g)
            pos, k = ftn.fusion_from_group[g]
            "$(g)\nfusion $(k) at $(pos)"
        else
            T = ftn.ttn.tensortype_from_group[g]
            "$(g)\n$(T)"
        end
    end

    scatterresult
end

"""
Wires the FibTN hover interaction onto pre-created axes `tax` (tensor network) and
`qax` (qubit lattice): mousing over a segment in `tax` highlights its qubits in `qax`.
`f` is the root `Figure`, needed for pixel-accurate `pick()`. Links the axes so
zoom/pan stays in sync. Hides decorations on both axes.

Callers are responsible for creating `tax` and `qax` at whatever grid positions they
need. This is the single place that owns the ttn plot + ql plot + hover-highlight
interaction.
"""
function _wire_ftn_axes!(f::Figure, tax::Axis, qax::Axis, ftn::FibTN)
    scatterresult = _visualize_ttn(tax, ftn)
    qubitvals_obs, _, _ = visualize(qax, ftn.ql, ftn.ipos)

    sorted_groups = sort(collect(get_groups(ftn.ttn.tn)))
    gpos_from_group = Dict(s.group => s.gpos for (_, s) in ftn.segments)
    zeros_dict = Dict(q => 0 for q in get_qubits(ftn.ql))
    last_highlighted = Ref(Set{Int}())

    register_interaction!(tax, :highlight_qubits) do event::MouseEvent, ax
        event.type == MouseEventTypes.over || return
        p, i = pick(f, 30)
        # determine which segment (if any) is under the cursor
        new_highlighted = Set{Int}()
        if p == scatterresult && i != 0
            g = sorted_groups[i]
            if haskey(gpos_from_group, g)
                for port in (:TP, :MP, :BP)
                    il = IL(g, port)
                    haskey(ftn.ql.qubits_from_index, il) || continue
                    for q in get_qubits(ftn.ql, il)
                        q != 0 && push!(new_highlighted, q)
                    end
                end
            end
        end
        new_highlighted == last_highlighted[] && return
        last_highlighted[] = new_highlighted
        # build a fresh dict: 1 for highlighted qubits, 0 for all others
        new_qubitvals = copy(zeros_dict)
        for q in new_highlighted
            new_qubitvals[q] = 1
        end
        qubitvals_obs[] = new_qubitvals
    end

    linkaxes!(tax, qax)
    for ax in (tax, qax)
        hidespines!(ax)
        hidedecorations!(ax)
    end
    nothing
end

"""
Plots `ftn`: the qubit lattice (left) and tensor network (right), side by side and
hover-linked so mousing over a segment highlights its qubits.
Returns `(f, tax, qax)`.
"""
function FibErrThresh.visualize(ftn::FibTN)
    f = Figure()
    qax = Axis(f[1, 1]; aspect=DataAspect())
    tax = Axis(f[1, 2]; aspect=DataAspect())
    _wire_ftn_axes!(f, tax, qax, ftn)
    DataInspector(f, range=30)
    resize_to_layout!(f)
    f, tax, qax
end

"""
Plots `ftn` with an interactive results explorer on the right: the stacked qubit
lattice and tensor network occupy col -1 (left), a thin vertical bar at col 0, and a
clickable qubit lattice showing the amplitude of the selected state at col 1 (right).
`inds` and `data` should come from contracting `ftn` (e.g. via `naive_contract`).
Returns `(f, tax, qax, rax)`.
"""
function FibErrThresh.visualize(ftn::FibTN, inds::Vector{IndexLabel}, data::AbstractArray)
    f = Figure()
    rax = Axis(f[1, 1]; aspect=DataAspect())
    visualize(f, rax, ftn.ql, ftn.ipos, inds, data)
    hidespines!(rax)
    hidedecorations!(rax)
    Box(f[1, 0]; color=:gray50, strokewidth=0)
    colsize!(f.layout, 0, Fixed(2))
    gl = GridLayout(f[1, -1])
    qax = Axis(gl[1, 1]; aspect=DataAspect())
    tax = Axis(gl[2, 1]; aspect=DataAspect())
    _wire_ftn_axes!(f, tax, qax, ftn)
    add_inspector_lazily!(f)
    resize_to_layout!(f)
    f, tax, qax, rax
end

"""
Plots `ftn` with a scrollable paned results view: the figure is split into two equal
halves — left (col 1) has the stacked qubit lattice and tensor network, right (col 3)
has the paned qubit lattice showing multiple states.
Forwards keyword arguments to the underlying `visualize` for the paned qubit lattice.
Returns `(f, tax, qax, pane_axs)`.
"""
function FibErrThresh.visualize(ftn::FibTN, states::Vector{Dict{Int,Int}},
        amps::Vector{<:Number}; kwargs...)
    f = Figure()
    # left half: stacked ttn (top) + ql (bottom)
    left_gl = GridLayout(f[1, 1])
    qax = Axis(left_gl[1, 1]; aspect=DataAspect())
    tax = Axis(left_gl[2, 1]; aspect=DataAspect())
    _wire_ftn_axes!(f, tax, qax, ftn)
    # thin vertical separator
    Box(f[1, 2]; color=:gray50, strokewidth=0)
    colsize!(f.layout, 2, Fixed(2))
    # right half: paned QL view
    right_gl = GridLayout(f[1, 3])
    pane_axs, _, _ = visualize(right_gl, ftn.ql, ftn.ipos, states, amps; kwargs...)
    # equal halves: col 1 and col 3 each get half the width; col 2 is fixed at 2px
    colsize!(f.layout, 1, Relative(0.5))
    colsize!(f.layout, 3, Relative(0.5))
    resize_to_layout!(f)
    f, tax, qax, pane_axs
end

"""
Plots `ftn` with both an interactive results explorer and a paned results view. The
figure is split into two equal halves by a thin separator:
- Left half (col 1): a GridLayout with two rows:
  - Top: qubit lattice (left) and tensor network (right), hover-linked
  - Bottom: clickable qubit lattice showing the amplitude of the selected state
- Right half (col 3): scrollable paned qubit lattice showing multiple states

`inds` and `data` should come from contracting `ftn` (e.g. via `naive_contract`).
`states` and `amps` are the states to display in the paned view.
Forwards `kwargs` to the underlying paned `visualize`.
Returns `(f, tax, qax, rax, pane_axs)`.
"""
function FibErrThresh.visualize(ftn::FibTN,
        inds::Vector{IndexLabel}, data::AbstractArray,
        states::Vector{Dict{Int,Int}}, amps::Vector{<:Number}; kwargs...)
    f = Figure()
    # left half: top row has qax+tax hover-linked; bottom row has interactive amplitude explorer
    left_gl = GridLayout(f[1, 1])
    qax = Axis(left_gl[1, 1]; aspect=DataAspect())
    tax = Axis(left_gl[1, 2]; aspect=DataAspect())
    _wire_ftn_axes!(f, tax, qax, ftn)
    rax = Axis(left_gl[2, 1:2]; aspect=DataAspect())
    visualize(f, rax, ftn.ql, ftn.ipos, inds, data)
    hidespines!(rax)
    hidedecorations!(rax)
    # thin vertical separator
    Box(f[1, 2]; color=:gray50, strokewidth=0)
    colsize!(f.layout, 2, Fixed(2))
    # right half: paned QL view
    right_gl = GridLayout(f[1, 3])
    pane_axs, _, _ = visualize(right_gl, ftn.ql, ftn.ipos, states, amps; kwargs...)
    # equal halves: col 1 and col 3 each get half the width; col 2 is fixed at 2px
    colsize!(f.layout, 1, Relative(0.5))
    colsize!(f.layout, 3, Relative(0.5))
    add_inspector_lazily!(f)
    resize_to_layout!(f)
    f, tax, qax, rax, pane_axs
end
