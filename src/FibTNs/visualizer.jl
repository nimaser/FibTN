using FibErrThresh.FibTNs
using FibErrThresh.SegmentTensorTypes

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
            "$(g)\nsegment at $(gpos) with mask $(hex2mask(getmask(T)))"
        elseif haskey(ftn.edgecrossing_from_group, g)
            edge, k = ftn.edgecrossing_from_group[g]
            "$(g)\ncrossing $(k) on $(edge)"
        elseif haskey(ftn.edgereflector_from_group, g)
            edge, k = ftn.edgereflector_from_group[g]
            "$(g)\nreflector $(k) on $(edge)"
        else
            pos, k = ftn.fusion_from_group[g]
            "$(g)\nfusion $(k) at $(pos)"
        end
    end

    scatterresult
end

"""
Plots the PEPS tensor network on `tax` and the QubitLattice on `qax`.
"""
function FibErrThresh.visualize(tax::Axis, qax::Axis, ftn::FibTN)
    _visualize_ttn(tax, ftn)
    visualize(qax, ftn.ql, ftn.ipos)
end

"""
Plots `ftn`, creating (and returning) a new Makie `Figure` and pair of `Axis` objects,
hiding axis decorations, adding the `DataInspector`, and ensuring the layout is fitted
correctly to the plotted objects.
"""
function FibErrThresh.visualize(ftn::FibTN)
    f = Figure()
    tax = Axis(f[1, 1]; aspect=DataAspect())
    qax = Axis(f[1, 2]; aspect=DataAspect())
    visualize(tax, qax, ftn)
    DataInspector(f, range=30)
    axs = [tax, qax]
    for ax in axs
        hidespines!(ax)
        hidedecorations!(ax)
    end
    resize_to_layout!(f)
    f, axs
end

"""
Plots the PEPS tensor network on `tax` and an interactive results explorer on `qax`.
Left-clicking an edge in `qax` toggles that qubit's value and updates the displayed
amplitude. `inds` and `data` should come from contracting `ftn` (e.g. via `naive_contract`).

`f` must be the `Figure` containing both axes; it is forwarded to the interactive
visualizer for pixel-accurate picking.
"""
function FibErrThresh.visualize(f::Figure, tax::Axis, qax::Axis, ftn::FibTN,
        inds::Vector{IndexLabel}, data::AbstractArray)
    visualize(tax, qax, ftn)
    visualize(f, qax, ftn.ql, ftn.ipos, inds, data)
    nothing
end

"""
Plots `ftn` with an interactive results explorer: the left axis shows the tensor
network and the right axis is a clickable qubit lattice that displays the amplitude
of the currently selected state. Creates and returns a new `Figure` and axes.
"""
function FibErrThresh.visualize(ftn::FibTN, inds::Vector{IndexLabel}, data::AbstractArray)
    f = Figure()
    tax = Axis(f[1, 1]; aspect=DataAspect())
    qax = Axis(f[1, 2]; aspect=DataAspect())
    visualize(f, tax, qax, ftn, inds, data)
    DataInspector(f, range=30)
    axs = [tax, qax]
    for ax in axs
        hidespines!(ax)
        hidedecorations!(ax)
    end
    resize_to_layout!(f)
    f, axs
end

"""
Plots all states in a paned view, with the tensor network on the left and the
scrollable paned qubit lattice on the right. The left axis spans the full height
of the pane grid.

Forwards keyword arguments to the underlying `visualize` for qubit lattices.
"""
function FibErrThresh.visualize(ftn::FibTN, states::Vector{Dict{Int,Int}},
        amps::Vector{<:Real}; kwargs...)
    # build the paned QL view first — it creates the figure and layout
    f, axs = visualize(ftn.ql, ftn.ipos, states, amps; kwargs...)
    # axs is an h×w matrix; insert TN axis at column 0, spanning rows 1:h
    h = size(axs, 1)
    tax = Axis(f[1:h, 0]; aspect=DataAspect())
    _visualize_ttn(tax, ftn)
    hidespines!(tax)
    hidedecorations!(tax)
    resize_to_layout!(f)
    f, tax, axs
end
