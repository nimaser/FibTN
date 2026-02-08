
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

color_from_qubit(qubitvals::Dict{Int, Int}) = Dict(q => v == 1 ? :red : :black for (q, v) in qubitvals)

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

function plot(ql::QubitLattice, positions::Vector{Point2}, qubitvals::Dict{Int, Int}, amp::Real)
    qlds = QubitLatticeDisplaySpec(position_from_index(collect(QubitLattices.get_indices(ql)), positions), color_from_qubit(qubitvals), 0.5)

    f = Figure()
    ax = Axis(f[1, 1])
    QubitLatticeVisualizer.visualize(ql, qlds, ax)
    ax.title = "$amp"
    DataInspector(f, range=30)
    finalize!(f, [ax])
    display(GLMakie.Screen(), f)
    ax
end

function plot_interactive(ql::QubitLattice, positions::Vector{Point2}, states::Vector{Dict{Int, Int}})
    qlds = QubitLatticeDisplaySpec(position_from_index(collect(QubitLattices.get_indices(ql)), positions), color_from_qubit(qubitvals), 0.5)

# f, ax, p = graphplot(g, edge_width=4, edge_color=[colorant"black" for i in 1:ne(g)])
# julia> function action(idx, event, axis)
#            p.edge_color[][idx] = rand(RGB)
#            p.edge_color[] = p.edge_color[]
#        end
# julia> register_interaction!(ax, :edgeclick, EdgeClickHandler(action))
end
