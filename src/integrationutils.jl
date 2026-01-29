using FibTN.Indices
using FibTN.TensorNetworks
using FibTN.Executor
using FibTN.QubitLattices
using FibTN.Visualizer
using FibTN.FibTensorTypes

using GLMakie

using SparseArrayKit

### INDEX HELPERS ###

const IP = IndexPair
const IL = IndexLabel
index_labels(::Type{T}, group::Int) where T <: AbstractFibTensorType = [IL(group, p) for p in tensor_ports(T)]

### TN HELPERS ###

make_g2tt(tt2gs::Dict{DataType, Vector{Int}}) = Dict(g => tt for (tt, gs) in tt2gs for g in gs)

function build_tn(g2tt::Dict{Int, DataType}, contractions::Vector{IP})
    tn = TensorNetwork()
    for (g, tt) in g2tt add_tensor!(tn, TensorLabel(g, index_labels(tt, g))) end
    for ip in contractions add_contraction!(tn, ip) end
    tn
end

function in_edge_contractions(tn::TensorNetwork, g2tt::Dict{Int, DataType})
    contractions = Vector{IP}
    for (g, tt) in g2tt
        if tt ∈ Set(Reflection, Boundary, VacuumLoop)
            idx = first(tensor_with_group(tn, g).indices)
            push!(contractions, tn.contraction_with_index(idx))
        end
    end
    contractions
end
    
pinds(tn::TensorNetwork) = filter(idx -> idx.port == :p, collect(indices(tn)))

### CONTRACTION HELPERS ###

function contractionchain(n1::Int, n2::Int, s1::Symbol, s2::Symbol)
    contractions = [IP(IL(i, s1), IL(i+1, s2)) for i in n1:n2-1]
end

### EXECUTION HELPERS ###

build_en(tn::TensorNetwork, g2tt::Dict{Int, DataType}) = ExecNetwork(tn, Dict(g=>tensor_data(tt) for (g,tt) in g2tt))

do_steps(en::ExecNetwork, execsteps::Vector{ExecStep}) = for es in execsteps execute_step!(en, es) end

fetch_result(en::ExecNetwork) = begin et = execute_step!(en, FetchResult()); et.indices, et.data end

function specified_contract(tn::TensorNetwork, g2tt::Dict{int, DataType}, contractions::Vector{IP})
    execsteps = [Contraction(c) for c in contractions]
    en = build_en(tn, g2tt)
    do_steps(en, execsteps)
    fetch_result(en)
end

naive_contract(tn::TensorNetwork, g2tt::Dict{Int, DataType}) = specified_contract(tn, g2tt, tn.contractions)

function in_edge_first_contract(tn::TensorNetwork, g2tt::Dict{Int, DataType})
    in_edge = in_edge_contractions(tn, g2tt)
    others = [c for c in setdiff(Set(tn.contractions), Set(in_edge))]
    contractions = append(in_edge, others)
    specified_contract(tn, g2tt, contractions)
end

### QL HELPERS ###

function build_ql(qubits_from_index::Dict{IL, Vector{Int}})
    ql = QubitLattice()
    for (idx, q) in qubits_from_index add_index!(ql, idx, q) end
    ql
end

function get_states_and_amps(ql::QubitLattice, inds::Vector{IndexLabel}, data::SparseArray)
    lattice_states, amplitudes = Vector{Dict{Int, Int}}(), Vector{Real}()
    for (cidx, amp) in nonzero_pairs(data)
        push!(lattice_states, get_lattice_state(ql, inds, [Tuple(cidx)...]))
        push!(amplitudes, amp)
    end
    lattice_states, amplitudes
end

### GEOMETRY HELPERS ###

# basic 2D point type alias
const Point2 = NTuple{2, Float64}

# translate a set of points
translate(points::Vector{Point2}, c::Point2) = [(x + c[1], y + c[2]) for (x, y) in points]

# rotate points about origin
rotate(points::Vector{Point2}, θ::Real) = [(cos(θ)*x - sin(θ)*y, sin(θ)*x + cos(θ)*y) for (x, y) in points]

# scale points about origin
scale(points::Vector{Point2}, s::Real) = [(s*x, s*y) for (x, y) in points]

# counterclockwise vertices of an n-gon of circumradius r, center c, and phase θ
function regular_polygon(
    n::Int;
    r::Real = 1.0,
    c::Point2 = (0.0, 0.0),
    θ::Real = 0.0,
)
    pts = [(r*cos(θ + 2π*k/n), r*sin(θ + 2π*k/n)) for k in 0:n-1]
    translate(pts, c)
end

triangle(; kwargs...)  = regular_polygon(3; kwargs...)
square(; kwargs...)    = regular_polygon(4; phase=π/4, kwargs...)
pentagon(; kwargs...)  = regular_polygon(5; kwargs...)
hexagon(; kwargs...)   = regular_polygon(6; kwargs...)

# n points uniformly spaced from a to b (inclusive)
function line_segment(a::Point2, b::Point2, n::Int)
    [( (1-t)*a[1] + t*b[1], (1-t)*a[2] + t*b[2] )
     for t in range(0, 1; length=n)]
end

# generate n points in a zig-zag line
function zigzag(
    n::Int;
    step::Real = 1.0,
    amplitude::Real = 1.0,
    origin::Point2 = (0.0, 0.0),
)
    pts = Vector{Point2}(undef, n)
    for i in 1:n
        x = step * (i-1)
        y = amplitude * ((i-1) % 2)
        pts[i] = (x, y)
    end
    translate(pts, origin)
    pts
end

function insert_midpoints(
    pts::Vector{Point2};
    counts::Vector{Int} = fill(1, length(pts) - 1)
)
    n = length(pts)
    n ≥ 2 || error("need at least two points")
    length(counts) == n - 1 || error("counts must have length length(pts)-1")

    out = Point2[]
    push!(out, pts[1])

    for i in 1:n-1
        (x1, y1) = pts[i]
        (x2, y2) = pts[i+1]
        k = counts[i]

        Δx = x2 - x1
        Δy = y2 - y1
        for j in 1:k
            t = j / (k + 1)

            push!(out, (
                        x1 + Δx * t,
                        y1 + Δy * t,
                       )
                 )
        end

        push!(out, float.(pts[i+1]))
    end
    out
end

### POST PROCESSING ###

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

color_from_qubit(lattice_state::Dict{Int, Int}) = Dict(q => v == 1 ? :red : :black for (q, v) in lattice_state)

position_from_index(indices::Vector{IL}, positions::Vector{Point2}) = Dict(i => positions[i.group] for i in indices)

function plot(tn::TensorNetwork, positions::Vector{Point2}, g2tt::Dict{Int, DataType})
    groups = sort(collect(keys(g2tt)))
    colors = [tt2color(g2tt[g]) for g in groups]
    markers = [tt2marker(g2tt[g]) for g in groups]
    tnds = TensorNetworkDisplaySpec(groups, positions, colors, markers)
    
    f = Figure()
    ax = Axis(f[1, 1])
    hidedecorations!(ax)
    hidespines!(ax)
    visualize(tn, tnds, ax)
    DataInspector(f, range=30)
    display(f)
end

function plot(ql::QubitLattice, positions::Vector{Point2}, s::Dict{Int, Int}) 
    qlds = QubitLatticeDisplaySpec(position_from_index(indices(ql), positions), color_from_qubit(s), 0.5)

    f = Figure()
    ax = Axis(f[1, 1])
    hidedecorations!(ax)
    hidespines!(ax)
    visualize(ql, qlds, ax)
    DataInspector(f, range=30)
    display(f)
end
