using GeometryBasics

# translate a set of points
translate(points::Vector{Point2f}, c::Point2f) = [(x + c[1], y + c[2]) for (x, y) in points]

# rotate points about origin
rotate(points::Vector{Point2f}, θ::Real) = [(cos(θ)*x - sin(θ)*y, sin(θ)*x + cos(θ)*y) for (x, y) in points]

# scale points about origin
scale(points::Vector{Point2f}, s::Real) = [(s*x, s*y) for (x, y) in points]

# clockwise vertices of an n-gon of circumradius r, center c, and phase θ
function regular_polygon(
    n::Int;
    r::Real = 1.0,
    c::Point2f = (0.0, 0.0),
    θ::Real = 0.0,
    duplicatefirst::Bool = false
)
    pts = [(r*cos(θ - 2π*k/n), r*sin(θ - 2π*k/n)) for k in 0:n-1]
    if duplicatefirst push!(pts, pts[1]) end
    translate(pts, c)
end

triangle(; kwargs...)  = regular_polygon(3; kwargs...)
square(; kwargs...)    = regular_polygon(4; kwargs...)
pentagon(; kwargs...)  = regular_polygon(5; kwargs...)
hexagon(; kwargs...)   = regular_polygon(6; kwargs...)

# n points uniformly spaced from a to b (inclusive)
function line_segment(a::Point2f, b::Point2f, n::Int)
    [( (1-t)*a[1] + t*b[1], (1-t)*a[2] + t*b[2] )
     for t in range(0, 1; length=n)]
end

# generate n points in a zig-zag line
function zigzag(
    n::Int;
    step::Real = 1.0,
    amplitude::Real = 1.0,
    origin::Point2f = (0.0, 0.0),
)
    pts = Vector{Point2f}(undef, n)
    for i in 1:n
        x = step * (i-1)
        y = amplitude * ((i-1) % 2)
        pts[i] = (x, y)
    end
    translate(pts, origin)
    pts
end

function insert_midpoints(
    pts::Vector{Point2f};
    counts::Vector{Int} = fill(1, length(pts) - 1),
)
    n = length(pts)
    n ≥ 2 || error("need at least two points")
    length(counts) == n - 1 || error("counts must have length length(pts)-1")

    out = Point2f[]
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
