using FibTN.Segments

@testset "Segment basics" begin
    # not testing all inputs rigorously, just some combinations
    # and making sure things run
    vv = Segment(Vector, true, true, Vector)
    vt = Segment(Vector, true, true, Tail)
    tv = Segment(Tail, false, false, Vector)
    tt = Segment(Tail, false, false, Tail)
    s = Segment(Vertex, false, true, Tail)

    vvdata = tensor_data(vv)
    vtdata = tensor_data(vt)
    tvdata = tensor_data(tv)
    ttdata = tensor_data(tt)
    sdata = tensor_data(s)
end
