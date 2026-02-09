using FibTN.Segments

@testset "Segment basics" begin
    # not testing all inputs rigorously, just some combinations
    # and making sure things run
    vv = Segment(Vertex, true, true, Vertex)
    vt = Segment(Vertex, true, true, Tail)
    tv = Segment(Tail, false, false, Vertex)
    tt = Segment(Tail, false, false, Tail)
    s = Segment(Vertex, false, true, Tail)

    vvdata = segment_data(vv)
    vtdata = segment_data(vt)
    tvdata = segment_data(tv)
    ttdata = segment_data(tt)
    sdata = segment_data(s)
end
