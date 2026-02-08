using FibTN.FibTensorTypes
using FibTN.FibTensorNetworks

@testset "FibTensorNetworks basics" begin
    # construction, adding tensors, adding contractions
    ftn = FibTensorNetwork()
    @test length(ftn.tensortype_from_group) == 0
    add_tensor!(ftn, 1, Tail)
    add_tensor!(ftn, 2, Reflector)
    add_tensor!(ftn, 3, Vertex)
    ic1 = IndexContraction(IndexLabel(1, :b), IndexLabel(2, :a))
    ic2 = IndexContraction(IndexLabel(2, :b), IndexLabel(3, :a))
    add_contraction!(ftn.tn, ic1)
    add_contraction!(ftn.tn, ic2)
    # fetching data
    @test tensordata_from_group(ftn) = Dict(
        1 => tensor_data(Tail),
        2 => tensor_data(Reflector),
        3 => tensor_data(Vertex)
    )
end
