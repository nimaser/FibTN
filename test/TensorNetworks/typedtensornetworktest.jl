using FibTN.TensorNetworks

import FibTN.TensorNetworks: tensor_ports, tensor_data
abstract type DummyTensorType <: TensorType end
struct DummyTensorType2 <: TensorType end
tensor_data(::Type{DummyTensorType}) = [1, 2, 3]
tensor_ports(::Type{DummyTensorType}) = (:a,)
tensor_data(::Type{DummyTensorType2}) = [0, 3, 6]
tensor_ports(::Type{DummyTensorType2}) = (:b,)

@testset "TypedTensorNetworks basics" begin
    # construction, adding tensors, adding contractions
    ttn = TypedTensorNetwork()
    @test length(ttn.tensortype_from_group) == 0
    add_tensor!(ttn, 1, DummyTensorType)
    add_tensor!(ttn, 2, DummyTensorType2)
    ic = IndexContraction(IndexLabel(1, :a), IndexLabel(2, :b))
    add_contraction!(ttn.tn, ic)
    # fetching data
    @test tensordata_from_group(ttn) == Dict(
        1 => tensor_data(DummyTensorType),
        2 => tensor_data(DummyTensorType2),
    )
end
