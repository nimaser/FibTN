using FibErrThresh.TensorNetworks

import FibErrThresh.TensorNetworks: tensor_ports, tensor_data
abstract type DummyTensorType <: TensorType end
struct DummyTensorType2 <: TensorType end
struct DummyTensorType3 <: TensorType end
struct DummyTensorType4 <: TensorType end
tensor_ports(::Type{DummyTensorType}) = (:a,)
tensor_ports(::Type{DummyTensorType2}) = (:b,)
tensor_ports(::Type{DummyTensorType3}) = (:a, :c)
tensor_ports(::Type{DummyTensorType4}) = (:a, :c, :d)
tensor_data(::Type{DummyTensorType}) = [1, 2, 3]
tensor_data(::Type{DummyTensorType2}) = [0, 3, 6]

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

@testset "TypedTensorNetwork replace_tensor!" begin
    # three tensors: group 1 (:a, :c) contracted with group 2 (:b) and group 3 (:b)
    ttn = TypedTensorNetwork()
    add_tensor!(ttn, 1, DummyTensorType3)
    add_tensor!(ttn, 2, DummyTensorType2)
    add_tensor!(ttn, 3, DummyTensorType2)
    add_contraction!(ttn.tn, IndexContraction(IndexLabel(1, :a), IndexLabel(2, :b)))
    add_contraction!(ttn.tn, IndexContraction(IndexLabel(1, :c), IndexLabel(3, :b)))

    # replace group 1 (DummyTensorType3) with DummyTensorType4, keeping only the :a contraction
    replace_tensor!(ttn, 1, DummyTensorType4; preserve_contractions=[:a])
    @test ttn.tensortype_from_group[1] == DummyTensorType4
    @test has_contraction(ttn.tn, IndexLabel(1, :a))
    @test !has_contraction(ttn.tn, IndexLabel(1, :c))
    @test has_index(ttn.tn, IndexLabel(1, :d))
    @test length(get_contractions(ttn.tn)) == 1
end
