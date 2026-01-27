using FibTN.Indices
using FibTN.TensorNetworks
using FibTN.Executor
using FibTN.QubitLattices
using FibTN.Visualizer

using FibTN.FibTensorTypes

using SparseArrayKit

@testset "integration 1-plaquettes 3-sides" begin
    # build network
    tn = TensorNetwork()
    add_tensor!(tn, TensorLabel(1, index_labels(Tail, 1)))
    add_tensor!(tn, TensorLabel(2, index_labels(Tail, 2)))
    add_tensor!(tn, TensorLabel(3, index_labels(Tail, 3)))
    add_tensor!(tn, TensorLabel(4, index_labels(LoopAmplitude, 4)))
    add_contraction!(tn, IndexPair(IndexLabel(1, :b), IndexLabel(4, :a)))
    add_contraction!(tn, IndexPair(IndexLabel(4, :b), IndexLabel(2, :a)))
    add_contraction!(tn, IndexPair(IndexLabel(2, :b), IndexLabel(3, :a)))
    add_contraction!(tn, IndexPair(IndexLabel(3, :b), IndexLabel(1, :a)))
    # create execution network and contract
    T1 = SparseArray(tensor_data(Tail))
    T2 = SparseArray(tensor_data(Tail))
    T3 = SparseArray(tensor_data(Tail))
    T4 = SparseArray(tensor_data(LoopAmplitude))
    en = ExecNetwork(tn, Dict(1 => T1, 2 => T2, 3 => T3, 4 => T4))
    for c in tn.contractions
        @show c
        execute_step!(en, Contraction(c))
    end
    et = execute_step!(en, FetchResult())
    @show et.data
    @show et.indices
end

@testset "integration 2-plaquettes 3-sides" begin

end

@testset "integration 3-plaquettes 3-sides" begin

end

@testset "integration 1-plaquettes 6-sides" begin

end

@testset "integration 2-plaquettes 6-sides" begin

end

@testset "integration 3-plaquettes 6-sides" begin

end

