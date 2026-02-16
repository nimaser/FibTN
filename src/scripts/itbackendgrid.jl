using GLMakie
using FibTN
using FibTN.SegmentGrids

using FibTN.TensorNetworks.ITBackend

function itcontract1()
    sg = SegmentGrid(2, 2)
    indices, itensors = get_itensor_network(sg.ttn)
    res = optimized_exact_contract(collect(values(itensors)))
end

function itcontract2()
    sg = SegmentGrid(2, 2)
    indices, itensors = get_itensor_network(sg.ttn)
    res = optimized_exact_contract(collect(values(itensors)))
end
