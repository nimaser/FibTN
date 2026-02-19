using GLMakie
using FibErrThresh
using FibErrThresh.FibTNs

using FibErrThresh.TensorNetworks.ITBackend

function itcontract1()
    sg = FibTN(2, 2)
    indices, itensors = get_itensor_network(sg.ttn)
    res = optimized_exact_contract(collect(values(itensors)))
end

function itcontract2()
    sg = FibTN(2, 2)
    indices, itensors = get_itensor_network(sg.ttn)
    res = optimized_exact_contract(collect(values(itensors)))
end
