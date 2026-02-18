using GLMakie
using FibErrThresh
using FibErrThresh.TensorNetworks
using FibErrThresh.FibTNs

using FibErrThresh.TensorNetworks.TOBackend
using SparseArrayKit

function tocontract1()
    sg = FibTN(3, 4)
    csteps = [ContractionStep(c) for c in get_contractions(sg.ttn.tn)]
    sg, do_contractions(sg, csteps)...
end

function tocontract2()
    ftn = FibTN(3, 4)
    csteps = Vector{ContractionStep}()
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(1, :U))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(2, :U))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(3, :U))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(4, :U))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(5, :U))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(6, :U))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(7, :U))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(8, :U))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(9, :U))))

    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(1, :R))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(4, :R))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(7, :R))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(10, :R))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(2, :R))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(5, :R))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(8, :R))))
    push!(csteps, ContractionStep(get_contraction(ftn.ttn.tn, IndexLabel(11, :R))))

    ftn, do_contractions(ftn, csteps)...
end

function tocontract3()
    ftn = FibTN(2, 8)
    csteps = [ContractionStep(c) for c in get_contractions(ftn.ttn.tn)]
    ftn, do_contractions(ftn, csteps)...
end

function tocontract4()
    ftn = FibTN(8, 2)
    csteps = [ContractionStep(c) for c in get_contractions(ftn.ttn.tn)]
    ftn, do_contractions(ftn, csteps)...
end

function tocontract5()
    ftn = FibTN(14, 2)
    csteps = [ContractionStep(c) for c in get_contractions(ftn.ttn.tn)]
    ftn, do_contractions(ftn, csteps)...
end

function do_contractions(ftn::FibTN, csteps::Vector{ContractionStep})
    es = ExecutionState(ftn.ttn)
    for cstep in csteps
         @show cstep.a, cstep.b
         @show Base.summarysize(es)

         t1 = get_tensor(es, cstep.a)
         @show length(nonzero_pairs(t1.data))
         @show Base.summarysize(t1)

         t2 = get_tensor(es, cstep.b)
         @show length(nonzero_pairs(t2.data))
         @show Base.summarysize(t2)

         @show @allocated execute_step!(es, cstep)
         print("\n")
    end
    et = es.tensor_from_id[only(get_ids(es))]
    inds, data = et.indices, et.data
    inds, data
end
