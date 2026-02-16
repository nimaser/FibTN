using GLMakie
using FibTN
using FibTN.TensorNetworks
using FibTN.SegmentGrids

using FibTN.TensorNetworks.TOBackend
using SparseArrayKit

function tocontract1()
    sg = SegmentGrid(3, 4)
    csteps = [ContractionStep(c) for c in get_contractions(sg.ttn.tn)]
    sg, do_contractions(sg, csteps)...
end

function tocontract2()
    sg = SegmentGrid(3, 4)
    csteps = Vector{ContractionStep}()
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(1, :U))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(2, :U))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(3, :U))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(4, :U))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(5, :U))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(6, :U))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(7, :U))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(8, :U))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(9, :U))))

    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(1, :R))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(4, :R))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(7, :R))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(10, :R))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(2, :R))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(5, :R))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(8, :R))))
    push!(csteps, ContractionStep(get_contraction(sg.ttn.tn, IndexLabel(11, :R))))

    sg, do_contractions(sg, csteps)...
end

function tocontract3()
    sg = SegmentGrid(2, 8)
    csteps = [ContractionStep(c) for c in get_contractions(sg.ttn.tn)]
    sg, do_contractions(sg, csteps)...
end

function tocontract4()
    sg = SegmentGrid(8, 2)
    csteps = [ContractionStep(c) for c in get_contractions(sg.ttn.tn)]
    sg, do_contractions(sg, csteps)...
end

function tocontract5()
    sg = SegmentGrid(14, 2)
    csteps = [ContractionStep(c) for c in get_contractions(sg.ttn.tn)]
    sg, do_contractions(sg, csteps)...
end

function do_contractions(sg::SegmentGrid, csteps::Vector{ContractionStep})
    es = ExecutionState(sg.ttn)
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
