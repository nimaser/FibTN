module TensorTypes

export AbstractTensorType, index_labels
export Elbow, Reflector, LoopAmplitude, Vertex, Tail, Crossing, Fusion, End, Excitation, DoubledFusion

using ..IndexLabels

abstract type AbstractTensorType end

struct Reflector        <: AbstractTensorType end
struct LoopAmplitude    <: AbstractTensorType end
struct Vertex           <: AbstractTensorType end
struct Tail             <: AbstractTensorType end
struct Crossing         <: AbstractTensorType end
struct Fusion           <: AbstractTensorType end
struct End              <: AbstractTensorType end
struct Excitation       <: AbstractTensorType end
struct DoubledFusion    <: AbstractTensorType end

index_labels(::Type{Reflector}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
]

index_labels(::Type{LoopAmplitude}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
]

index_labels(::Type{Vertex}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
    IndexLabel(:c, VIRT),
    IndexLabel(:q, PHYS),
]

index_labels(::Type{Tail}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
    IndexLabel(:q, PHYS),
]

index_labels(::Type{Crossing}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
    IndexLabel(:c, VIRT),
    IndexLabel(:d, VIRT),
]

index_labels(::Type{Fusion}) = [
    IndexLabel(:a, VIRT),
    IndexLabel(:b, VIRT),
    IndexLabel(:c, VIRT),
]

index_labels(::Type{End}) = [
#TODO
]

index_labels(::Type{Excitation}) = [
#TODO
]

index_labels(::Type{DoubledFusion}) = [
#TODO
]

end # module TensorTypes
