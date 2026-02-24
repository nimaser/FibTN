using GLMakie
using FibErrThresh
using FibErrThresh.FibTNs
using FibErrThresh.SegmentTensorTypes
using FibErrThresh.FibTensorTypes

using FibErrThresh.MiscUtils

# make a 3x3 segment grid
smg = segmentmaskgrid(3, 3; middle=STT_V)
# remove the corners so that we have just three plaquettes
remove_segments!(smg, [(1, 1), (3, 3)])
# force-place boundary on the bottom left corner
add_right_boundaries!(smg)
replace_segment!(smg, 2, 2, STT_E | STT_V)
replace_segment!(smg, 1, 3, STT_E | STT_V)
ftn = FibTN(smg)
edge = ((1, 3), (2, 3))
add_crossings!(ftn, edge..., 1)
add_contraction!(ftn, (2, 2), edge, 1, :D; with_vl=true)
add_contraction!(ftn, (1, 3), edge, 1, :U)
fix_excitation!(ftn, (2, 2), 0, 1, 1)
fix_excitation!(ftn, (1, 3), 0, 1, 1; root=true)
# display(visualize(ftn)[1])
inds, data = naive_contract(ftn)
s, a = get_states_and_amps(ftn.ql, inds, data)
# display(visualize(ftn, inds, data)[1])
# display(visualize(ftn, s, a)[1])
display(GLMakie.Screen(), visualize(ftn, inds, data, s, a; maxstatesperpane=6)[1])
data
