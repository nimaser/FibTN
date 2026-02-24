using GLMakie
using FibErrThresh
using FibErrThresh.FibTNs
using FibErrThresh.SegmentTensorTypes
using FibErrThresh.FibTensorTypes

using FibErrThresh.MiscUtils

# make a 3x3 segment grid
smg = segmentmaskgrid(4, 4; middle=STT_V)
remove_segments!(smg, [(1, 1), (2, 1), (3, 1), (4, 4), (3, 4), (4, 3)])
fixmiddles!(smg)
replace_segment!(smg, (3, 2)..., STT_E)
replace_segment!(smg, (2, 3)..., STT_E)
# force-place boundary on the bottom left corner
add_right_boundaries!(smg)

ftn = FibTN(smg)
fix_excitation!(ftn, (3, 2), 0, 1, 1)
fix_excitation!(ftn, (2, 3), 0, 1, 1; root=true)

# add crossings and contraction between them
e1 = ((2, 2), (2, 3))
e2 = ((1, 3), (2, 3))
add_crossings!(ftn, e1..., 1)
add_crossings!(ftn, e2..., 1)
add_contraction!(ftn, e1, 1, :U, e2, 1, :D; with_vl=true)

# add contractions from crossings to segments
add_contraction!(ftn, e1, 1, :D, (3, 2); with_vl=true)
add_contraction!(ftn, e2, 1, :U, (2, 3); with_vl=true)
display(GLMakie.Screen(), visualize(ftn)[1])

inds, data = naive_contract(ftn)
s, a = get_states_and_amps(ftn.ql, inds, data)
display(GLMakie.Screen(), visualize(ftn, inds, data, s, a; maxstatesperpane=6)[1])
data
