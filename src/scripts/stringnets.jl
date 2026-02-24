using GLMakie
using FibErrThresh
using FibErrThresh.FibTNs
using FibErrThresh.SegmentTensorTypes
using FibErrThresh.FibTensorTypes

using FibErrThresh.MiscUtils

# make a 3x3 segment grid
smg = segmentmaskgrid(3, 3; middle=STT_V)
# force-place boundary on the bottom left corner
add_right_boundaries!(smg)
ftn = FibTN(smg)
e1 = ((2, 1), (2, 2))
e2 = ((1, 2), (2, 2))
e3 = ((2, 2), (2, 3))
e4 = ((2, 2), (3, 2))
add_crossings!(ftn, e1..., 1)
add_crossings!(ftn, e2..., 1)
add_crossings!(ftn, e3..., 1)
add_crossings!(ftn, e4..., 1)
add_contraction!(ftn, e1, 1, :U, e2, 1, :D; with_vl=true)
add_contraction!(ftn, e2, 1, :U, e3, 1, :U; with_vl=true)
add_contraction!(ftn, e3, 1, :D, e4, 1, :U; with_vl=true)
add_contraction!(ftn, e1, 1, :D, e4, 1, :D; with_vl=true, with_sc=true, sc=1)
# display(visualize(ftn)[1])
inds, data = naive_contract(ftn)
s, a = get_states_and_amps(ftn.ql, inds, data)
display(visualize(ftn, inds, data, s, a; maxstatesperpane=6)[1])
data
