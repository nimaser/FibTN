using GLMakie
using FibErrThresh
using FibErrThresh.FibTNs
using FibErrThresh.SegmentTensorTypes
using FibErrThresh.FibTensorTypes

using FibErrThresh.MiscUtils

# make a 3x3 segment grid
smg = segmentmaskgrid(4, 4; middle=STT_V)
remove_segments!(smg, [(1, 1), (1, 2), (2, 1)])
fixmiddles!(smg)
# force-place boundary on the bottom left corner
add_right_boundaries!(smg)
ftn = FibTN(smg)
inds, data = naive_contract(ftn)
s, a = get_states_and_amps(ftn.ql, inds, data)
display(GLMakie.Screen(), visualize(ftn, inds, data, s, a; maxstatesperpane=6)[1])
data
