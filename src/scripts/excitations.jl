using GLMakie
using FibErrThresh
using FibErrThresh.FibTNs
using FibErrThresh.SegmentTensorTypes

using FibErrThresh.MiscUtils

# make a 3x3 segment grid
smg = segmentmaskgrid(3, 3; middle=STT_V)
# remove the corners so that we have just three plaquettes
removals = [(1, 1), (3, 3)]
for r in removals remove_segment!(smg, r...) end
# force-place boundary on the bottom left corner
add_right_boundaries!(smg)
# replace_segment!(smg, 2, 2, STT_E)
# replace_segment!(smg, 1, 3, STT_E)
ftn = FibTN(smg)
# edge = ((1, 3), (2, 3))
# add_crossings!(ftn, edge..., 1)
# add_contraction!(ftn, edge, 1, :D, )
display(visualize(ftn)[1])
inds, data = naive_contract(ftn)
s, a = get_states_and_amps(ftn.ql, inds, data)
# display(visualize(ftn, inds, data)[1])
# display(visualize(ftn, s, a)[1])
display(visualize(ftn, inds, data, s, a; maxstatesperpane=6)[1])
# data
