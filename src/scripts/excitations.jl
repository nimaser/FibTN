using GLMakie
using FibErrThresh
using FibErrThresh.FibTNs
using FibErrThresh.SegmentTensorTypes

using FibErrThresh.MiscUtils

smg = segmentmaskgrid(3, 3; middle=STT_T)
remove_segment!(smg, 1, 1)
remove_segment!(smg, 3, 3)
replace_segment!(smg, 3, 1, STT_M)
replace_segment!(smg, 2, 2, STT_E)
ftn = FibTN(smg)
add_crossings!(ftn, (1, 3), (2, 3), 1)
display(visualize(ftn)[1])
