# this script must define rsg, contractionsequences, pindict, offset, scale, and
# nlabeloffsetscale before calculateGS.jl is run
len = 2
rsg, pindict = hex_lattice(len, 1)

contractionsequences = [[2, 3, 4], [1, 6, 5], [2, 7, 8], [1, 10, 9], [1, 2]]# [2, 11, 12], [1, 14, 13], [2, 15, 16], [1, 18, 17], [2, 19, 20], [1, 22, 21]]

offset = (0, -1)
scale = 2
nlabeloffsetscale = 0.3
