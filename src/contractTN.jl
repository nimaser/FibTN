# before this script is run, rsg, contractionsequences, pindict, offset, scale, and
# nlabeloffsetscale should've been added to the environment
#
# before this script is run,  genTN.jl should be run to add ig, tg, qg, and the NetworkLayout
# to the env

# contract the tg
contractcaps!(tg)
for cs in contractionsequences
    contractsequence!(tg, cs)
end
T = contractionresult(tg)
s = tensor2states(T)

# display the result of contracting the tg
f = Figure()
w, h, axs = getaxisgrid(f, length(s))
plots = statesplot(axs, qg, s, vlabels=false, layout=l, popoutargs=Dict(:titlesize=>28))
finalize(f, axs)
display(f)

# store results?
