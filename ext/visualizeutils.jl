"""
Adds a `DataInspector` to `f` lazily: defers until the figure has nonzero pixel
dimensions to avoid a GLMakie crash in `pick_sorted`. Also checks immediately in
case the figure is already rendered (e.g. during Revise reload).
"""
function add_inspector_lazily!(f::Figure; range::Int=30)
    inspector_added = Ref(false)
    function try_add(area)
        if !inspector_added[] && area.widths[1] > 0 && area.widths[2] > 0
            DataInspector(f; range=range)
            inspector_added[] = true
        end
    end
    on(try_add, f.scene.viewport)
    try_add(f.scene.viewport[])  # fire immediately in case already rendered
end
