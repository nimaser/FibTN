using FibErrThresh.SegmentTensorTypes
using FibErrThresh.FibTensorTypes

@testset "SegmentTensorTypes mask basics" begin
    # mask constants exist
    @test hasU(STT_U)
    @test hasR(STT_R)
    @test hasD(STT_D)
    @test hasL(STT_L)
    @test hasM(STT_M)
    @test hasT(STT_T)
    @test hasE(STT_E)
    @test hasV(STT_V)
    @test hasB(STT_B)

    # physical index checks
    @test  hasTP(STT_U)
    @test  hasTP(STT_R)
    @test !hasTP(STT_D | STT_L)
    @test  hasMP(STT_T)
    @test  hasMP(STT_E)
    @test !hasMP(STT_V | STT_M)
    @test  hasBP(STT_D)
    @test  hasBP(STT_L)
    @test !hasBP(STT_U | STT_R)

    # hasX on SegmentTensorType{Mask} works as expected
    T = SegmentTensorType{STT_U | STT_R | STT_D | STT_L | STT_M | STT_T | STT_V | STT_B}
    @test hasU(T) && hasR(T) && hasD(T) && hasL(T) && hasM(T) && hasT(T) && hasV(T) && hasB(T)
    T = SegmentTensorType{STT_U | STT_D | STT_L | STT_M | STT_E}
    @test hasU(T) && hasD(T) && hasL(T) && hasM(T) && hasE(T)

    # getmask round-trip
    @test getmask(SegmentTensorType{STT_U | STT_R | STT_L | STT_M}) == STT_U | STT_R | STT_L | STT_M
end

@testset "SegmentTensorTypes infermask" begin
    # T/E/V each imply M
    @test infermask(STT_T) == STT_T | STT_M
    @test infermask(STT_E) == STT_E | STT_M
    @test infermask(STT_V) == STT_V | STT_M
    @test infermask(STT_T | STT_V) == STT_T | STT_V | STT_M
    # B implies R
    @test infermask(STT_B) == STT_B | STT_R
    # already-correct masks are unchanged
    @test infermask(STT_U | STT_R) == STT_U | STT_R
    @test infermask(STT_T | STT_M) == STT_T | STT_M
    # zero mask stays zero
    @test infermask(0) == 0
end

@testset "SegmentTensorTypes validatemask" begin
    # valid masks pass through unchanged
    @test validatemask(SegmentTensorType{STT_U | STT_R}) === SegmentTensorType{STT_U | STT_R}
    @test validatemask(SegmentTensorType{STT_U | STT_R | STT_M | STT_T | STT_D | STT_L}) ===
          SegmentTensorType{STT_U | STT_R | STT_M | STT_T | STT_D | STT_L}

    # M without any top ports → error
    @test_throws ArgumentError validatemask(SegmentTensorType{STT_M | STT_D | STT_L})
    # M without any bottom ports → error
    @test_throws ArgumentError validatemask(SegmentTensorType{STT_M | STT_U | STT_R})
    # single top port without M → error
    @test_throws ArgumentError validatemask(SegmentTensorType{STT_U})
    @test_throws ArgumentError validatemask(SegmentTensorType{STT_R})
    # single bottom port without M → error
    @test_throws ArgumentError validatemask(SegmentTensorType{STT_D})
    @test_throws ArgumentError validatemask(SegmentTensorType{STT_L})
    # T and E together → error
    @test_throws ArgumentError validatemask(SegmentTensorType{STT_U | STT_R | STT_M | STT_T | STT_E | STT_D | STT_L})
    # T without M → error
    @test_throws ArgumentError validatemask(SegmentTensorType{STT_U | STT_R | STT_T | STT_D | STT_L})
    # B without R → error
    @test_throws ArgumentError validatemask(SegmentTensorType{STT_B | STT_U})
end

@testset "SegmentTensorTypes basics" begin
    # Enumerate every combination of the 9 bits, keep only valid masks via validatemask.
    # experimentally speaking, there are 95 of them
    all_masks = filter(0x000:0x1FF) do mask
        try
            validatemask(SegmentTensorType{mask})
            true
        catch
            false
        end
    end

    @test length(all_masks) > 0  # sanity: at least some masks are valid
    for mask in all_masks
        T = SegmentTensorType{mask}
        ports = tensor_ports(T)
        td1 = tensor_data(T)
        td2 = tensor_data(T)
        @test length(ports) == ndims(td1)  # ports/data consistency
        @test td1 === td2                  # caching: same object returned
    end
end

@testset "SegmentTensorTypes data" begin
    # STT_U|STT_R (no middle): REFLECTOR * ELBOW_T3 * REFLECTOR, ports (U, R, TP)
    @tensor ref[U, R, TP] :=
        tensor_data(REFLECTOR)[U, a] * tensor_data(ELBOW_T3)[a, b, TP] *
        tensor_data(REFLECTOR)[b, R]
    @test tensor_data(SegmentTensorType{STT_U | STT_R}) ≈ ref

    # STT_U|STT_R|STT_B: same but with BOUNDARY+REFLECTOR extending the right port
    @tensor ref[U, R, TP] :=
        tensor_data(REFLECTOR)[U, a] * tensor_data(ELBOW_T3)[a, b, TP] *
        tensor_data(REFLECTOR)[b, c] * tensor_data(BOUNDARY)[c, d] *
        tensor_data(REFLECTOR)[d, R]
    @test tensor_data(SegmentTensorType{STT_U | STT_R | STT_B}) ≈ ref

    # STT_U|STT_R|STT_M|STT_L: VERTEX top, ELBOW_T1 bottom; ports (U, R, L, TP, BP)
    @tensor ref[U, R, L, TP, BP] :=
        tensor_data(REFLECTOR)[U, a] * tensor_data(VERTEX)[a, b, c, TP] *
        tensor_data(REFLECTOR)[b, R] *
        tensor_data(REFLECTOR)[c, d] * tensor_data(ELBOW_T1)[L, d, BP]
    @test tensor_data(SegmentTensorType{STT_U | STT_R | STT_M | STT_L}) ≈ ref

    # STT_U|STT_M|STT_D|STT_L: ELBOW_T2 top, VERTEX bottom; ports (U, D, L, TP, BP)
    @tensor ref[U, D, L, TP, BP] :=
        tensor_data(REFLECTOR)[U, a] * tensor_data(ELBOW_T2)[a, c, TP] *
        tensor_data(REFLECTOR)[c, d] * tensor_data(VERTEX)[D, L, d, BP]
    @test tensor_data(SegmentTensorType{STT_U | STT_M | STT_D | STT_L}) ≈ ref

    # STT_U|STT_M|STT_L: ELBOW_T2 top, ELBOW_T1 bottom; ports (U, L, TP, BP)
    @tensor ref[U, L, TP, BP] :=
        tensor_data(REFLECTOR)[U, a] * tensor_data(ELBOW_T2)[a, c, TP] *
        tensor_data(REFLECTOR)[c, d] * tensor_data(ELBOW_T1)[d, L, BP]
    @test tensor_data(SegmentTensorType{STT_U | STT_M | STT_L}) ≈ ref
end
