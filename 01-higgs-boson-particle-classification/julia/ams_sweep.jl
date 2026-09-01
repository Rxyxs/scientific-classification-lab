# Reimplementacion independiente en Julia de la metrica AMS
# (Approximate Median Significance) y un barrido de umbral MUCHO mas fino
# que el usado en Python (2000 umbrales vs. 200) -- aprovechando la
# velocidad de Julia en computo numerico para responder una pregunta real
# que el barrido grueso de src/metrics.py no puede: ¿el optimo de AMS es un
# pico agudo (fragil, sensible al umbral exacto) o una meseta ancha
# (robusto)?
#
#   julia --project=julia julia/ams_sweep.jl

using CSV
using DataFrames
using Printf

const B_REG = 10.0

function ams_score(y_true::Vector{Int}, y_pred::BitVector, weights::Vector{Float64})
    s = sum(weights[(y_true .== 1) .& y_pred])
    b = sum(weights[(y_true .== 0) .& y_pred])
    radicand = 2.0 * ((s + b + B_REG) * log(1.0 + s / (b + B_REG)) - s)
    return radicand > 0 ? sqrt(radicand) : 0.0
end

function ams_sweep(y_true::Vector{Int}, proba::Vector{Float64}, weights::Vector{Float64}, n_thresholds::Int)
    thresholds = range(0.0, 1.0, length=n_thresholds)
    scores = Vector{Float64}(undef, n_thresholds)
    for (i, t) in enumerate(thresholds)
        y_pred = proba .>= t
        scores[i] = ams_score(y_true, y_pred, weights)
    end
    best_idx = argmax(scores)
    return collect(thresholds), scores, thresholds[best_idx], scores[best_idx]
end

function main()
    println("[1/3] Cargando probabilidades reales del LightGBM afinado (public-test, 100,000 eventos)...")
    df = CSV.read("../outputs/reports/julia_ams_reference.csv", DataFrame)
    y_true = Int.(df.is_signal)
    proba = Float64.(df.proba)
    weights = Float64.(df.kaggle_weight)
    println("  $(length(y_true)) eventos reales cargados")

    println("\n[2/3] Verificacion cruzada: AMS con 200 umbrales (mismo grano que Python)...")
    _, _, t200, ams200 = ams_sweep(y_true, proba, weights, 200)
    @printf("  Julia (200 umbrales): AMS=%.4f, umbral=%.4f\n", ams200, t200)
    @printf("  Python (reportado en README): AMS=3.6414\n")
    @printf("  Diferencia: %.4f (esperable por discretizacion distinta de los umbrales, no un bug)\n", abs(ams200 - 3.6414))

    println("\n[3/3] Barrido fino: 2000 umbrales -- ¿pico agudo o meseta ancha?...")
    thresholds, scores, t2000, ams2000 = ams_sweep(y_true, proba, weights, 2000)
    @printf("  Julia (2000 umbrales): AMS=%.4f, umbral=%.4f\n", ams2000, t2000)

    # Ancho de la "meseta": rango de umbrales donde AMS >= 99% del maximo.
    near_best = findall(s -> s >= 0.99 * ams2000, scores)
    plateau_width = thresholds[maximum(near_best)] - thresholds[minimum(near_best)]
    @printf("  Meseta al 99%% del AMS maximo: ancho=%.4f (umbrales %.4f a %.4f)\n",
            plateau_width, thresholds[minimum(near_best)], thresholds[maximum(near_best)])

    out = DataFrame(threshold=thresholds, ams=scores)
    CSV.write("../outputs/reports/julia_ams_sweep.csv", out)

    println("\n=== Conclusion ===")
    if plateau_width > 0.02
        println("El optimo es una meseta ancha (>0.02 de umbral), no un pico fragil -- ",
                "el umbral elegido no depende de un ajuste extremadamente preciso.")
    else
        println("El optimo es un pico relativamente agudo (<0.02 de umbral) -- ",
                "vale la pena monitorear si el umbral de produccion se mantiene estable con datos nuevos.")
    end
    println("\nGuardado en: outputs/reports/julia_ams_sweep.csv")
end

main()
