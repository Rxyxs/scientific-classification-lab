[ 🇺🇸 Read in English ](README.md) | [ 🇨🇱 Español ]

# Scientific Classification Lab

Dos problemas de clasificación en ciencias físicas, mismo enfoque central: extraer features específicas del dominio desde datos científicos crudos, y comparar un modelo con gradient boosting contra una red neuronal. Cada carpeta es autocontenida, con su propio README, dependencias y tests. Este repo reemplaza dos repos separados de un solo dominio que antes vivían en este perfil.

## Técnicas

| # | Dominio | Carpeta | Qué hace |
|---|---|---|---|
| 01 | Física de partículas (ATLAS/CERN) | [`01-higgs-boson-particle-classification`](01-higgs-boson-particle-classification) | Clasifica eventos de decaimiento Higgs-a-tau-tau vs. background a partir de variables cinemáticas: modelo con gradient boosting vs. red neuronal en PyTorch, evaluado con la métrica AMS, servido vía FastAPI. |
| 02 | Detección de exoplanetas (Kepler) | [`02-exoplanet-transit-classification`](02-exoplanet-transit-classification) | Clasifica Objetos de Interés Kepler como exoplanetas confirmados vs. falsos positivos a partir de features de la señal de tránsito. |

## Por qué un repo en vez de dos

Ambos proyectos son reales, ejecutables y probados de forma independiente — esto no es esconder alcance, es representarlo con precisión. Dos repos en dos dominios que suenan sin relación (física de partículas, astronomía) esconden el hecho de que comparten la misma técnica central — ingeniería de features desde mediciones científicas crudas hacia un clasificador supervisado, GBM vs. red neuronal comparados cara a cara; un laboratorio hace de ese método compartido el punto real.

## Cómo correr una técnica

Cada carpeta es autocontenida — ver su propio README para el setup exacto y el entry point, resultados reales de una corrida real, y cualquier hallazgo negativo honesto.

## Autor

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Código: MIT — ver [LICENSE](LICENSE)
