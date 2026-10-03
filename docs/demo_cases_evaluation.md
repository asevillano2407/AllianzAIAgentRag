# Evaluación de los cinco casos de demostración

## Propósito

Este conjunto pequeño comprueba el sistema extremo a extremo y conserva los
fallos que impulsaron las mejoras. No pretende representar toda la casuística
del manual ni estimar precisión de producción.

Los casos están versionados en
`evaluation/datasets/demo_accident_cases.jsonl`. Cada uno define:

- relato de entrada;
- grupos de páginas que retrieval debe cubrir;
- aplicabilidad esperada;
- responsabilidad esperada según el convenio.

## Configuración final

- Embedding: `qwen3-embedding:0.6b`.
- Vector store: Qdrant local.
- Expansión determinista y fusión RRF.
- Reranker: `BAAI/bge-reranker-v2-m3` multi-query.
- Candidatos: 12; contexto final: 3 chunks.
- LLM: `qwen3:4b` mediante Ollama.
- Prompt: `analysis-v5`; temperatura 0 y semilla 42.
- Timeout de evaluación: 600 segundos por la ejecución en CPU.

## Cómo leer las métricas

- `retrieval_page_group_recall=1.0`: llegó al contexto al menos una página de
  cada concepto requerido. No significa que el LLM usara bien esa evidencia.
- `status=completed`: la salida superó el esquema y las validaciones/ajustes
  permitidos. No significa que la decisión sea correcta.
- `applicability_correct`: coincide la decisión sobre uso del convenio.
- `responsibility_correct`: coincide la atribución convencional esperada.
- `business_correct`: ambas anteriores son verdaderas.
- `adjustments`: reparaciones o renderizados deterministas aplicados después de
  la generación.

## Resultado final de Qwen 3 4B

| Caso | Páginas | Retrieval | Técnico | Aplicabilidad | Responsabilidad | Negocio | Latencia |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A. Alcance trasero | 75, 87, 67 | 1,0 | sí | sí | sí | sí | 487,6 s |
| B. Cinco vehículos | 56, 18, 58 | 1,0 | sí | sí | sí | sí | 355,5 s |
| C. Aparcado y contrario desconocido | 73, 34, 73 | 1,0 | sí | no | no | no | 375,9 s |
| D. Cambio de carril | 41, 75, 72 | 1,0 | sí | sí | sí | sí | 554,3 s |
| E. Alcoholemia y lesiones | 9, 5, 57 | 1,0 | sí | sí | sí | sí | 357,0 s |

Resumen:

- éxito técnico: `5/5 = 100 %`;
- corrección de negocio: `4/5 = 80 %`;
- recall de grupos de páginas: `5/5 = 100 %`;
- latencia media de generación: `426,0 s`.

La duración registrada es la inferencia de Ollama, no todo el retrieval. La CPU
es el principal cuello de botella observado.

## Análisis por caso

### A. Alcance trasero

La página 75 contiene la regla correcta. Qwen separa aplicabilidad y
responsabilidad y atribuye correctamente el caso a B. La alegación de frenada
brusca no altera la atribución convencional. La conclusión se renderiza desde
los campos estructurados para impedir contradicciones.

### B. Intervención de cinco vehículos

La página 56 indica que el CIDE no aplica cuando intervienen más de dos
vehículos identificados. Qwen devuelve `not_applicable`. Como inicialmente
combinó esa decisión con otra responsabilidad, el fail-soft la alineó a
`not_applicable`; el ajuste quedó registrado como
`responsibility_aligned_with_non_applicable_convention`.

### C. Vehículo aparcado y contrario desconocido

Retrieval recupera los dos conceptos etiquetados: página 73 para vehículo
aparcado y página 34 para identificación del contrario. Qwen usa la primera
regla y decide que B es responsable, pero omite que un «SUV rojo» sin matrícula
no identifica suficientemente al contrario para aplicar el convenio.

Este es el único fallo de negocio final. No se debe al retriever, sino a la
composición de dos reglas. Es candidato a una regla determinista futura o a un
dataset de evaluación más amplio, pero no se añade otro parche al MVP.

### D. Cambio de carril

La página 75 establece que quien cambia de carril resulta culpable según el
convenio. Qwen identifica a A y resuelve correctamente. En rondas anteriores
este caso fallaba por citas no literales; la normalización de formato y el
fail-soft evitan perder toda la respuesta por diferencias recuperables.

### E. Alcoholemia y lesiones

La página 9 indica que la alcoholemia no excluye los convenios. El modelo decide
correctamente `applicable` y mantiene la responsabilidad `undetermined`, porque
la alcoholemia y la detención no bastan para asignarla. Este caso valida la
separación entre aplicabilidad, responsabilidad convencional y consecuencias
legales o penales.

## Comparativa de modelos

### Baseline anterior a los guardrails finales

| Modelo | Éxito técnico | Latencia media | Hallazgo principal |
| --- | ---: | ---: | --- |
| `llama3.2:3b` | 60 % | 201 s | alucinó hechos, falló citas y contradijo reglas |
| `qwen3:4b` | 80 % | 212 s | mejor equilibrio; único acierto completo manual en B |
| `qwen3:8b` | 60 % | 479 s | timeout y coste doble sin mejora suficiente |

Esta ronda todavía no tenía las etiquetas automáticas de negocio actuales. La
valoración se realizó manualmente y motivó la separación entre aplicabilidad y
responsabilidad.

### Gemma 3 4B con esquema final

| Métrica | Resultado |
| --- | ---: |
| Casos completados | 3/5 |
| Éxito técnico | 60 % |
| Corrección de negocio | 20 % |
| Latencia media de completados | 291,4 s |

Gemma resolvió A, devolvió decisiones indeterminadas incorrectas en B y C y
falló D/E por errores de cita. Era más rápido, pero no suficientemente fiable.

### Qwen 3 4B final

| Métrica | Resultado |
| --- | ---: |
| Casos completados | 5/5 |
| Éxito técnico | 100 % |
| Corrección de negocio | 80 % |
| Latencia media | 426,0 s |

Se selecciona Qwen porque completa todos los casos y cuadruplica el número de
casos correctos de negocio frente a Gemma, aun con mayor latencia.

## Qué se adquirió y qué se descartó

Se conserva:

- `qwen3-embedding:0.6b`;
- query expansion determinista;
- reranking híbrido multi-query;
- `candidate_k=12` y `top_k=3` para generación;
- `qwen3:4b`;
- esquema estricto por tipo de consulta;
- citas específicas por decisión;
- evaluación técnica y de negocio separadas;
- fail-soft observable y conclusión determinista.

Se descarta para el MVP:

- `llama3.2:3b` y `qwen3:8b` como modelos finales;
- `gemma3:4b` pese a su menor latencia;
- reranking solo contra el relato original;
- incrementar K indiscriminadamente;
- usar un LLM juez o servicios externos de evaluación;
- seguir afinando retrieval después de alcanzar cobertura completa en el set.

## Reproducción

La ejecución completa es lenta en CPU:

```powershell
$env:ALLIANZ_OLLAMA_TIMEOUT_SECONDS="600"

allianz-evaluate-llms `
  --model qwen3:4b `
  --rerank `
  --candidate-k 12 `
  --top-k 3 `
  --output .tmp/qwen3_final_five_cases.jsonl
```

Para validar solo retrieval, sin esperar al LLM:

```powershell
allianz-evaluate-llms `
  --retrieval-only `
  --rerank `
  --candidate-k 12 `
  --top-k 3
```
