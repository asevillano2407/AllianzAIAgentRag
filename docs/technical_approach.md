# Enfoque técnico, decisiones y trazabilidad

## Resumen ejecutivo

La solución es un RAG textual local sobre el manual CIDE/ASCIDE/CICOS. Separa
ingestión, retrieval, generación, validación y orquestación para medir cada
componente sin atribuir al LLM mejoras que proceden del buscador. No utiliza
servicios de pago: Ollama ejecuta embeddings y LLM; Qdrant persiste los vectores;
un cross-encoder local realiza el reranking; LangGraph controla el flujo.

Configuración seleccionada para la demo:

| Componente | Selección | Justificación |
| --- | --- | --- |
| Ingestión | texto del PDF, chunks de 1.200 caracteres y solape 150 | MVP reproducible y suficiente para reglas textuales |
| Embedding | `qwen3-embedding:0.6b` | mejor Precision, Recall y MRR que `bge-m3` en el dataset local |
| Vector store | Qdrant local, coseno | metadatos y persistencia sin servicio externo |
| Expansión | reglas deterministas de vocabulario | conecta el relato con términos del manual sin otra llamada LLM |
| Candidatos | 12 | diversidad antes de filtrar |
| Reranker | `BAAI/bge-reranker-v2-m3`, híbrido multi-query | mejora el orden sin perder la cobertura de cada expansión |
| Contexto final | 3 chunks | recall de grupos 1,0 en los cinco casos y menos ruido para el LLM |
| LLM | `qwen3:4b` | 100 % de éxito técnico y 80 % de negocio en la prueba final |
| Orquestación | LangGraph | estado explícito, rutas deterministas, reintentos acotados y fallback |

## Arquitectura

```text
PDF
 -> extracción y limpieza
 -> chunks con fuente, página y sección
 -> qwen3-embedding:0.6b
 -> Qdrant local

Consulta
 -> validación y routing determinista
 -> query expansion
 -> búsqueda vectorial + RRF
 -> reranking local multi-query
 -> 3 chunks
 -> qwen3:4b con JSON Schema
 -> validación técnica + guardrails de negocio
 -> respuesta estructurada o fallback
```

Los contratos de dominio no dependen de Ollama, Qdrant ni LangGraph. Los
adaptadores concretos se inyectan en los servicios, lo que permite probar la
lógica con dobles deterministas y sustituir componentes sin reescribir el flujo.

## Separación de evaluaciones

### Retrieval

Se mide antes de ejecutar el LLM:

- `Precision@K`: cuánto contexto recuperado es relevante;
- `Recall@K`: cuánto soporte etiquetado se recupera;
- `MRR`: a qué altura aparece el primer soporte relevante;
- recall de grupos de páginas: cobertura de los conceptos requeridos por cada
  caso de accidente.

El resultado final de retrieval fue recall de grupos `1,0` en los cinco casos.
Esto permite localizar el fallo restante del caso C en generación/razonamiento,
no en ausencia de evidencia.

### Validez técnica de generación

`technical_success_rate` responde a «¿la ejecución produjo una salida utilizable
por el sistema?». Requiere:

1. respuesta JSON compatible con el esquema Pydantic;
2. `query_type` sin alteraciones;
3. decisiones obligatorias en una descripción de accidente;
4. combinación coherente de aplicabilidad y responsabilidad;
5. citas referidas únicamente a chunks entregados al modelo;
6. páginas consistentes con sus chunks;
7. citas literales tras normalizar diferencias tipográficas seguras;
8. ausencia de una contradicción explícita entre la polaridad de la decisión de
   aplicabilidad y su evidencia.

Una ejecución técnicamente válida puede estar equivocada desde negocio. Esta es
la razón por la que `completed` y `business_correct` son métricas diferentes.

### Corrección de negocio

El dataset versiona para cada caso dos etiquetas independientes:

- `expected_applicability`: si CIDE/ASCIDE puede aplicarse;
- `expected_responsibility`: atribución según el convenio, o
  `undetermined`/`not_applicable`.

`business_correct` solo es verdadero cuando ambas coinciden exactamente. No es
un LLM juez ni una comparación textual: es una evaluación determinista contra
etiquetas revisables. Con cinco casos es una señal de regresión útil, no una
estimación estadística de producción.

## Aplicabilidad y responsabilidad

Se separaron porque responden a preguntas distintas. Que el convenio aplique no
implica que el relato permita atribuir responsabilidad. Ejemplo: la alcoholemia
no excluye los convenios, pero tampoco determina por sí sola quién es responsable.

Valores de aplicabilidad:

- `applicable`;
- `not_applicable`;
- `undetermined`.

Valores de responsabilidad:

- `vehicle_a`, `vehicle_b` o `shared`;
- `undetermined` cuando falta una regla o un hecho exigido;
- `not_applicable` cuando el convenio no aplica.

La conclusión mostrada al usuario se genera de forma determinista a partir de
estos campos. Así no puede afirmar lo contrario que la estructura validada.

## Citas, confianza y fail-soft

Cada cita contiene `chunk_id`, página y texto literal. Las citas de aplicabilidad
y responsabilidad están separadas para comprobar qué evidencia sostiene cada
decisión. Una coincidencia literal demuestra trazabilidad, pero no garantiza por
sí sola relevancia semántica; por eso existen además reglas de polaridad y
guardrails de evidencia normativa.

La confianza es cualitativa, no una probabilidad calibrada:

- `high`: el modelo considera que existe evidencia directa y suficiente;
- `medium`: hay soporte, pero persiste alguna limitación;
- `low`: no hay soporte suficiente o un guardrail degradó la decisión.

Las reglas del sistema prevalecen sobre la confianza propuesta por el modelo.
Una decisión definitiva sin cita válida se degrada a `low`.

El modo fail-soft evita perder toda la respuesta por un error recuperable:

| Situación | Acción |
| --- | --- |
| Página incorrecta, pero chunk y cita válidos | sustituir por la página real del chunk |
| Cita inventada o chunk desconocido | eliminar la cita |
| Aplicabilidad definitiva sin cita válida | degradar aplicabilidad y responsabilidad a `undetermined` |
| Responsabilidad definitiva sin cita válida | conservar aplicabilidad y degradar responsabilidad |
| Responsabilidad sin una regla normativa | degradar responsabilidad a `undetermined` |
| Convenio no aplicable con responsabilidad de un vehículo | alinear responsabilidad a `not_applicable` |
| Convenio aplicable con responsabilidad `not_applicable` | degradar responsabilidad a `undetermined` |
| JSON inválido o campos obligatorios ausentes | error técnico y ruta de fallback |

Cada reparación se registra en `generation.adjustments`; no ocurre de forma
silenciosa. En preguntas generales del manual se mantiene la validación estricta.

## Reranking y reducción de contexto

El primer cross-encoder reordenaba todos los candidatos solo respecto al relato
original. Mejoró las diez preguntas generales, pero redujo de `1,0` a `0,9` el
recall medio de grupos en los accidentes al perder la página de identificación
del contrario en C.

La solución fue rerankear respecto a cada consulta original/expandida, preservar
el mejor resultado de cada intención y fusionar los rankings mediante RRF. El
resultado recuperó todos los grupos esperados. El reranker no genera respuestas
ni aplica reglas de responsabilidad: únicamente ordena evidencia candidata.

## Selección del LLM

Las rondas no son todas directamente comparables porque el prompt, el esquema y
los guardrails evolucionaron. Se conservan para mostrar qué fallo motivó cada
cambio:

| Ronda | Modelo | Éxito técnico | Negocio | Latencia media | Lectura |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline inicial | `llama3.2:3b` | 60 % | revisión manual baja | 201 s | alucinaciones y errores de cita |
| Baseline inicial | `qwen3:4b` | 80 % | 1/5 manual | 212 s | mejor equilibrio inicial |
| Baseline inicial | `qwen3:8b` | 60 % | sin mejora global | 479 s | demasiado lento y con timeout |
| Esquema estricto inicial | `qwen3:4b` | 0 % | 0 % | n/d | no rellenaba aún los nuevos campos obligatorios |
| Smoke A | `gemma3:4b` | 100 % | 100 % | 276 s | justificó probar los cinco casos |
| Cinco casos | `gemma3:4b` | 60 % | 20 % | 291 s | más rápido, pero menos fiable |
| Final con prompt v5 y guardrails | `qwen3:4b` | **100 %** | **80 %** | 426 s | modelo seleccionado |

`qwen3:4b` se selecciona por fiabilidad global, no por velocidad. En CPU tarda
aproximadamente siete minutos por caso en la prueba final. Esto es aceptable para
una demo técnica local, pero no para un SLA de producción.

## Riesgos, supuestos y mitigaciones

| Riesgo o supuesto | Impacto | Mitigación o decisión |
| --- | --- | --- |
| Manual de 2004 | reglas potencialmente desactualizadas | declarar alcance y no usar como asesoramiento legal actual |
| Dataset de cinco casos | posible sobreajuste y métricas inestables | separar conjunto de retrieval de diez preguntas y documentar la limitación |
| Validación literal no prueba relevancia | una cita válida puede apoyar otra regla | citas por decisión, polaridad y guardrail normativo |
| CPU local | latencia alta y timeouts | contexto de 3 chunks, timeout configurable y demo con un caso preparado |
| Reranker grande | descarga y memoria adicionales | dependencia opcional y caché local |
| Query expansion por reglas | vocabulario no contemplado | reglas observables, testeadas y ampliables |
| Caso C aún incorrecto | confunde aparcamiento con identificación del contrario | mostrarlo como limitación real y futura regla determinista de aplicabilidad |
| Confidence no calibrada | puede interpretarse como probabilidad | documentarla como categoría cualitativa y forzar `low` al degradar |

## Hitos y estado

1. Requisitos, arquitectura y dominio: cerrados.
2. Ingestión, limpieza y chunking: cerrados.
3. Embeddings, Qdrant, query expansion y reranking: cerrados.
4. Generación, salida estructurada, evaluación y selección del LLM: cerrados.
5. LangGraph, reintentos y fallback: cerrados.
6. Siguiente entrega: interfaz/demo breve, revisión final del README y PPTX.

## Evidencias reproducibles

```powershell
# Retrieval sin ejecutar el LLM
allianz-evaluate-retrieval --top-k 4

# Retrieval de los cinco casos con configuración final
allianz-evaluate-llms --retrieval-only --rerank --candidate-k 12 --top-k 3

# Benchmark final de generación (lento en CPU)
$env:ALLIANZ_OLLAMA_TIMEOUT_SECONDS="600"
allianz-evaluate-llms --model qwen3:4b --rerank --candidate-k 12 --top-k 3

# Tests deterministas, sin Ollama
python -m pytest -q
python -m ruff check .
```
