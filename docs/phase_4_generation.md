# Fase 4: generación local, validación y selección del LLM

## Objetivo

Transformar una consulta y los chunks recuperados en una respuesta estructurada,
trazable y útil, sin aceptar ciegamente lo que produzca el LLM. Retrieval y
generación permanecen separados: esta fase consume evidencia ya recuperada y no
decide cómo se indexa el manual.

## Componentes

1. `StructuredLlmProvider` desacopla el servicio del runtime concreto.
2. `OllamaStructuredLlm` ejecuta el modelo local y registra tokens y duración.
3. `prompts.py` versiona las instrucciones; la evaluación final usa
   `analysis-v5`.
4. `AnswerGenerator` selecciona el esquema, valida, repara errores seguros y
   aplica los guardrails.
5. `evaluation/cases.py` compara las decisiones con etiquetas de negocio.
6. `evaluation/llm_benchmark.py` congela retrieval y compara modelos sin mezclar
   sus respuestas.

Ollama recibe el JSON Schema generado por Pydantic. Se usa temperatura cero y
semilla fija para reducir variabilidad, aunque la reproducibilidad exacta puede
depender del runtime y del hardware.

## Dos contratos de salida

Las preguntas sobre el manual utilizan `AnalysisResponse`. Las descripciones de
accidente utilizan `AccidentAnalysisResponse`, que obliga a informar:

```json
{
  "query_type": "accident_description",
  "conclusion": "...",
  "convention_applicability": "applicable",
  "convention_responsibility": "vehicle_b",
  "applicability_citations": [],
  "responsibility_citations": [],
  "facts": [],
  "missing_information": [],
  "confidence": "high",
  "citations": []
}
```

La separación entre aplicabilidad y responsabilidad es una decisión central:

- aplicabilidad responde si CIDE/ASCIDE puede usarse en el supuesto;
- responsabilidad responde quién resulta responsable según el convenio;
- no se infiere responsabilidad legal, penal, cobertura ni indemnización.

La alcoholemia ilustra la diferencia: no excluye la aplicación de los convenios,
pero se ignora al atribuir responsabilidad convencional.

## Validación técnica

Pydantic comprueba tipos, campos obligatorios y ausencia de campos inesperados.
Después se verifica que:

- el LLM no cambie el `query_type` determinado por la aplicación;
- aplicabilidad y responsabilidad formen una combinación coherente;
- una decisión definitiva tenga citas en su lista específica;
- cada `chunk_id` pertenezca al contexto enviado al modelo;
- la página coincida con el chunk;
- la cita sea literal tras normalizar acentos, puntuación, comillas y guiones de
  salto de línea;
- una cita que omite una negación previa no se acepte;
- la polaridad explícita de la evidencia no contradiga la aplicabilidad.

Los chunks se consideran entrada no confiable. El prompt prohíbe seguir
instrucciones encontradas dentro del contexto recuperado.

## Guardrails de negocio

El esquema correcto no garantiza una decisión correcta. Se añadieron reglas
deterministas para los fallos observados:

- una responsabilidad definitiva necesita una regla normativa, no solo una
  descripción del caso;
- una frenada alegada no cambia por sí sola la regla de alcance trasero;
- alcoholemia, drogas o detención no atribuyen responsabilidad convencional;
- la conclusión se renderiza desde los campos estructurados y no puede
  contradecirlos;
- si el convenio no aplica, la responsabilidad se alinea a `not_applicable`;
- si el convenio aplica, una responsabilidad `not_applicable` se degrada a
  `undetermined`.

La comprobación de contradicciones no es un sistema NLI general. Detecta
polaridades explícitas conocidas y evita la contradicción entre conclusión y
estructura. La relevancia semántica completa de una cita sigue siendo una
limitación documentada.

## Fail-soft: responder sin inventar

La primera versión descartaba toda la respuesta ante cualquier error de cita.
Eso era seguro, pero demasiado estricto para un asistente útil. La política
final distingue errores recuperables de salidas estructuralmente inválidas:

| Error | Resultado |
| --- | --- |
| Página errónea con chunk y texto válidos | página corregida desde metadatos |
| Cita no literal o chunk desconocido | cita eliminada |
| Aplicabilidad sin ninguna cita válida | aplicabilidad y responsabilidad `undetermined`, confianza baja |
| Responsabilidad sin cita válida o normativa | solo responsabilidad `undetermined`, confianza baja |
| Incoherencia reparable entre decisiones | valores alineados de forma conservadora |
| JSON inválido o campo obligatorio ausente | error técnico; LangGraph usa fallback |

Las reparaciones se devuelven en `generation.adjustments`, de forma que son
observables y evaluables. Fail-soft no significa relajar la verdad: nunca crea
una cita ni convierte falta de evidencia en una decisión positiva.

## Confianza

`low`, `medium` y `high` son categorías cualitativas propuestas por el modelo,
no probabilidades calibradas. La aplicación puede reducirlas:

- una degradación por falta de cita fuerza `low`;
- una responsabilidad sin evidencia normativa fuerza `low`;
- una respuesta de confianza media o alta debe conservar alguna cita válida.

Para producción sería necesario calibrar confianza con un conjunto mayor. En
esta prueba se usa como señal de suficiencia y transparencia.

## Evaluación técnica y evaluación de negocio

`technical_success_rate` mide si se obtuvo una respuesta estructurada que supera
las validaciones o reparaciones permitidas. No mide si la decisión es correcta.

La evaluación de negocio compara exactamente:

- `convention_applicability` con `expected_applicability`;
- `convention_responsibility` con `expected_responsibility`.

`business_correct` exige que ambas sean correctas. Esta separación hizo visible
que un modelo podía alcanzar `completed` y, aun así, equivocarse en el caso.

## Evolución de las pruebas

Las rondas se realizaron sobre CPU local. No todas son comparables uno a uno,
porque el prompt, el esquema y los guardrails se fueron corrigiendo a partir de
los fallos observados.

| Ronda | Modelo | Éxito técnico | Corrección de negocio | Latencia media | Decisión |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline | `llama3.2:3b` | 60 % | baja en revisión manual | 201 s | descartado por alucinaciones y citas |
| Baseline | `qwen3:4b` | 80 % | 1/5 manual | 212 s | mejor candidato inicial |
| Baseline | `qwen3:8b` | 60 % | sin mejora global | 479 s | descartado por coste y timeout |
| Primer esquema estricto | `qwen3:4b` | 0 % | 0 % | n/d | reveló campos obligatorios no guiados |
| Smoke caso A | `gemma3:4b` | 100 % | 100 % | 276 s | se amplió a cinco casos |
| Cinco casos | `gemma3:4b` | 60 % | 20 % | 291 s | descartado por baja fiabilidad |
| Final, prompt v5 + guardrails | `qwen3:4b` | **100 %** | **80 %** | 426 s | seleccionado |

El primer esquema estricto fue una prueba de transición: el modelo devolvía
`null` en los nuevos campos y un caso agotó el timeout. Sirvió para hacer el
esquema de accidente explícito en el prompt y no debe interpretarse como la
capacidad final de Qwen.

## Resultado final por caso

Configuración: `qwen3:4b`, reranking híbrido, 12 candidatos y 3 chunks finales.

| Caso | Aplicabilidad | Responsabilidad | Negocio | Latencia LLM |
| --- | --- | --- | ---: | ---: |
| A. Alcance trasero | correcta | B, correcta | sí | 487,6 s |
| B. Cinco vehículos | no aplicable, correcta | no aplicable, correcta | sí | 355,5 s |
| C. Aparcado y contrario desconocido | aplicable, incorrecta | B, incorrecta | no | 375,9 s |
| D. Cambio de carril | aplicable, correcta | A, correcta | sí | 554,3 s |
| E. Alcoholemia | aplicable, correcta | indeterminada, correcta | sí | 357,0 s |

El caso C falla pese a que retrieval recupera las páginas 73 y 34. El modelo
prioriza la regla que culpa al vehículo que colisiona con el aparcado, pero no
aplica correctamente el requisito previo de identificación del contrario. Es un
fallo de razonamiento de negocio y la principal limitación conocida.

## Selección final

`qwen3:4b` queda como modelo predeterminado. Gemma fue aproximadamente un 32 %
más rápido en la ronda de cinco casos, pero solo completó tres y acertó uno.
Qwen completó los cinco y acertó cuatro. Qwen 8B duplicó aproximadamente la
latencia del baseline sin una mejora global, y Llama 3.2 3B mostró menor calidad.

La selección prioriza fiabilidad sobre velocidad. Una media de 426 segundos en
CPU no es un rendimiento de producción; para la demo se utilizará un único caso
representativo y se explicará que una GPU o un servidor local optimizado reduce
la latencia sin cambiar la arquitectura.

## Reproducción

```powershell
$env:ALLIANZ_OLLAMA_TIMEOUT_SECONDS="600"

allianz-evaluate-llms `
  --model qwen3:4b `
  --rerank `
  --candidate-k 12 `
  --top-k 3 `
  --output .tmp/qwen3_final_five_cases.jsonl
```

Los tests unitarios usan proveedores falsos y no ejecutan benchmarks ni
descargan modelos.
