# Fase 4: LLM local y salida estructurada

## Objetivo

Transformar una consulta y los chunks recuperados en una respuesta estructurada
sin permitir que el LLM invente fuentes. Esta fase implementa generación y
validación; el router, los reintentos y LangGraph pertenecen a la fase siguiente.

## Separación de responsabilidades

1. `StructuredLlmProvider` define una interfaz independiente del runtime.
2. `OllamaStructuredLlm` llama a `/api/chat` y conserva métricas de inferencia.
3. `prompts.py` mantiene el prompt versionado fuera de la lógica de negocio.
4. `AnswerGenerator` valida el JSON y comprueba cada cita contra la evidencia.

Ollama recibe el JSON Schema generado por `AnalysisResponse` mediante Pydantic.
Se utiliza generación no streaming, temperatura cero y una semilla fija para
facilitar comparaciones reproducibles entre modelos.

## Defensa en profundidad

Forzar un JSON Schema garantiza la forma de la salida, pero no garantiza que su
contenido sea cierto. Después de la generación se comprueba que:

- `query_type` no haya sido modificado por el modelo;
- una respuesta con confianza media o alta incluya citas;
- cada `chunk_id` citado estuviera en el contexto enviado al LLM;
- la página citada coincida con la del chunk;
- `quote` sea un fragmento literal del texto recuperado.

Los chunks se tratan como entrada no confiable. El prompt indica expresamente
que cualquier instrucción encontrada dentro de `retrieved_context` debe
ignorarse, reduciendo el riesgo de prompt injection procedente de documentos.

## Salida de dominio

La respuesta validada utiliza el contrato ya creado:

```json
{
  "query_type": "manual_question",
  "conclusion": "...",
  "facts": ["..."],
  "missing_information": [],
  "confidence": "high",
  "citations": [
    {
      "chunk_id": "...",
      "page": 14,
      "quote": "..."
    }
  ]
}
```

Junto a ella se conservan el modelo, tokens de entrada y salida y duración. Las
métricas permiten comparar al menos dos LLM manteniendo constantes el prompt,
el esquema, la consulta y los chunks recuperados.

## Recorrido paso a paso

1. La CLI valida la consulta y el tipo de entrada.
2. `SemanticRetriever` crea el embedding y solicita los tres chunks más próximos.
3. `build_user_prompt` serializa consulta, esquema y evidencia como JSON.
4. Ollama genera localmente una respuesta que debe respetar el JSON Schema.
5. Pydantic valida tipos y campos; después `AnswerGenerator` valida las citas.
6. Solo una respuesta que supera ambas capas se devuelve junto con tokens y latencia.

El comando manual une retrieval y generación sin anticipar todavía el router:

```powershell
allianz-generate "¿Cuándo caduca un siniestro en CICOS?" `
  --query-type manual_question
```

`manual_question` y `accident_description` son valores de dominio explícitos. El
router automático se añadirá en la fase agentic; introducirlo aquí mezclaría dos
responsabilidades y haría más difícil probar la generación de forma aislada.

## Comparación local de LLM

Se mantuvieron constantes consulta, embedding, prompt, esquema, parámetros y
fragmentos recuperados. La pregunta fue: «¿Cuál es el plazo de caducidad de una
reclamación CICOS?». El contexto incluía como primer resultado el chunk de la
página 14 con la regla de caducidad.

| Modelo | Contexto | Resultado | Validez y calidad | Latencia observada |
| --- | ---: | --- | --- | ---: |
| `qwen3:4b` | top 3 | timeout | Sin salida validable | > 300 s |
| `llama3.2:3b` | top 3 | correcto | Esquema, página y cita literal válidos | 238,1 s |

Llama concluyó correctamente que el plazo es un año desde el accidente, citó
la página 14 y produjo 156 tokens de salida a partir de 1.574 tokens de prompt.
Es una medición controlada, no un benchmark estadístico. Sirve para descartar
una configuración que no cumple el límite operativo en este equipo.

## Decisión

`llama3.2:3b` queda como LLM predeterminado: es local, gratuito, soporta español,
ocupa aproximadamente 2 GB y completó el flujo dentro del timeout. `qwen3:4b`
se conserva como candidato de mayor coste computacional, pero no como valor por
defecto para la demo en CPU.

El valor por defecto pasa de seis a tres chunks. En la evaluación de retrieval,
K=3 mantuvo Recall@3 de 0,90 y MRR de 0,85, con mejor precisión y un prompt más
corto. Para una demo local es un mejor equilibrio entre cobertura y latencia.

## Límites de esta medición

- Una observación no permite generalizar la calidad de un modelo.
- La CPU domina la latencia; una GPU dedicada o un servidor de inferencia la
  reducirían sin cambiar el diseño del RAG.
- Los casos correcto, ambiguo y sin evidencia formarán parte de la evaluación
  final del agente, cuando ya existan router, reintento y fallback.
- Los tests actuales sí cubren de forma determinista salida válida, falta de
  evidencia, JSON inválido y citas inventadas, sin depender de Ollama.
