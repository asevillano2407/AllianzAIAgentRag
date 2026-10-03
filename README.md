# Allianz Claims RAG Agent

Repositorio de aprendizaje para construir, paso a paso, un sistema agentic RAG
sobre el manual CIDE, ASCIDE y CICOS.

## Estado de la rama develop

La **Fase 5: orquestación agentic** está cerrada. El grafo clasifica la entrada,
ejecuta retrieval y generación, reintenta solo fallos transitorios y siempre
termina con una respuesta validada o un fallback seguro. El retrieval se
evaluó con dos embeddings y `qwen3-embedding:0.6b` quedó seleccionado. Para la
generación comparamos `llama3.2:3b`, `qwen3:4b`, `qwen3:8b` y `gemma3:4b`.
`qwen3:4b` queda seleccionado: en la evaluación final completó 5/5 casos y
alcanzó un 80 % de corrección de negocio.

La solución se diseñará para ejecutarse completamente en local y con coste
monetario cero. Compararemos varios modelos locales de embeddings y generación
antes de justificar la configuración final.

El MVP debe ser objetivamente entregable en cuatro días. Por ello trabajará
exclusivamente con el texto extraído del manual; el tratamiento multimodal se
documentará como posible evolución, pero no se implementará.

Documentos de esta fase:

- [Índice y estado de la documentación](docs/README.md)
- [Requisitos y criterios de aceptación](docs/phase_0_requirements.md)
- [Decisiones iniciales de arquitectura](docs/phase_0_architecture.md)
- [Fundamentos del proyecto](docs/phase_1_foundation.md)
- [Extracción y chunking del manual](docs/phase_2_ingestion.md)
- [Embeddings y recuperación vectorial](docs/phase_3_retrieval.md)
- [LLM local y salida estructurada](docs/phase_4_generation.md)
- [Orquestación agentic con LangGraph](docs/phase_5_agentic_workflow.md)
- [Enfoque técnico y decisiones](docs/technical_approach.md)
- [Evaluación de los cinco casos de demostración](docs/demo_cases_evaluation.md)

## Método de trabajo

Cada fase seguirá el mismo ciclo:

1. Entender el objetivo técnico.
2. Implementar el cambio mínimo.
3. Revisar el código y sus alternativas.
4. Ejecutar pruebas.
5. Validar el resultado antes de continuar.

La solución completa permanece en `master` como referencia. En `develop` la
reconstruiremos sin copiar su implementación.

## Preparación local

Requisitos: Python 3.12 y PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

El modo editable refleja los cambios del código fuente sin reinstalar el paquete.
El archivo `.env` es local y Git lo ignora para evitar versionar configuración o
futuros secretos.

## Validación

```powershell
python -m ruff check .
python -m pytest --cov=allianz_claims_rag_agent --cov-report=term-missing
```

## Ingesta local del manual

El PDF permanece en `data/raw/` y los datos derivados en `data/processed/`;
ambos directorios están excluidos de Git.

```powershell
allianz-ingest `
  --input data/raw/Manual-cide-ascide-y-cicos.pdf `
  --output data/processed/chunks.jsonl
```

El comando informa del número de páginas, chunks y páginas sin texto
recuperable. El JSONL conserva para cada fragmento la fuente, página física,
sección e identificador determinista.

## Indexado y búsqueda local

Con Ollama en ejecución y `qwen3-embedding:0.6b` descargado:

```powershell
allianz-index --input data/processed/chunks.jsonl
allianz-search "¿Cuándo caduca un siniestro en CICOS?" --top-k 6
allianz-evaluate-retrieval --top-k 6
```

Cada modelo usa una colección Qdrant distinta para evitar mezclar espacios
vectoriales incompatibles. El indexado informa del progreso por lotes y emite
al terminar un resumen JSON con el modelo, la colección y la dimensión.

El reranking local es opcional para no instalar PyTorch en el flujo básico. Para
comparar recuperación vectorial frente a recuperación con cross-encoder:

```powershell
python -m pip install -e ".[dev,rerank]"

allianz-evaluate-retrieval --top-k 4
allianz-evaluate-retrieval --rerank --candidate-k 12 --top-k 4
```

Por defecto, `--rerank` utiliza `BAAI/bge-reranker-v2-m3`. La primera ejecución
descarga el modelo; las siguientes pueden funcionar con la copia local en caché.

## Generación local validada

Con `qwen3:4b` descargado en Ollama:

```powershell
ollama pull qwen3:4b
allianz-generate "¿Cuál es el plazo de caducidad de una reclamación CICOS?" `
  --query-type manual_question
```

La aplicación no acepta ciegamente el texto del modelo: valida el esquema, la
coherencia entre aplicabilidad y responsabilidad y el soporte de las citas. En
accidentes, los errores seguros de metadatos se reparan y una decisión sin
evidencia se degrada a `undetermined`, en vez de descartar toda la respuesta. El
modelo, los ajustes aplicados, los tokens y la duración se incluyen en la salida.

Las descripciones de accidentes utilizan dos consultas cuando se reconoce una
maniobra: el relato original y una expansión con vocabulario del manual, como
«choca por detrás» → «alcance trasero». Sus resultados se fusionan antes de
generar la respuesta, sin utilizar un segundo LLM ni inventar hechos.

Configuración recomendada para la demo completa:

```powershell
allianz-agent `
  "El vehículo A cambia de carril y roza lateralmente al vehículo B." `
  --rerank `
  --candidate-k 12 `
  --top-k 3
```
