# Allianz Claims RAG Agent

Repositorio de aprendizaje para construir, paso a paso, un sistema agentic RAG
sobre el manual CIDE, ASCIDE y CICOS.

## Estado de la rama develop

La **Fase 4: LLM local y salida estructurada** está cerrada. El retrieval se
evaluó con dos embeddings y `qwen3-embedding:0.6b` quedó seleccionado. Para la
generación comparamos `qwen3:4b` con `llama3.2:3b`; Llama queda como opción local
por defecto porque completó el flujo validado dentro del límite de tiempo.

La solución se diseñará para ejecutarse completamente en local y con coste
monetario cero. Compararemos varios modelos locales de embeddings y generación
antes de justificar la configuración final.

El MVP debe ser objetivamente entregable en cuatro días. Por ello trabajará
exclusivamente con el texto extraído del manual; el tratamiento multimodal se
documentará como posible evolución, pero no se implementará.

Documentos de esta fase:

- [Requisitos y criterios de aceptación](docs/phase_0_requirements.md)
- [Decisiones iniciales de arquitectura](docs/phase_0_architecture.md)
- [Fundamentos del proyecto](docs/phase_1_foundation.md)
- [Extracción y chunking del manual](docs/phase_2_ingestion.md)
- [Embeddings y recuperación vectorial](docs/phase_3_retrieval.md)
- [LLM local y salida estructurada](docs/phase_4_generation.md)

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

## Generación local validada

Con `llama3.2:3b` descargado en Ollama:

```powershell
ollama pull llama3.2:3b
allianz-generate "¿Cuál es el plazo de caducidad de una reclamación CICOS?" `
  --query-type manual_question
```

La aplicación no acepta ciegamente el texto del modelo: valida el esquema y
comprueba que el identificador, la página y la cita literal existan en los
chunks recuperados. El modelo, el número de tokens y la duración se incluyen en
la salida para facilitar la evaluación.

Las descripciones de accidentes utilizan dos consultas cuando se reconoce una
maniobra: el relato original y una expansión con vocabulario del manual, como
«choca por detrás» → «alcance trasero». Sus resultados se fusionan antes de
generar la respuesta, sin utilizar un segundo LLM ni inventar hechos.
