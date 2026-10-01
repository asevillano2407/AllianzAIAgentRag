# Fase 3: embeddings y recuperación vectorial

## Objetivo

Convertir los chunks de texto en vectores, almacenarlos localmente y recuperar
la evidencia semánticamente más próxima a una pregunta. Esta fase no genera aún
la respuesta final: retrieval y generación se evalúan por separado.

## Primer incremento: contratos y Qdrant embebido

El código separa cuatro responsabilidades:

1. `EmbeddingProvider` define cómo convertir documentos y consultas en vectores.
2. `IndexingService` aplica el modelo por lotes para limitar el uso de memoria.
3. `VectorStore` define las operaciones de persistencia y búsqueda.
4. `QdrantVectorStore` implementa esas operaciones sobre Qdrant local.

Esta separación permite comparar E5, Qwen y BGE sin cambiar el indexado ni la
búsqueda. Los tests usan un proveedor determinista pequeño, pero Qdrant es real
y se ejecuta en memoria. Así las pruebas no descargan modelos, no necesitan red
y permiten localizar los fallos en la capa correcta.

## Por qué una colección por modelo

Los vectores generados por modelos distintos no comparten necesariamente:

- el mismo número de dimensiones;
- el mismo espacio semántico;
- la misma normalización o instrucciones de consulta.

Por ello `collection_name_for_model` genera un nombre específico y estable a
partir del prefijo configurado y del modelo. Añade una huella SHA-256 corta para
evitar colisiones entre nombres que se normalicen de forma parecida.

## Identidad y reindexado

Qdrant acepta enteros o UUID como identificadores de punto. Nuestros `chunk_id`
son cadenas del dominio, así que se transforman de manera determinista con
UUIDv5 y el identificador original permanece en el payload. Indexar de nuevo el
mismo chunk reemplaza su punto; no crea duplicados.

## Validaciones

Antes de persistir se comprueba que:

- exista al menos un chunk;
- haya exactamente un vector por chunk;
- todos los vectores tengan la misma dimensión;
- los valores sean finitos;
- la colección existente use esa dimensión y distancia coseno.

La colección nunca se elimina automáticamente si hay una incompatibilidad. Un
borrado silencioso ocultaría un cambio de modelo y podría destruir datos.

## Configuración

```ini
ALLIANZ_QDRANT_PATH=data/qdrant
ALLIANZ_QDRANT_COLLECTION_PREFIX=allianz_manual
ALLIANZ_EMBEDDING_MODEL=qwen3-embedding:0.6b
ALLIANZ_EMBEDDING_BATCH_SIZE=8
ALLIANZ_OLLAMA_TIMEOUT_SECONDS=300
ALLIANZ_RETRIEVAL_TOP_K=6
```

## Recorrido local reproducible

El adaptador de Ollama llama a `/api/embed` con `truncate=false`, valida cada
vector y diferencia los errores de conexión, HTTP, timeout y formato. El lector
de JSONL valida de nuevo todos los chunks antes de enviarlos al modelo.

```powershell
allianz-index --input data/processed/chunks.jsonl
allianz-search "¿Cuándo caduca un siniestro en CICOS?" --top-k 6
allianz-evaluate-retrieval --top-k 6
```

El indexado escribe el progreso de cada lote en stderr y reserva stdout para el
resumen JSON final. El dataset versionado contiene diez preguntas con páginas
relevantes verificadas y permite calcular Precision@K, Recall@K y MRR sin
mezclar la calidad del retrieval con la futura generación de respuestas.

## Comparación local de modelos

Ambos modelos se ejecutaron en CPU sobre los mismos 160 chunks, las mismas diez
preguntas y `top_k=6`:

| Modelo | Dimensiones | Precision@6 | Recall@6 | MRR | Consulta observada |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3-embedding:0.6b` | 1.024 | 0,2167 | 0,9333 | 0,8500 | 11,47 s |
| `bge-m3` | 1.024 | 0,1833 | 0,8667 | 0,8417 | 15,98 s |

| Modelo | Dimensiones | Precision@K | Recall@K | MRR |
| --- | ---: | ---: | ---: | ---: |
| `qwen3-embedding:0.6b` (K=3) | 1.024 | 0,4000 | 0,9000 | 0,8500 |
| `qwen3-embedding:0.6b` (K=4) | 1.024 | 0,3000 | 0,9000 | 0,8500 |
| `qwen3-embedding:0.6b` (K=5) | 1.024 | 0,2400 | 0,9000 | 0,8500 |
| `qwen3-embedding:0.6b` (K=7) | 1.024 | 0,1857 | 0,9333 | 0,8500 |
| `bge-m3` (K=3) | 1.024 | 0,3000 | 0,7333 | 0,8000 |
| `bge-m3` (K=4) | 1.024 | 0,2500 | 0,7667 | 0,8250 |
| `bge-m3` (K=5) | 1.024 | 0,2000 | 0,7667 | 0,8250 |
| `bge-m3` (K=7) | 1.024 | 0,1571 | 0,8667 | 0,8417 |

Los tiempos corresponden a una sola consulta de caducidad y sirven solo como
referencia del equipo local. La Precision@6 baja refleja que se solicitan seis
chunks aunque muchas preguntas tengan una sola página etiquetada; Recall y MRR
son más informativos para este primer conjunto.

Se mantiene `qwen3-embedding:0.6b` como modelo por defecto: supera a BGE-M3 en
las tres métricas, alcanzó menor latencia en la consulta observada y su descarga
es aproximadamente la mitad (639 MB frente a 1,2 GB). Ambos índices contienen
160 puntos, usan distancia coseno y permanecen disponibles en colecciones
separadas para repetir o ampliar la evaluación.
