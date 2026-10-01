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

## Ampliación posterior: query expansion

La prueba end-to-end con una descripción de alcance trasero descubrió un error
que el dataset inicial no mostraba. El relato incluía «semáforo en rojo» y «choca
por detrás», pero la búsqueda directa priorizó páginas sobre semáforos y no
recuperó la regla de `ALCANCE TRASERO`.

La solución no consiste en obligar al LLM a responder sin evidencia ni en
incrementar indiscriminadamente `top_k`. Para `accident_description` se generan
dos consultas:

1. Una consulta técnica construida mediante reglas deterministas.
2. El relato original sin modificar.

Ejemplos de vocabulario:

| Expresión del relato | Término añadido para recuperar el manual |
| --- | --- |
| choca o golpea por detrás | alcance trasero, daños delanteros y traseros |
| cambia de carril | cambio o invasión de carril |
| retrocede | marcha atrás |
| estaba aparcado | vehículo estacionado |
| se incorpora | incorporación a la circulación |

Cada consulta genera su propio ranking. `SemanticRetriever.retrieve_many`
elimina duplicados y aplica Reciprocal Rank Fusion:

```text
score(chunk) = suma de 1 / (60 + posición_en_cada_ranking)
```

La consulta técnica se procesa primero para resolver empates a su favor, pero
el relato original conserva los detalles que no cubre el diccionario. No se
añaden hechos ni se decide responsabilidad durante esta transformación.

### Resultado observado

| Configuración | Páginas recuperadas en las tres primeras posiciones |
| --- | --- |
| Solo relato original | 67, 97, 87 |
| Relato + expansión + fusión | 75, 67, 18 |

La página 75, ausente antes de la mejora, pasó a primera posición y contiene la
regla `MARCHA ATRÁS/ALCANCE TRASERO`.

## Decisión sobre top_k

Subir de K=3 a K=6 aumentó Recall de 0,9000 a 0,9333, pero redujo Precision de
0,4000 a 0,2167 y mantuvo MRR en 0,8500. Por tanto, el incremento aporta solo
3,33 puntos porcentuales de recall a cambio de casi duplicar el contexto y
añadir más fragmentos irrelevantes.

En el caso de alcance trasero, la evidencia correcta ya ocupa la primera
posición después de la expansión. Aumentar K no puede mejorar su presencia y sí
puede dificultar la generación. Se mantiene K=3 como valor predeterminado. Una
futura evaluación podrá comparar K dinámico o reranking, pero no se incorpora
sin medirlo sobre el conjunto completo.
