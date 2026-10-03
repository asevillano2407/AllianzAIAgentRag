# Fase 0 Decisiones iniciales de arquitectura

## 1 Flujo propuesto

```text
Usuario
  |
  v
Validación de entrada
  |
  v
Router determinista
  |
  v
Recuperación de fragmentos del manual
  |
  v
Generación estructurada con LLM
  |
  v
Validación de esquema y citas
  |
  +---- respuesta válida ----> Respuesta al usuario
  |
  +---- error ----> un reintento ----> fallback seguro
```

## 2 Decisiones propuestas

Estas decisiones son un punto de partida. Las validaremos durante la
implementación en lugar de asumir que todas son correctas.

### Python 3.12

Es compatible con el stack de LangGraph, Qdrant y modelos locales y permite
usar tipos modernos. Es además la versión disponible actualmente en el equipo,
por lo que evitamos instalar y mantener un segundo runtime sin necesidad.

### RAG en lugar de fine tuning

El conocimiento está concentrado en un manual pequeño. Necesitamos citar la
fuente y actualizar el sistema cuando cambie el documento. Reindexar es más
barato, rápido y auditable que entrenar el modelo.

### RAG exclusivamente textual para el MVP

La recuperación se realizará sobre el texto extraído del manual. No
implementaremos OCR, embeddings visuales ni un LLM multimodal. Aunque esas
capacidades podrían aportar valor sobre formularios, croquis o fotografías, no
son necesarias para validar el caso principal y pondrían en riesgo una entrega
completa en cuatro días.

La propuesta multimodal quedará descrita en la sección de evolución futura,
incluyendo la limitación de que una página basada en imagen no será recuperable
por el MVP.

### Qdrant local como base vectorial candidata

Permite persistencia en disco sin desplegar un servidor y ofrece filtros por
metadatos y un camino claro hacia recuperación híbrida. No esperamos que mejore
por sí solo la calidad sobre un corpus tan pequeño. Lo elegimos para aprender y
mostrar conceptos de una base vectorial más próxima a producción.

La aplicación dependerá de una interfaz propia de almacenamiento. Si Qdrant
introduce una complejidad desproporcionada, podremos contrastarlo o sustituirlo
por Chroma sin modificar el dominio ni la orquestación.

### Embeddings multilingües locales y evaluados

El manual está en español y contiene vocabulario específico. Compararemos como
mínimo un baseline ligero con un candidato de mayor capacidad. Los candidatos
iniciales son `multilingual-e5-small`, `qwen3-embedding:0.6b` y `BAAI/bge-m3`.
La elección final dependerá de Recall@K, MRR, latencia y memoria en nuestro
conjunto versionado, no de un benchmark genérico.

### LLM locales mediante Ollama

La generación no utilizará Vertex AI, Gemini API ni ningún proveedor externo.
Ollama ejecutará modelos descargados en el equipo y el proveedor quedará detrás
de una interfaz propia. Compararemos al menos dos modelos de tamaño adecuado a
los 31 GB de RAM disponibles, manteniendo constantes el prompt, los fragmentos y
los parámetros de generación. Los 15 GB indicados por el sistema son memoria GPU
total/compartida, no VRAM dedicada garantizada, por lo que la aplicación debe
funcionar correctamente mediante CPU y RAM. La aceleración GPU será opcional.

Mediremos fidelidad, corrección, relevancia, citas, rechazo seguro, latencia,
memoria y tokens por segundo. La elección del modelo final quedará documentada.

### LangGraph para la orquestación

El estado del agente será explícito. Los nodos tendrán una sola responsabilidad
y el número de iteraciones estará limitado. Las decisiones deterministas no se
delegarán innecesariamente al LLM.

### FastAPI y Streamlit

FastAPI expondrá el caso de uso y Streamlit servirá para la demo. Ambos deberán
usar el mismo servicio de aplicación para evitar lógica duplicada.

## 3 Capas previstas

```text
Interfaces        CLI | FastAPI | Streamlit
Aplicación        Servicio de análisis
Orquestación      Grafo y estado
Dominio           Modelos, reglas y validación
Infraestructura   PDF | embeddings locales | Qdrant | Ollama
```

## 4 Orden de implementación

1. Estructura del proyecto y configuración.
2. Extracción, limpieza y chunking.
3. Embeddings y almacenamiento vectorial.
4. Evaluación del retrieval.
5. LLM y salida estructurada.
6. Grafo agentic y guardrails.
7. API e interfaz.
8. Docker local, CI y observabilidad.
9. Evaluación final y preparación de la entrevista.

El despliegue cloud no forma parte del MVP. El README incluirá solamente una
propuesta de evolución a producción para demostrar criterio arquitectónico sin
introducir coste, credenciales ni trabajo de infraestructura en la prueba.

Tampoco forma parte del MVP la recuperación multimodal. La arquitectura futura
podría incorporar OCR y un modelo visual local para extraer hechos de una D.A.A.
o un croquis, siempre con validación humana antes de consultar el RAG textual.

## 5 Primera decisión que validaremos

La Fase 1 no instalará todo el stack. Crearemos un paquete Python mínimo con
configuración, modelos y pruebas. Esto permite confirmar la estructura antes de
añadir dependencias pesadas como Qdrant, sentence-transformers u Ollama.
