# Fase 6: demo local con Streamlit

## Objetivo

Presentar el agente de forma comprensible sin duplicar la lógica del RAG ni
depender de una inferencia de varios minutos durante la exposición. La interfaz
es una capa de presentación local; no contiene reglas de negocio.

## Por qué Streamlit

Los entregables no exigen una API pública ni un frontend de producción. Para una
prueba técnica, Streamlit permite construir una interfaz visual con poco código,
se ejecuta localmente y no añade servicios de pago. Se incorpora como dependencia
opcional `demo`, por lo que la librería principal no obliga a instalarlo.

No se añade FastAPI: introducir una API, servidor y cliente separados aumentaría
el alcance sin mejorar la demostración de retrieval, generación o LangGraph.

## Dos modos explícitos

### Resultado evaluado

Carga los cinco resultados reales de la evaluación final de `qwen3:4b` desde un
JSONL incluido en el paquete. Es instantáneo y aparece etiquetado como resultado
guardado; no se presenta como una inferencia en directo.

Este modo permite enseñar también el caso C incorrecto. La demo no oculta la
limitación: muestra su evaluación automática como fallo de negocio.

### Agente real

Invoca el mismo `run_local_agent` utilizado por la CLI, con reranking opcional,
12 candidatos y 3 chunks finales. Necesita Ollama, Qdrant y los modelos locales.
La interfaz avisa de la latencia de 6–9 minutos observada en CPU y conserva el
resultado en `session_state` después de la ejecución.

## Información mostrada

- aplicabilidad de CIDE/ASCIDE;
- responsabilidad según el convenio;
- confianza cualitativa;
- conclusión determinista;
- hechos e información ausente;
- citas separadas por decisión, con página y chunk;
- evaluación automática en resultados guardados;
- consultas y páginas de retrieval;
- recorrido LangGraph en ejecuciones reales;
- fallback, reintentos y ajustes fail-soft;
- modelo, tokens y duración.

La información técnica permanece dentro de un desplegable para no saturar la
vista principal orientada a negocio.

El encabezado utiliza una copia local del logotipo de Allianz SE. La interfaz lo
identifica expresamente como prototipo técnico no oficial y conserva la fuente y
atribución en `demo/assets/README.md`; no necesita descargar imágenes al abrirse.

## Reutilización del runtime

El ensamblaje de Ollama, Qdrant, reranker y grafo se extrajo a
`orchestration/runtime.py`. Tanto la CLI como Streamlit llaman a
`run_local_agent`; la demo no mantiene una segunda implementación del flujo.

Las transformaciones de datos se encuentran en `demo/service.py` y no importan
Streamlit. Esto permite probar carga, validación y serialización sin interfaz,
red, Ollama ni modelos.

## Ejecución

```powershell
python -m pip install -e ".[dev,rerank,demo]"
allianz-demo
```

Para el modo real deben existir previamente:

- Ollama en `http://localhost:11434`;
- `qwen3-embedding:0.6b` y `qwen3:4b`;
- colección Qdrant indexada;
- reranker descargado o accesible en la caché local.

## Guion recomendado para la presentación

1. Abrir el modo evaluado y seleccionar D, cambio de carril.
2. Mostrar la decisión y la cita de la página 75.
3. Abrir el detalle técnico y explicar retrieval, ajustes y latencia.
4. Seleccionar E para explicar aplicabilidad frente a responsabilidad.
5. Seleccionar C para reconocer la limitación conocida.
6. Enseñar el modo real sin depender de terminar la inferencia durante la sesión.

## Límites

- No es una interfaz de producción ni implementa autenticación.
- No se expone fuera de la máquina local.
- Una ejecución real bloquea esa sesión de Streamlit mientras Ollama responde.
- Los resultados guardados son evidencia del benchmark, no respuestas generadas
  en el momento.

## Validación

La aplicación se ejecutó mediante el framework headless de pruebas de Streamlit:

- modo evaluado y caso A: título, selector, seis métricas y conclusión sin errores;
- caso C: evaluación completa visible como incorrecta;
- modo agente real: formulario, aviso de latencia y botón renderizados sin invocar
  Ollama;
- carga de los cinco JSONL y serialización de resultados reales cubiertas por
  tests unitarios.
