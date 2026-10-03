# Fase 5: orquestación agentic con LangGraph

## Objetivo

Convertir retrieval y generación en un flujo explícito, observable y acotado.
El agente debe clasificar la entrada, recuperar evidencia, generar una respuesta
validada y terminar siempre, incluso cuando un servicio local falla.

## Por qué LangGraph

El caso no necesita un agente autónomo que elija herramientas libremente. Sí
necesita estado compartido, rutas condicionales y control de reintentos. LangGraph
permite representar estas decisiones de forma visible sin mezclar la lógica de
negocio con un bucle manual.

```text
START
  |
  v
route ---------------- validación + clasificación determinista
  |
  v
retrieve <----------- un reintento si el fallo es transitorio
  |
  v
generate <----------- un reintento si Ollama falla temporalmente
  |  \
  |   +-------------- salida inválida o reintentos agotados
  v                                      |
 END                                 fallback
                                          |
                                         END
```

## Estado explícito

`ClaimsAgentState` conserva únicamente datos necesarios entre nodos:

- consulta validada y tipo detectado;
- consultas de retrieval original y expandida;
- chunks recuperados;
- generación y respuesta validadas;
- contador de reintentos;
- estado final y uso de fallback;
- último error esperado;
- recorrido de nodos para observabilidad.

Los prompts no se almacenan en el estado porque pueden reconstruirse a partir de
la consulta, el tipo y los chunks. Esto evita duplicar información derivada.

## Nodos

### `route`

Valida la entrada con `AnalysisRequest`, determina `manual_question` o
`accident_description` mediante reglas estables y construye las consultas para
retrieval. No utiliza el LLM: gastar una inferencia lenta en una decisión que
puede resolverse con reglas sería innecesario.

### `retrieve`

Ejecuta retrieval semántico y query expansion. Los errores de embeddings o
Qdrant se consideran potencialmente transitorios y admiten como máximo el número
de reintentos configurado.

### `generate`

Solicita la salida estructurada y aplica los guardrails de la Fase 4. Un error de
Ollama puede reintentarse. En accidentes, una página incorrecta o una cita
inválida se trata primero mediante fail-soft: se repara el metadato seguro o se
degrada únicamente la decisión que perdió soporte. Solo una salida que continúa
violando el esquema o una validación no reparable pasa directamente al fallback.

Una respuesta válida con confianza baja también es un resultado válido y seguro.
No se reintenta automáticamente para evitar duplicar varios minutos de inferencia
sin una estrategia nueva.

Los ajustes del generador se conservan junto a la respuesta. Esto distingue una
decisión emitida directamente por el modelo de otra alineada o degradada por la
aplicación, sin exponer detalles internos como parte de la conclusión al usuario.

### `fallback`

Devuelve una `AnalysisResponse` válida, con confianza baja, sin citas inventadas
y sin exponer detalles internos del error en la conclusión destinada al usuario.

## Garantía de terminación

`ALLIANZ_MAX_AGENT_RETRIES=1` permite una ejecución inicial y como máximo un
segundo intento para un fallo transitorio. El contador forma parte del estado y
las aristas condicionales terminan en éxito o fallback cuando se supera el límite.

No existe ninguna transición desde `fallback` hacia un nodo anterior, por lo que
el grafo no puede entrar en un bucle infinito.

## Comando

Después de reinstalar el paquete editable para registrar el nuevo ejecutable:

```powershell
python -m pip install -e ".[dev]"

allianz-agent `
  "El vehículo A está detenido y el vehículo B choca por detrás contra A."
```

La salida muestra respuesta, tipo detectado, estado, fallback, reintentos,
recorrido, consultas utilizadas, chunks y métricas de generación.

## Pruebas deterministas

Los tests usan retriever y generador falsos; no dependen de Ollama ni Qdrant.
Cubren:

- routing de pregunta y descripción de accidente;
- ruta feliz;
- query expansion dentro del grafo;
- reintento de retrieval;
- reintento de generación;
- fallback inmediato ante salida inválida;
- fallback al agotar el límite;
- validación de configuración.

## Smoke test end-to-end

Se ejecutó el grafo real con la pregunta «¿Cuál es el plazo de caducidad de una
reclamación CICOS?». El resultado fue:

- tipo detectado: `manual_question`;
- recorrido: `route -> retrieve -> generate`;
- estado: `completed`;
- reintentos: 0;
- fallback: no utilizado;
- conclusión correcta: un año desde la fecha del accidente;
- cita literal validada: página física 14;
- latencia de generación observada: 191,2 segundos sobre CPU.

Es una prueba funcional controlada, no un benchmark de rendimiento.

## Cierre y limitaciones

- El router está orientado al dominio y requiere ampliar sus ejemplos si aparecen
  nuevas formas de consulta.
- El reintento no cambia todavía el modelo ni el contexto; solo recupera fallos
  transitorios.
- No se necesita checkpointer porque cada análisis es independiente y no existe
  una pausa humana que deba reanudarse.
- El grafo y sus rutas están implementados y cubiertos por tests deterministas;
  la Fase 5 queda cerrada.
- La experiencia de demo pertenece a la fase siguiente: formato de entrada y
  forma de mostrar recorrido, evidencia, ajustes y latencia sin saturar al usuario.
- La siguiente fase expondrá este mismo grafo mediante una interfaz mínima, sin
  duplicar lógica de negocio.

La configuración final recomendada para la demo inyecta el retriever híbrido con
`candidate_k=12`, `top_k=3` y `qwen3:4b`. LangGraph no conoce esos proveedores
concretos: solo consume los protocolos `EvidenceRetriever` y
`GroundedGenerator`, preservando la separación entre orquestación y negocio.
