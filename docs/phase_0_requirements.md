# Fase 0 Requisitos y criterios de aceptación

## 1 Problema que queremos resolver

Un tramitador necesita consultar el manual CIDE, ASCIDE y CICOS y analizar
descripciones de accidentes de automóvil. Leer el manual completo para cada caso
es lento y una respuesta de un LLM sin evidencia sería difícil de verificar.

La solución debe recuperar los apartados relevantes del manual, generar una
respuesta estructurada y mostrar las páginas que sustentan la conclusión.

## 2 Usuario principal

El usuario del MVP es un tramitador o especialista de siniestros que quiere:

- Formular una pregunta concreta sobre el manual.
- Describir un accidente en lenguaje natural.
- Identificar vehículos, maniobras y circunstancias relevantes.
- Entender la responsabilidad según los convenios.
- Revisar la evidencia utilizada antes de aceptar la respuesta.

## 3 Distinción esencial

La responsabilidad dentro de CIDE, ASCIDE o CICOS no equivale automáticamente a
responsabilidad legal, cobertura de una póliza, importe de indemnización o
responsabilidad penal.

El sistema será una ayuda para una persona. No ejecutará pagos, rechazos ni otras
acciones que afecten directamente a un expediente.

## 4 Alcance del MVP

### Incluido

- El manual PDF facilitado como única fuente de conocimiento.
- Preguntas y descripciones de accidentes en español.
- Extracción y limpieza reproducible del PDF.
- Fragmentos con metadatos de fuente, sección y página.
- Recuperación semántica de los fragmentos relevantes.
- Generación de una respuesta estructurada mediante un LLM.
- Citas verificables contra los fragmentos recuperados.
- Respuesta explícita cuando no exista evidencia suficiente.
- API, interfaz sencilla, evaluación y pruebas automatizadas.

### Fuera de alcance

- Integración con sistemas reales de Allianz.
- Tratamiento de datos personales de asegurados.
- Decisiones automáticas de pago, cobertura o rechazo.
- Asesoramiento jurídico.
- Entrenamiento o fine tuning de un modelo fundacional.
- Alta disponibilidad, autenticación corporativa e infraestructura productiva.
- Despliegue real en servicios cloud.
- Dependencia de APIs comerciales, créditos promocionales o free tiers.
- OCR, embeddings de imagen y recuperación multimodal.
- Interpretación automática de fotografías, croquis o declaraciones escaneadas.

## 5 Requisitos funcionales

| ID | Requisito |
| --- | --- |
| RF01 | El sistema debe cargar el PDF configurado sin rutas codificadas en el código. |
| RF02 | Debe conservar la página y la fuente de cada fragmento. |
| RF03 | Debe crear identificadores deterministas para evitar duplicados al reindexar. |
| RF04 | Debe recuperar los fragmentos más relacionados con una consulta. |
| RF05 | Debe distinguir una pregunta general de una descripción de accidente. |
| RF06 | Debe devolver una respuesta con conclusión, hechos, información ausente y confianza. |
| RF07 | Toda conclusión material debe incluir citas válidas. |
| RF08 | Una cita debe corresponder a un fragmento entregado al LLM. |
| RF09 | Si la evidencia es insuficiente, el sistema debe evitar una conclusión forzada. |
| RF10 | El flujo agentic debe tener una condición de terminación y un límite de reintentos. |

## 6 Requisitos no funcionales

| ID | Requisito |
| --- | --- |
| RNF01 | La solución completa debe poder ejecutarse localmente con coste monetario cero. |
| RNF02 | La aplicación y sus pruebas no dependerán de APIs ni de credenciales externas. |
| RNF03 | Ingestión, recuperación, generación y orquestación estarán separadas. |
| RNF04 | El código usará tipos, validación de entradas y errores explícitos. |
| RNF05 | El mismo servicio de aplicación alimentará CLI, API e interfaz. |
| RNF06 | El resultado mostrará que el manual facilitado data de 2004. |
| RNF07 | La ejecución será observable mediante logs y métricas sin registrar datos sensibles. |
| RNF08 | Los modelos se descargarán una vez y podrán utilizarse posteriormente sin conexión. |
| RNF09 | El tamaño de los modelos se adaptará a 31 GB de RAM; la memoria GPU total/compartida no se considerará VRAM dedicada garantizada. |
| RNF10 | El alcance debe permitir obtener un MVP probado y demostrable en cuatro días. |

## 7 Casos mínimos de demostración

1. Alcance trasero ante un semáforo en rojo.
2. Colisión múltiple con cinco vehículos.
3. Vehículo estacionado dañado por un vehículo no identificado.
4. Colisión lateral durante un cambio de carril.
5. Accidente con alcoholemia y lesiones.
6. Pregunta cuya respuesta no aparezca en el manual para validar el rechazo seguro.

## 8 Evaluación

### Recuperación

- Recall at K para comprobar si aparece una página relevante.
- Mean Reciprocal Rank para medir su posición.
- Revisión manual de relevancia sobre un conjunto de preguntas etiquetadas.
- Comparación de al menos dos modelos de embeddings con el mismo corpus, preguntas y chunking.

### Generación

- Validez de las citas.
- Completitud del esquema de respuesta.
- Faithfulness, correctness y relevancia de la respuesta.
- Comparación de al menos dos LLM locales usando exactamente los mismos fragmentos recuperados.

### Sistema

- Latencia por consulta.
- Tasa de fallback.
- Errores del runtime local del LLM.
- Tokens por segundo y memoria utilizada.
- Coste monetario de servicios externos, que debe permanecer en cero.

## 9 Criterios de aceptación de la solución

- El repositorio puede instalarse desde cero siguiendo el README.
- La ingestión procesa las 111 páginas y no mezcla texto entre páginas.
- Un conjunto versionado contiene al menos diez preguntas de evaluación.
- El retrieval alcanza inicialmente un Recall at 6 de al menos 0.80 sobre ese conjunto.
- Ninguna cita puede apuntar a un fragmento que el LLM no haya recibido.
- El agente siempre termina y realiza como máximo un reintento.
- Las pruebas importantes funcionan sin credenciales externas.
- El informe de evaluación justifica los modelos elegidos con métricas reproducibles.
- La demo incluye un caso correcto, uno ambiguo y uno sin evidencia suficiente.

## 10 Riesgos conocidos

- La capa de texto del PDF antiguo pierde algunos caracteres acentuados.
- Una página basada en imagen o la estructura visual de una tabla puede perderse en un RAG exclusivamente textual; se documentará como limitación conocida.
- El manual data de 2004 y puede no representar criterios actuales.
- Un embedding semántico puede recuperar texto relacionado pero no suficiente.
- Un LLM puede producir una conclusión plausible pero incorrecta.
- Un modelo demasiado grande puede provocar falta de memoria o una latencia inaceptable.
- La descarga inicial de modelos requiere red y espacio suficiente en disco.

Estos riesgos deberán convertirse en controles técnicos o quedar visibles en la
respuesta y en la presentación.
