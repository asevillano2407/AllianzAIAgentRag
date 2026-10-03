# Guía de documentación por fases

Este índice conecta requisitos, decisiones, implementación, pruebas y resultados.
Cada fase cerrada conserva su razonamiento para que el proyecto pueda explicarse
sin depender del historial de commits ni de una conversación externa.

| Fase | Estado | Objetivo | Documento principal |
| --- | --- | --- | --- |
| 0 | Cerrada | Alcance, requisitos y arquitectura | `phase_0_requirements.md`, `phase_0_architecture.md` |
| 1 | Cerrada | Paquete, configuración y dominio | `phase_1_foundation.md` |
| 2 | Cerrada | Extracción, limpieza y chunking | `phase_2_ingestion.md` |
| 3 | Cerrada y ampliada | Embeddings, Qdrant, evaluación y query expansion | `phase_3_retrieval.md` |
| 4 | Cerrada y evaluada | Generación local, decisiones de negocio, citas y selección del LLM | `phase_4_generation.md` |
| 5 | Cerrada | Grafo agentic, router, reintento y fallback | `phase_5_agentic_workflow.md` |
| 6 | Pendiente | API e interfaz de demostración | Se creará al comenzar la fase |
| 7 | Pendiente | Docker, CI y observabilidad | Se creará al comenzar la fase |
| 8 | Pendiente | Evaluación final y presentación | Se creará al comenzar la fase |

## Criterio de cierre documental

Una fase se considera documentada cuando su archivo explica:

1. El problema y el objetivo.
2. El flujo y las responsabilidades implementadas.
3. Las decisiones y alternativas relevantes.
4. La configuración y los comandos reproducibles.
5. Los tests y resultados observados.
6. Las limitaciones y el siguiente paso.

Las mejoras descubiertas en fases posteriores se registran también en la fase
donde se originó la responsabilidad técnica. Por eso la query expansion aparece
en Fase 3 como mejora de retrieval y en Fase 4 como parte del flujo end-to-end.

Evaluaciones transversales:

- `technical_approach.md`: arquitectura, decisiones, riesgos y trazabilidad para los entregables.
- `demo_cases_evaluation.md`: evolución de las pruebas y resultados finales de los cinco casos.
