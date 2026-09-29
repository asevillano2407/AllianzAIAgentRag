# Allianz Claims Copilot

Repositorio de aprendizaje para construir, paso a paso, un sistema agentic RAG
sobre el manual CIDE, ASCIDE y CICOS.

## Estado de la rama develop

Estamos en la **Fase 0: definición del problema y arquitectura**. Todavía no hay
código del RAG. Primero validaremos qué debe hacer el sistema, qué queda fuera y
cómo sabremos si funciona.

La solución se diseñará para ejecutarse completamente en local y con coste
monetario cero. Compararemos varios modelos locales de embeddings y generación
antes de justificar la configuración final.

El MVP debe ser objetivamente entregable en cuatro días. Por ello trabajará
exclusivamente con el texto extraído del manual; el tratamiento multimodal se
documentará como posible evolución, pero no se implementará.

Documentos de esta fase:

- [Requisitos y criterios de aceptación](docs/phase_0_requirements.md)
- [Decisiones iniciales de arquitectura](docs/phase_0_architecture.md)

## Método de trabajo

Cada fase seguirá el mismo ciclo:

1. Entender el objetivo técnico.
2. Implementar el cambio mínimo.
3. Revisar el código y sus alternativas.
4. Ejecutar pruebas.
5. Validar el resultado antes de continuar.

La solución completa permanece en `master` como referencia. En `develop` la
reconstruiremos sin copiar su implementación.
