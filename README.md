# Allianz Claims RAG Agent

Repositorio de aprendizaje para construir, paso a paso, un sistema agentic RAG
sobre el manual CIDE, ASCIDE y CICOS.

## Estado de la rama develop

Estamos en la **Fase 1: fundamentos del proyecto**. La definición del problema y
la arquitectura inicial ya están validadas. En esta fase construiremos un paquete
Python mínimo con configuración, contratos de dominio, errores y pruebas.

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
