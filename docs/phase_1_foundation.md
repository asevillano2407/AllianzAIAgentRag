# Fase 1 Fundamentos del proyecto

## Objetivo

Crear una base Python instalable, validada y testeable antes de incorporar PDF,
embeddings, Qdrant, Ollama o LangGraph.

## Estructura

```text
src/allianz_claims_rag_agent/
|-- config.py        Configuración y variables de entorno
|-- domain/          Contratos independientes de infraestructura
|-- errors.py        Categorías explícitas de error
tests/               Pruebas que reflejan la estructura del paquete
```

El `src` layout obliga a que el paquete se importe como una dependencia
instalada o mediante una ruta configurada de forma explícita. Esto evita que una
prueba pase accidentalmente solo porque se ejecuta desde la raíz del repositorio.

## Decisiones

- Pydantic valida entradas y salidas en los límites de la aplicación.
- `extra="forbid"` detecta campos inesperados en lugar de ignorarlos.
- Las colecciones usan `default_factory` para evitar estado mutable compartido.
- La configuración utiliza el prefijo `ALLIANZ_` y admite un `.env` local no
  versionado.
- Los errores esperados tienen categorías propias; no se capturará `Exception`
  silenciosamente.
- Los proveedores concretos se añadirán en fases posteriores detrás de
  interfaces, por lo que el dominio no importa Qdrant, Ollama ni LangGraph.

## Criterios de validación

- El paquete se instala en modo editable.
- `pytest` ejecuta las pruebas sin servicios externos.
- `ruff check` no informa de problemas.
- La configuración rechaza valores fuera de los límites acordados.
- Los modelos rechazan entradas vacías, páginas inválidas y campos desconocidos.

## Resultado de la primera validación

- 10 pruebas superadas.
- 90 % de cobertura del paquete inicial.
- Ruff sin incidencias.
- Instalación editable completada con Python 3.12.
