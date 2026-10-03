# Fase 2 Extracción y chunking del manual

## Objetivo

Convertir el PDF en fragmentos limpios, reproducibles y trazables antes de
introducir embeddings o una base vectorial.

## Flujo implementado

```text
PDF local
  -> extracción página a página
  -> normalización conservadora
  -> detección de sección
  -> chunking dentro de cada página
  -> JSONL inspeccionable
```

### Extracción

`pypdf` devuelve las 111 páginas físicas, incluidas aquellas que no contienen
texto. Una página nunca comparte contenido con otra. El campo `page` utiliza
numeración física basada en uno y no el número impreso dentro del manual.

### Normalización

La limpieza se limita a transformaciones justificables:

- normalización Unicode NFC;
- unificación de saltos de línea;
- sustitución de espacios no separables;
- eliminación de guiones invisibles;
- reducción de espacios horizontales repetidos;
- eliminación del número de página impreso solo cuando coincide con el esperado;
- conservación de los límites de párrafo.

No se corrigen palabras mediante reglas lingüísticas porque una corrección
agresiva podría modificar vocabulario jurídico o de seguros.

### Chunking

La configuración inicial utiliza 1200 caracteres y 150 caracteres de solapamiento.
El corte busca primero párrafos, después frases y finalmente espacios. Se ha
elegido una ventana por caracteres para que el corpus sea independiente del
tokenizador del modelo que evaluaremos en la siguiente fase.

Los dos valores se configuran sin modificar código mediante `.env`:

```ini
ALLIANZ_CHUNK_SIZE=1200
ALLIANZ_CHUNK_OVERLAP=150
```

La aplicación valida que el tamaño esté entre 200 y 4000 caracteres y que el
solapamiento no supere la mitad del chunk. Una configuración inválida detiene la
ejecución con un error en lugar de producir silenciosamente un corpus defectuoso.

Los chunks:

- nunca mezclan páginas;
- heredan la sección más reciente;
- omiten páginas vacías o que solo contienen un encabezado;
- reciben un ID SHA-256 derivado de fuente, página, posición y contenido.

## Resultado sobre el manual

| Métrica | Resultado |
| --- | ---: |
| Páginas físicas | 111 |
| Páginas representadas por chunks | 109 |
| Página sin texto extraíble | 32 |
| Páginas sin evidencia indexable | 31, 32 |
| Chunks | 160 |
| IDs únicos | 160 |
| Longitud mínima | 53 caracteres |
| Longitud máxima | 1197 caracteres |
| SHA-256 del JSONL de validación | `2F75FA123AA6137B093F3E38371CE0301E0D66E008F3F6BBC937396334337FAF` |

Dos ejecuciones consecutivas produjeron el mismo archivo y el mismo hash.

## Limitaciones confirmadas visualmente

- La página física 32 contiene una imagen de la Declaración Amistosa de
  Accidente y no tiene capa de texto utilizable. Queda fuera del MVP textual.
- La página 31 contiene únicamente un encabezado de continuación y no genera un
  chunk sin evidencia.
- Las tablas conservan su texto, pero la extracción lineal puede perder relaciones
  espaciales entre filas y columnas. Esta limitación debe cubrirse con preguntas de
  evaluación antes de decidir si hace falta un parser de tablas.

## Validación automatizada

- 29 pruebas superadas.
- 95 % de cobertura del paquete.
- Ruff sin incidencias.
- Casos cubiertos: PDF inexistente, extensión incorrecta, páginas vacías,
  normalización, IDs deterministas, límites de chunking, serialización UTF-8 y
  error legible de CLI.
