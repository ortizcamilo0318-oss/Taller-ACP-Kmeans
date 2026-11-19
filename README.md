# Perfiles de estudiantes universitarios y depresión: ACP y K-Means

Taller individual · Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali

- **Aplicación desplegada:** https://taller-acp-kmeans-7kmsofdqiexzfjk8jgswfp.streamlit.app/
- **Video de sustentación:** _pega aquí el enlace del video_

## Contexto y pregunta

La salud mental de los estudiantes tiene costos económicos: deserción, menor rendimiento y menor
acumulación de capital humano. Como los recursos de bienestar universitario son limitados, interesa
saber si los estudiantes se agrupan en perfiles reconocibles y si alguno concentra más casos de depresión.

**Pregunta:** ¿qué dimensiones resumen las diferencias entre estudiantes, existen segmentos con
características similares y difiere entre ellos la proporción de estudiantes con depresión?

## Datos

- **Fuente:** [*Student Depression Detection Dataset*](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset),
  publicado en Kaggle por MKhoirul Mustaqim. La base original fue publicada en Mendeley Data por
  Obaidul Haque y Samiul Alom.
- **Licencia:** CC BY-SA 4.0.
- **Archivo:** `data/Student_Depression_Dataset.csv` (incluido en el repositorio).
- **Observación:** cada fila es un estudiante que respondió la encuesta. La fuente no nombra el país;
  las ocho divisiones de origen corresponden a las de Bangladesh.
- **Tamaño:** 4.464 estudiantes y 13 variables; sin valores faltantes ni duplicados.

| Variable | Tipo | Uso |
|---|---|---|
| `Age` | Cuantitativa | Modelo |
| `Current_semester` | Cuantitativa | Modelo |
| `CGPA` (promedio acumulado, 0-4) | Cuantitativa | Modelo |
| `Cost_without_study` (gastos de vida estimados, sin costos académicos; la fuente no indica moneda ni periodo) | Cuantitativa | Modelo |
| `Family_background` (nivel socioeconómico de la familia) | Ordinal → 1-3 | Modelo |
| `Health_condition` (salud física general) | Ordinal → 1-4 | Modelo |
| `Depression` | Sí/No | Contraste (no entra al modelo) |
| Género, relación, estado civil, vivienda, amigos, división | Categóricas | Descripción de los grupos |

**Limitaciones de la fuente:** encuesta autorreportada sin diseño muestral documentado; la depresión
no es un diagnóstico clínico; corte transversal (asociaciones, no causas); las variables ordinales
se tratan como numéricas asumiendo igual distancia entre categorías.

## Metodología

1. **Exploración:** estructura, tipos, estadísticas descriptivas, faltantes, duplicados y matriz de correlaciones.
2. **Preparación:** codificación de las dos variables ordinales, exclusión de las categóricas sin orden,
   reserva de la depresión como variable de contraste y estandarización (media 0, desviación 1).
3. **ACP:** varianza explicada y acumulada, gráfico de sedimentación, cargas y biplot. El número de
   componentes se elige con los resultados (umbral de 80 % de varianza acumulada, comparado con el criterio de Kaiser).
4. **K-Means:** sobre los componentes conservados; comparación de K = 2…10 con el método del codo y
   el coeficiente de silueta; perfiles de los grupos en las variables originales y contraste con la depresión.

Todo se calcula en el código (`analysis.py`) cada vez que se abre la aplicación o se cambia la
configuración; no hay resultados fijos ni capturas.

## Resultados principales (configuración por defecto)

- **ACP:** las correlaciones entre variables son bajas, salvo gasto–nivel socioeconómico (0,60) y
  edad–semestre (0,33). Dos componentes solo acumulan 49,5 % de la varianza; se conservan **4 componentes
  (82,7 %)**: capacidad económica, avance en la carrera y dos que combinan rendimiento académico y salud.
- **K-Means:** la silueta es baja y casi plana (0,19 a 0,22 entre K = 2 y K = 10) y no hay un codo nítido:
  los datos forman un continuo, no grupos separados. Se toma **K = 4** por parsimonia e interpretabilidad (silueta 0,203).

| Grupo | % muestra | Rasgo distintivo | % con depresión |
|---|---|---|---|
| A | 24,8 | Gasto y nivel socioeconómico altos, más jóvenes, salud algo menor | 32,8 |
| B | 26,8 | Mayores, semestres avanzados, buen promedio | 18,1 |
| C | 26,3 | Nivel socioeconómico y gasto bajos, primeros semestres | 22,0 |
| D | 22,1 | Promedio académico muy bajo | 36,9 |

(Depresión en toda la muestra: 26,9 %. Las letras son etiquetas, no un orden.)

- **Lectura:** la depresión aparece en todos los segmentos, pero es más frecuente donde el rendimiento
  académico o la salud son peores. La capacidad económica diferencia mucho a los estudiantes pero casi no se
  correlaciona con la depresión.
- **Limitación central:** con siluetas tan bajas los grupos son perfiles típicos con fronteras difusas,
  no categorías para clasificar a un estudiante individual.

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`.

## Estructura

```
app.py              Interfaz de Streamlit (6 secciones)
analysis.py         Preparación, ACP y K-Means
data/               Base de datos original
requirements.txt    Dependencias
.streamlit/         Tema visual
```
