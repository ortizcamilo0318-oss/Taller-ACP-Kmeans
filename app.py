"""Perfiles de estudiantes universitarios y depresión: ACP y K-Means.

Ejecutar con:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an

st.set_page_config(page_title="Perfiles de estudiantes · ACP y K-Means", page_icon="🎓", layout="wide")

PALETTE = ["#1f6f8b", "#e07a5f", "#3d9970", "#8e6bbf", "#d4a017", "#c44569", "#5c6b73", "#2a9d8f", "#bc6c25", "#577590"]
DIVERGING = "RdBu_r"
L = an.LABELS


# ----------------------------------------------------------------------------
# Cálculos (en caché: se recalculan solo cuando cambia la configuración)
# ----------------------------------------------------------------------------
@st.cache_data
def get_raw():
    return an.load_raw()


@st.cache_data
def get_prepared(drop_zero):
    return an.prepare(get_raw(), drop_semester_zero=drop_zero)


@st.cache_data
def get_pca(drop_zero, variables):
    z = an.standardize(get_prepared(drop_zero), list(variables))
    return z, an.run_pca(z)


def get_model_input(drop_zero, variables, n_comp, base):
    z, pca = get_pca(drop_zero, variables)
    return pca["scores"].iloc[:, :n_comp] if base == "Componentes principales" else z


@st.cache_data(show_spinner="Comparando valores de K…")
def get_k_evaluation(drop_zero, variables, n_comp, base, k_max):
    return an.evaluate_k(get_model_input(drop_zero, variables, n_comp, base), 2, k_max)


@st.cache_data(show_spinner="Ajustando K-Means…")
def get_clusters(drop_zero, variables, n_comp, base, k):
    return an.fit_kmeans(get_model_input(drop_zero, variables, n_comp, base), k)


def show(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=60, b=10), template="plotly_white")
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------------
# Barra lateral: navegación y configuración del análisis
# ----------------------------------------------------------------------------
st.sidebar.title("🎓 Perfiles de estudiantes")
page = st.sidebar.radio(
    "Sección",
    ["1. Contexto", "2. Exploración", "3. Preparación", "4. ACP", "5. K-Means", "6. Conclusiones"],
)
st.sidebar.divider()
st.sidebar.subheader("Configuración del análisis")
variables = st.sidebar.multiselect(
    "Variables del modelo", an.MODEL_VARS, default=an.DEFAULT_VARS, format_func=lambda v: L[v]
)
drop_zero = st.sidebar.checkbox(
    "Excluir registros con semestre 0",
    value=False,
    help="42 estudiantes reportan semestre 0. Puede ser un error de captura o estudiantes aún sin iniciar.",
)
if len(variables) < 4:
    st.warning("Selecciona al menos cuatro variables en la barra lateral para ejecutar el análisis.")
    st.stop()
variables = tuple(variables)

raw = get_raw()
df = get_prepared(drop_zero)
z, pca = get_pca(drop_zero, variables)
variance, loadings, scores = pca["variance"], pca["loadings"], pca["scores"]
suggested = an.suggest_components(variance, 80)
kaiser = an.kaiser_components(variance)

n_comp = st.sidebar.slider(
    "Componentes a conservar",
    2,
    len(variables),
    suggested,
    key=f"ncomp_{len(variables)}_{suggested}",
    help="Por defecto: el mínimo número de componentes que acumula al menos 80 % de la varianza.",
)
base = st.sidebar.radio(
    "K-Means se aplica sobre",
    ["Componentes principales", "Variables estandarizadas"],
    help="Usar los componentes conservados descarta el ruido de los últimos componentes.",
)
k = st.sidebar.slider("Número de grupos (K)", 2, 10, 4)
retained = variance["Varianza acumulada (%)"].iloc[n_comp - 1]
st.sidebar.caption(f"{len(df):,} estudiantes · {len(variables)} variables · {n_comp} componentes ({retained:.1f} % de la varianza)")

cfg = (drop_zero, variables, n_comp, base)


# ----------------------------------------------------------------------------
# 1. Contexto
# ----------------------------------------------------------------------------
if page.startswith("1"):
    st.title("¿Qué perfiles de estudiantes universitarios existen y cómo se relacionan con la depresión?")
    st.markdown(
        """
La salud mental de los estudiantes universitarios tiene costos económicos directos: deserción,
menor rendimiento académico y, a largo plazo, menor acumulación de capital humano. Las universidades
y los programas de bienestar disponen de recursos limitados, por lo que resulta útil saber **si los
estudiantes se agrupan en perfiles reconocibles** (académicos, económicos y de salud) y **si alguno
de esos perfiles concentra más casos de depresión**.

### Pregunta que orienta el ejercicio
> ¿Qué dimensiones resumen las diferencias entre estudiantes y existen segmentos con características
> similares? Y, sin haberle mostrado esa variable al modelo, ¿difiere la proporción de estudiantes
> con depresión entre los segmentos?
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estudiantes", f"{len(raw):,}")
    c2.metric("Variables en la fuente", raw.shape[1])
    c3.metric("Variables cuantitativas del modelo", len(an.MODEL_VARS))
    c4.metric("Reportan depresión", f"{(raw['Depression'] == 'Yes').mean():.1%}")

    st.markdown(
        """
### Los datos
- **Fuente:** [*Student Depression Detection Dataset*](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset),
  publicado en Kaggle por MKhoirul Mustaqim con licencia CC BY-SA 4.0. La base original fue publicada en
  Mendeley Data por Obaidul Haque y Samiul Alom. Es una encuesta pública y sin datos que identifiquen a las personas.
- **Unidad de observación:** cada fila es un estudiante que respondió la encuesta. La fuente no nombra el
  país, pero las ocho divisiones de origen (Dhaka, Chittagong, Khulna…) son las de Bangladesh.
- **Cobertura:** 4.464 estudiantes de 17 a 28 años, de los semestres 0 a 13.
- **Variables del modelo:** edad, semestre, promedio acumulado (CGPA; en los datos va de 2,25 a 4),
  gastos de vida estimados sin contar los costos académicos (la fuente no indica moneda ni periodo)
  y dos variables ordinales codificadas: nivel socioeconómico de la familia (1-3) y salud física general (1-4).
- **Variable de contraste:** la depresión reportada (Sí/No) **no entra al modelo**; se usa al final
  para comparar los grupos.

### ¿Por qué ACP y K-Means?
- **ACP:** variables como los gastos de vida y el nivel socioeconómico, o la edad y el semestre,
  miden en parte lo mismo. El ACP permite comprobarlo y resumirlas en dimensiones interpretables
  (por ejemplo, «capacidad económica» o «avance en la carrera») antes de medir distancias.
- **K-Means:** no existe una clasificación previa de estudiantes; se quiere *descubrir* si hay
  segmentos. K-Means agrupa por cercanía en variables estandarizadas, lo que se ajusta a
  variables cuantitativas como estas y produce centros fáciles de describir.

### Limitaciones de la fuente
- Es una encuesta autorreportada, sin diseño muestral documentado: no es representativa de todos
  los estudiantes del país (Dhaka aporta un tercio de la muestra). La fuente tampoco documenta la moneda
  ni el periodo de los gastos, así que esa variable solo se interpreta en términos relativos.
- La depresión es una respuesta Sí/No, no un diagnóstico clínico.
- Nivel socioeconómico y estado de salud son categorías ordenadas; al codificarlas como 1, 2, 3…
  se asume que la distancia entre categorías es igual, lo cual es una simplificación.
- Es un corte transversal: muestra asociaciones, no causas.
"""
    )
    st.info("Usa la barra lateral para recorrer el análisis y cambiar variables, componentes y número de grupos.")

# ----------------------------------------------------------------------------
# 2. Exploración
# ----------------------------------------------------------------------------
elif page.startswith("2"):
    st.title("Exploración de los datos")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(raw):,}")
    c2.metric("Variables", raw.shape[1])
    c3.metric("Valores faltantes", int(raw.isna().sum().sum()))
    c4.metric("Filas duplicadas", int(raw.duplicated().sum()))

    tab1, tab2, tab3, tab4 = st.tabs(["Estructura", "Distribuciones", "Variables categóricas", "Correlaciones"])
    with tab1:
        st.subheader("Primeros registros")
        st.dataframe(raw.head(10), width="stretch")
        st.subheader("Tipos, faltantes y valores únicos")
        st.dataframe(an.quality_report(raw), width="stretch")
        st.subheader("Estadísticas descriptivas (variables del modelo)")
        st.dataframe(df[an.MODEL_VARS].rename(columns=L).describe().T.round(2), width="stretch")
        st.caption(
            "Las escalas son muy distintas: el gasto se mide en miles y el CGPA entre 2,25 y 4. "
            "Sin estandarizar, el gasto dominaría cualquier cálculo de distancias."
        )
    with tab2:
        var = st.selectbox("Variable", an.MODEL_VARS, format_func=lambda v: L[v])
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=var, nbins=30, color_discrete_sequence=PALETTE, title=f"Distribución de {L[var]}")
            fig.update_layout(xaxis_title=L[var], yaxis_title="Número de estudiantes", bargap=0.05)
            show(fig)
        with c2:
            fig = px.box(
                df, x="Depression", y=var, color="Depression", color_discrete_sequence=PALETTE,
                title=f"{L[var]} según depresión reportada",
            )
            fig.update_layout(xaxis_title="Depresión reportada", yaxis_title=L[var], showlegend=False)
            show(fig)
        st.caption(
            "El diagrama de caja compara la variable entre quienes reportan depresión y quienes no. "
            "La depresión no se usa para construir los grupos; solo sirve de contraste."
        )
    with tab3:
        cat = st.selectbox("Variable categórica", list(an.CATEGORICAL), format_func=lambda v: an.CATEGORICAL[v])
        counts = raw[cat].value_counts().reset_index()
        counts.columns = [cat, "Estudiantes"]
        fig = px.bar(counts, x=cat, y="Estudiantes", text="Estudiantes", color_discrete_sequence=PALETTE,
                     title=f"Estudiantes por {an.CATEGORICAL[cat].lower()}")
        fig.update_layout(xaxis_title=an.CATEGORICAL[cat], yaxis_title="Número de estudiantes")
        show(fig)
    with tab4:
        corr = df[list(variables)].rename(columns=L).corr()
        fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale=DIVERGING, zmin=-1, zmax=1,
                        title="Matriz de correlaciones de Pearson (variables del modelo)")
        show(fig, 520)
        pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().sort_values(key=np.abs, ascending=False)
        top = pairs.index[0]
        st.markdown(
            f"""
**Lectura.** La correlación más fuerte es entre **{top[0]}** y **{top[1]}** ({pairs.iloc[0]:.2f}).
En general las correlaciones son bajas: salvo el par económico (gasto y nivel socioeconómico) y el par
de trayectoria (edad y semestre), cada variable aporta información propia. Esto anticipa dos cosas:
el ACP **no podrá** resumir todo en dos componentes, y los grupos difícilmente serán nubes separadas.
"""
        )

# ----------------------------------------------------------------------------
# 3. Preparación
# ----------------------------------------------------------------------------
elif page.startswith("3"):
    st.title("Preparación de los datos")
    st.markdown(
        f"""
| Decisión | Qué se hizo | Por qué |
|---|---|---|
| Faltantes y duplicados | La base no tiene ninguno; el código igualmente los elimina si aparecieran | Reproducibilidad si la fuente se actualiza |
| Variables excluidas | Género, relación, estado civil, vivienda, amigos y división de origen | Son categóricas sin orden: no tienen distancia euclidiana con sentido |
| Depresión | Se **excluye** del modelo y se reserva como contraste | Si entrara, los grupos la reproducirían por construcción |
| Variables ordinales | Nivel socioeconómico (1-3) y estado de salud (1-4) codificados en su orden | Tienen orden natural y sentido analítico; se asume igual distancia entre categorías |
| Semestre 0 | {"Se excluyen" if drop_zero else "Se conservan"} los 42 registros (opción en la barra lateral) | No se puede confirmar que sean errores; se deja como decisión visible |
| Valores extremos | Se conservan (gastos de hasta 32.000) | Son plausibles dentro de la clase alta; eliminarlos borraría justo la variación económica |
| Estandarización | Media 0 y desviación 1 en todas las variables | ACP y K-Means dependen de la escala y de distancias |

**Registros que entran al modelo:** {len(df):,} de {len(raw):,}.
"""
    )
    st.subheader("Codificación de las variables ordinales")
    c1, c2 = st.columns(2)
    for col, box in zip(an.ORDINAL_MAPS, (c1, c2)):
        tab = pd.Series(an.ORDINAL_MAPS[col], name="Código").to_frame()
        tab["Estudiantes"] = raw[col].value_counts()
        box.markdown(f"**{an.CATEGORICAL[col]}**")
        box.dataframe(tab, width="stretch")

    st.subheader("Antes y después de estandarizar")
    before = df[list(variables)].agg(["mean", "std"]).T
    after = z.agg(["mean", "std"]).T
    comp = pd.concat([before, after], axis=1).round(2)
    comp.columns = ["Media original", "Desv. original", "Media estandarizada", "Desv. estandarizada"]
    comp.index = [L[v] for v in comp.index]
    st.dataframe(comp, width="stretch")

    long = z.rename(columns=L).melt(var_name="Variable", value_name="Valor estandarizado")
    fig = px.box(long, x="Variable", y="Valor estandarizado", color_discrete_sequence=PALETTE,
                 title="Variables estandarizadas: todas quedan en la misma escala")
    fig.update_layout(xaxis_title="", yaxis_title="Desviaciones estándar respecto a la media")
    show(fig)
    st.caption(
        "Tras estandarizar, cada variable pesa lo mismo. Un valor de +1 significa «una desviación estándar "
        "por encima del promedio de la muestra»."
    )

# ----------------------------------------------------------------------------
# 4. ACP
# ----------------------------------------------------------------------------
elif page.startswith("4"):
    st.title("Análisis de Componentes Principales")
    c1, c2, c3 = st.columns(3)
    c1.metric("Componentes conservados", n_comp)
    c2.metric("Varianza conservada", f"{retained:.1f} %")
    c3.metric("Criterio de Kaiser (valor propio > 1)", f"{kaiser} componentes")

    st.subheader("Varianza explicada")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(variance.set_index("Componente").round(2), width="stretch")
    with col2:
        fig = go.Figure()
        fig.add_bar(x=variance["Componente"], y=variance["Varianza explicada (%)"], name="Por componente",
                    marker_color=[PALETTE[0] if i < n_comp else "#c9d3d8" for i in range(len(variance))])
        fig.add_scatter(x=variance["Componente"], y=variance["Varianza acumulada (%)"], name="Acumulada",
                        mode="lines+markers", line=dict(color=PALETTE[1]))
        fig.add_hline(y=80, line_dash="dot", line_color="gray", annotation_text="80 %")
        fig.update_layout(title="Gráfico de sedimentación y varianza acumulada",
                          xaxis_title="Componente principal", yaxis_title="Varianza explicada (%)",
                          legend=dict(orientation="h", y=-0.2))
        show(fig)

    st.markdown(
        f"""
**¿Cuántos componentes conservar?** No hay un codo marcado: la varianza cae de forma gradual porque
las variables están poco correlacionadas. Los dos primeros componentes solo acumulan
**{variance['Varianza acumulada (%)'].iloc[1]:.1f} %**, de modo que quedarse con dos sería perder
demasiada información. El criterio de Kaiser sugiere **{kaiser}** componentes
({variance['Varianza acumulada (%)'].iloc[kaiser - 1]:.1f} %) y el umbral de 80 % sugiere **{suggested}**
({variance['Varianza acumulada (%)'].iloc[suggested - 1]:.1f} %). Con la configuración actual se conservan
**{n_comp}** componentes, que retienen **{retained:.1f} %** de la variación original.
"""
    )

    st.subheader("Cargas: qué mide cada componente")
    kept = loadings.iloc[:, :n_comp].rename(index=L)
    fig = px.imshow(kept.round(2), text_auto=True, color_continuous_scale=DIVERGING, zmin=-1, zmax=1, aspect="auto",
                    title="Cargas de las variables en los componentes conservados")
    fig.update_layout(xaxis_title="Componente", yaxis_title="")
    show(fig, 440)
    lines = []
    for cp in kept.columns:
        topv = kept[cp].abs().sort_values(ascending=False).head(2).index
        desc = " y ".join(f"**{v}** ({kept.loc[v, cp]:+.2f})" for v in topv)
        share = variance.set_index("Componente").loc[cp, "Varianza explicada (%)"]
        lines.append(f"- **{cp}** ({share:.1f} %): dominado por {desc}.")
    st.markdown("\n".join(lines))
    st.caption(
        "Una carga cercana a ±1 indica que la variable pesa mucho en el componente; el signo indica el sentido. "
        "El signo global de un componente es arbitrario: lo que importa es qué variables van juntas y cuáles se oponen."
    )
    if set(variables) == set(an.DEFAULT_VARS):
        st.markdown(
            """
**Interpretación con las seis variables.** El primer componente es una dimensión de **capacidad económica**
(gastos de vida y nivel socioeconómico suben juntos). El segundo es de **avance en la carrera**
(edad y semestre). El tercero y el cuarto combinan **rendimiento académico y salud**: en uno se suman
(estudiantes con buen promedio y buena salud frente a quienes tienen ambos bajos) y en el otro se oponen
(distingue buen promedio con mala salud de lo contrario).
"""
        )

    st.subheader("Estudiantes en el espacio de componentes")
    c1, c2, c3 = st.columns(3)
    cx = c1.selectbox("Eje horizontal", list(scores.columns), index=0)
    cy = c2.selectbox("Eje vertical", list(scores.columns), index=1)
    color_by = c3.selectbox("Colorear por", ["Depression", "Family_background", "Health_condition"],
                            format_func=lambda v: an.CATEGORICAL[v])
    vx = variance.set_index("Componente").loc[cx, "Varianza explicada (%)"]
    vy = variance.set_index("Componente").loc[cy, "Varianza explicada (%)"]
    plot = scores[[cx, cy]].join(df[[color_by]])
    fig = px.scatter(plot, x=cx, y=cy, color=color_by, opacity=0.45, color_discrete_sequence=PALETTE,
                     title=f"Estudiantes en {cx} y {cy} con las cargas de las variables (biplot)")
    scale = 3.5
    for v in loadings.index:
        fig.add_annotation(x=loadings.loc[v, cx] * scale, y=loadings.loc[v, cy] * scale, ax=0, ay=0,
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#222")
        fig.add_annotation(x=loadings.loc[v, cx] * scale * 1.12, y=loadings.loc[v, cy] * scale * 1.12,
                           text=L[v].split(" (")[0], showarrow=False, font=dict(size=11, color="#222"))
    fig.update_layout(xaxis_title=f"{cx} ({vx:.1f} % de la varianza)", yaxis_title=f"{cy} ({vy:.1f} % de la varianza)",
                      legend_title=an.CATEGORICAL[color_by])
    show(fig, 560)
    st.info(
        f"Este plano resume **{vx + vy:.1f} %** de la información original. El resto de la variación "
        "está en los demás componentes, así que dos puntos cercanos aquí pueden ser distintos en otras dimensiones."
    )

# ----------------------------------------------------------------------------
# 5. K-Means
# ----------------------------------------------------------------------------
elif page.startswith("5"):
    st.title("Segmentación con K-Means")
    if base == "Componentes principales":
        st.markdown(
            f"K-Means se aplica sobre los **{n_comp} componentes conservados** ({retained:.1f} % de la varianza). "
            "Así se agrupa con la información principal y se deja fuera la variación residual de los últimos componentes."
        )
    else:
        st.markdown(f"K-Means se aplica sobre las **{len(variables)} variables estandarizadas**, sin reducción previa.")

    st.subheader("¿Cuántos grupos? Codo y silueta")
    ev = get_k_evaluation(*cfg, 10)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(ev, x="K", y="Inercia", markers=True, color_discrete_sequence=PALETTE,
                      title="Método del codo: inercia según K")
        fig.add_vline(x=k, line_dash="dot", line_color="gray")
        fig.update_layout(xaxis_title="Número de grupos (K)", yaxis_title="Inercia (suma de distancias² al centro)")
        show(fig, 380)
    with c2:
        fig = px.line(ev, x="K", y="Silueta", markers=True, color_discrete_sequence=[PALETTE[1]],
                      title="Coeficiente de silueta promedio según K")
        fig.add_vline(x=k, line_dash="dot", line_color="gray")
        fig.update_layout(xaxis_title="Número de grupos (K)", yaxis_title="Silueta promedio (de -1 a 1)")
        show(fig, 380)

    best = ev.loc[ev["Silueta"].idxmax()]
    cur = ev.set_index("K").loc[k]
    drops = (-ev["Inercia"].pct_change() * 100).round(1)
    st.dataframe(
        ev.assign(**{"Reducción de inercia (%)": drops}).set_index("K").round({"Inercia": 0, "Silueta": 3}).T,
        width="stretch",
    )
    st.markdown(
        f"""
**Cómo leer estas gráficas.** La inercia siempre baja al aumentar K; se busca el punto donde la mejora
se vuelve pequeña. La silueta mide qué tan bien queda cada estudiante en su grupo frente al grupo vecino
(cerca de 1: grupos compactos y separados; cerca de 0: grupos que se tocan).

**Resultado.** La silueta más alta es **{best['Silueta']:.3f}** con K = {int(best['K'])}, y con el K elegido
(K = {k}) es **{cur['Silueta']:.3f}**. La curva de silueta es casi plana y todos los valores son bajos:
los datos **no contienen grupos naturalmente separados**, sino un continuo que K-Means divide en regiones.
Tampoco hay un codo nítido: la inercia se reduce de forma gradual. Por eso la elección de K no puede apoyarse
solo en las métricas. Se propone **K = 4** como valor por defecto por parsimonia: su silueta
({ev.set_index('K').loc[4, 'Silueta']:.3f}) queda a solo {best['Silueta'] - ev.set_index('K').loc[4, 'Silueta']:.3f}
del máximo, y cuatro grupos se pueden describir con claridad usando las variables originales, mientras que con
más grupos aparecen segmentos pequeños que solo subdividen a los anteriores. Puedes mover K en la barra lateral
para comprobar cómo cambian los perfiles.
"""
    )

    labels, sil = get_clusters(*cfg, k)
    groups = sorted(labels.unique())
    colors = {g: PALETTE[i] for i, g in enumerate(groups)}

    st.subheader(f"Los {k} grupos en el espacio de componentes")
    c1, c2 = st.columns(2)
    cx = c1.selectbox("Eje horizontal", list(scores.columns), index=0)
    cy = c2.selectbox("Eje vertical", list(scores.columns), index=1)
    vx = variance.set_index("Componente").loc[cx, "Varianza explicada (%)"]
    vy = variance.set_index("Componente").loc[cy, "Varianza explicada (%)"]
    plot = scores[[cx, cy]].assign(Grupo=labels).sort_values("Grupo")
    fig = px.scatter(plot, x=cx, y=cy, color="Grupo", opacity=0.5, color_discrete_map=colors,
                     title=f"Grupos de K-Means proyectados en {cx} y {cy}")
    cent = plot.groupby("Grupo")[[cx, cy]].mean()
    fig.add_scatter(x=cent[cx], y=cent[cy], mode="markers+text", text=cent.index, textposition="top center",
                    marker=dict(symbol="x", size=14, color="black"), name="Centro del grupo")
    fig.update_layout(xaxis_title=f"{cx} ({vx:.1f} % de la varianza)", yaxis_title=f"{cy} ({vy:.1f} % de la varianza)")
    show(fig, 540)
    st.caption(
        f"El plano muestra {vx + vy:.1f} % de la varianza. Los grupos pueden verse superpuestos aquí y estar "
        "separados en otros componentes: prueba con otros ejes."
    )

    st.subheader("Perfil de cada grupo con las variables originales")
    prof = an.cluster_profile(df, labels, list(variables))
    st.dataframe(prof.rename(columns=L).round(2), width="stretch")
    zprof = an.cluster_zscores(df, labels, list(variables))
    fig = px.imshow(zprof.rename(columns=L).round(2), text_auto=True, color_continuous_scale=DIVERGING,
                    zmin=-1.5, zmax=1.5, aspect="auto",
                    title="Promedio de cada grupo en desviaciones estándar respecto a toda la muestra")
    fig.update_layout(xaxis_title="", yaxis_title="Grupo")
    show(fig, 120 + 60 * k)
    st.markdown("**Descripción automática** (rasgos a más de 0,4 desviaciones del promedio):")
    st.markdown("\n".join(f"- **Grupo {g}** ({prof.loc[g, '% de la muestra']:.0f} %): {an.describe_cluster(zprof.loc[g])}."
                          for g in groups))
    st.caption("Las letras son solo etiquetas: no indican orden, tamaño ni jerarquía entre grupos.")

    st.subheader("Contraste: variables que no entraron al modelo")
    c1, c2 = st.columns(2)
    with c1:
        dep = (df["Depression"] == "Yes").groupby(labels).mean().mul(100).rename("pct").reset_index()
        fig = px.bar(dep, x="Grupo", y="pct", color="Grupo", color_discrete_map=colors, text_auto=".1f",
                     title="Estudiantes que reportan depresión por grupo (%)")
        overall = (df["Depression"] == "Yes").mean() * 100
        fig.add_hline(y=overall, line_dash="dot", line_color="gray", annotation_text=f"Muestra: {overall:.1f} %")
        fig.update_layout(xaxis_title="Grupo", yaxis_title="% con depresión reportada", showlegend=False)
        show(fig, 380)
    with c2:
        cat = st.selectbox("Otra variable categórica", [c for c in an.CATEGORICAL if c != "Depression"],
                           format_func=lambda v: an.CATEGORICAL[v])
        ct = pd.crosstab(labels, df[cat], normalize="index").mul(100).reset_index().melt(id_vars="Grupo", value_name="pct")
        fig = px.bar(ct, x="Grupo", y="pct", color=cat, color_discrete_sequence=PALETTE,
                     title=f"Composición de cada grupo por {an.CATEGORICAL[cat].lower()} (%)")
        fig.update_layout(xaxis_title="Grupo", yaxis_title="% del grupo", legend_title=an.CATEGORICAL[cat])
        show(fig, 320)

    st.subheader("Calidad de la asignación")
    sg = sil.groupby(labels).mean().rename("Silueta").reset_index()
    fig = px.bar(sg, x="Grupo", y="Silueta", color="Grupo", color_discrete_map=colors, text_auto=".3f",
                 title="Silueta promedio por grupo")
    fig.update_layout(xaxis_title="Grupo", yaxis_title="Silueta promedio", showlegend=False)
    show(fig, 340)
    st.markdown(
        f"**{(sil < 0).mean():.1%}** de los estudiantes tiene silueta negativa: quedarían igual o mejor en el grupo "
        "vecino. Son los casos de frontera, difíciles de separar."
    )

# ----------------------------------------------------------------------------
# 6. Conclusiones
# ----------------------------------------------------------------------------
else:
    st.title("Discusión de resultados y conclusiones")
    labels, sil = get_clusters(*cfg, k)
    prof = an.cluster_profile(df, labels, list(variables))
    zprof = an.cluster_zscores(df, labels, list(variables))
    dep = (df["Depression"] == "Yes").groupby(labels).mean() * 100
    overall = (df["Depression"] == "Yes").mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Componentes conservados", n_comp)
    c2.metric("Varianza conservada", f"{retained:.1f} %")
    c3.metric("Grupos (K)", k)
    c4.metric("Silueta promedio", f"{sil.mean():.3f}")
    st.caption("Esta sección se recalcula con la configuración elegida en la barra lateral.")

    st.subheader("1. ¿Qué variables definen los componentes?")
    kept = loadings.iloc[:, :n_comp].rename(index=L)
    for cp in kept.columns:
        topv = kept[cp].abs().sort_values(ascending=False).head(2).index
        st.markdown(f"- **{cp}**: " + " y ".join(f"{v} ({kept.loc[v, cp]:+.2f})" for v in topv))
    st.markdown(
        f"""
Las variables se organizan en parejas con sentido económico: gasto con nivel socioeconómico, y edad con semestre.
El promedio académico y la salud casi no se correlacionan con las demás, por lo que necesitan componentes propios.
Se conservaron **{n_comp} componentes**, que retienen **{retained:.1f} %** de la variación original. Reducir a dos
habría dejado fuera cerca de la mitad de la información; el ACP aquí sirvió más para **entender la estructura**
que para comprimirla.
"""
    )

    st.subheader("2. ¿Qué diferencias reales aparecen entre los grupos?")
    summary = pd.DataFrame(
        {
            "% de la muestra": prof["% de la muestra"].round(1),
            "Rasgos distintivos": [an.describe_cluster(zprof.loc[g]) for g in prof.index],
            "% con depresión": dep.round(1),
        }
    )
    st.dataframe(summary, width="stretch")
    hi, lo = dep.idxmax(), dep.idxmin()
    st.markdown(
        f"""
Las diferencias entre grupos se concentran en la capacidad económica, el avance en la carrera, el rendimiento
académico y la salud. Sin haber usado la depresión para agrupar, su frecuencia varía entre segmentos: va de
**{dep[lo]:.1f} %** en el grupo {lo} a **{dep[hi]:.1f} %** en el grupo {hi}, frente a {overall:.1f} % en toda la muestra.
En la exploración, el estado de salud es la variable más asociada con la depresión, de modo que los grupos con
peor salud promedio son los que concentran más casos.
"""
    )

    st.subheader("3. ¿Son grupos claramente distinguibles?")
    st.markdown(
        f"""
**No del todo.** La silueta promedio es **{sil.mean():.3f}** y **{(sil < 0).mean():.1%}** de los estudiantes
tiene silueta negativa. Los estudiantes forman un continuo: K-Means lo parte en regiones útiles para describir,
pero las fronteras son difusas y un estudiante cerca del borde podría cambiar de grupo con otra semilla u otro K.
Los grupos deben leerse como **perfiles típicos**, no como categorías cerradas.
"""
    )

    st.subheader("4. Relación con el problema")
    st.markdown(
        """
Para una oficina de bienestar universitario, el resultado sugiere dos ideas. Primero, no existe un único
«perfil de riesgo»: la depresión aparece en todos los segmentos, así que los programas no deberían dirigirse
solo a un grupo. Segundo, los segmentos con peor salud física autorreportada o menor rendimiento académico muestran
las tasas más altas. La capacidad económica es la dimensión que más diferencia a los estudiantes entre sí, pero
su correlación directa con la depresión es casi nula: cuando un grupo de mayor gasto presenta más depresión,
lo acompaña también un peor estado de salud promedio. Las señales académicas y de salud parecen, entonces,
más informativas para priorizar acompañamiento que el nivel de ingreso.
"""
    )

    st.subheader("5. Limitaciones")
    st.markdown(
        """
- **Separación débil:** con siluetas bajas, los grupos son una descripción conveniente, no una estructura
  comprobada. No deberían usarse para clasificar a un estudiante individual.
- **Variables ordinales como números:** codificar salud (1-4) y nivel socioeconómico (1-3) asume distancias
  iguales entre categorías. Al tener pocos valores, además, tienden a «cortar» los grupos por categoría.
- **Datos autorreportados y no representativos:** la muestra no tiene diseño muestral documentado y la
  depresión no es un diagnóstico clínico.
- **Asociación, no causalidad:** que un grupo tenga más depresión no indica que sus características la causen.
- **K-Means supone grupos esféricos y de tamaño similar,** y depende de la inicialización (aquí se fija la
  semilla 42 y se usan 10 reinicios).
"""
    )
