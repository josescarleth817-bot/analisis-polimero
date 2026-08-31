import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib


st.title(" Análisis Estadístico de Resistencia a la Tensión en Polímeros")
st.write("Aplicación para el procesamiento de datos del ensayo de tracción (DCA).")

st.sidebar.header("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo de Excel o CSV", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Lectura de datos
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, sheet_name="03_Export_Python")

    st.subheader(" Datos Recolectados")
    st.dataframe(df)

    # Estadísticos descriptivos
    st.subheader(" Estadísticos Descriptivos")
    resumen = df.describe().T[['mean', 'std', 'min', 'max']]
    resumen['variance'] = df.var()
    resumen.columns = ['Media (g)', 'Desv. Estándar (g)', 'Mínimo (g)', 'Máximo (g)', 'Varianza (g²)']
    st.dataframe(resumen)

    # Gráfico de Boxplot sin advertencias
    st.subheader(" Diagrama de Cajas (Boxplot)")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([df[col].dropna() for col in df.columns], tick_labels=df.columns)
    ax.set_ylabel("Carga Total de Ruptura (g)")
    ax.set_title("Comparativa de Resistencia Mecánica")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # Pruebas de Hipótesis
    st.subheader(" Pruebas de Hipótesis")
    list_data = [df[col].dropna() for col in df.columns]

    stat_b, p_val_b = stats.bartlett(*list_data)
    st.markdown(r"**Prueba de Bartlett (Homogeneidad de Varianzas):**")
    st.write(f"- Estadístico $K^2$: `{stat_b:.4f}`")
    st.write(f"- $p$-valor: `{p_val_b:.4f}`")

    alpha = 0.05
    if p_val_b > alpha:
        st.success(r" **RUTA PARAMÉTRICA (ANOVA):** Se cumple la homocedasticidad ($p > 0.05$).")
        stat_f, p_val_f = stats.f_oneway(*list_data)
        st.write(f"- Estadístico $F$: `{stat_f:.4f}`")
        st.write(f"- $p$-valor: `{p_val_f:.4f}`")
    else:
        st.warning(r" **RUTA NO PARAMÉTRICA (KRUSKAL-WALLIS):** Se viola la homocedasticidad ($p \le 0.05$).")
        stat_kw, p_val_kw = stats.kruskal(*list_data)
        st.write(f"- Estadístico $H$: `{stat_kw:.4f}`")
        st.write(f"- $p$-valor: `{p_val_kw:.4f}`")
else:
    st.info(" Por favor sube un archivo Excel o CSV para comenzar el análisis.")