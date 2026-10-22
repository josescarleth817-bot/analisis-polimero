import io
import re
import zipfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

st.set_page_config(page_title="Análisis de polímeros", page_icon="🧪", layout="wide")
st.title("🧪 Análisis estadístico de carga de ruptura en polímeros")
st.caption("LDPE · HDPE · Bioplástico de almidón | Investigación de Estadística II")
st.info("La carga de ruptura medida con pesas escalonadas es un intervalo, no una fuerza exacta. Las pruebas inferenciales con un valor representativo son análisis exploratorios y requieren interpretación cautelosa.")


def clean_name(name):
    return re.sub(r"\s+", " ", str(name)).strip()


def load_file(file):
    raw = io.BytesIO(file.getvalue())
    if file.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(raw, sep=None, engine="python")
        except UnicodeDecodeError:
            raw.seek(0)
            return pd.read_csv(raw, sep=None, engine="python", encoding="latin-1")
    xl = pd.ExcelFile(raw)
    sheet = "03_Export_Python" if "03_Export_Python" in xl.sheet_names else xl.sheet_names[0]
    if sheet != "03_Export_Python":
        st.warning(f"No se encontró '03_Export_Python'; se utilizó la hoja '{sheet}'.")
    return pd.read_excel(xl, sheet_name=sheet)


def extract_wide(df, cols):
    out = []
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        out.append((clean_name(col), vals))
    return out


def extract_long(df, group_col, value_col):
    tmp = df[[group_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[group_col, value_col])
    return [(str(name), grp[value_col].to_numpy(dtype=float)) for name, grp in tmp.groupby(group_col, sort=False)]


def descriptives(groups):
    rows = []
    for name, a in groups:
        n = len(a)
        sd = float(np.std(a, ddof=1)) if n > 1 else np.nan
        mean = float(np.mean(a)) if n else np.nan
        rows.append({"Material": name, "n": n, "Media (N)": mean, "Mediana (N)": np.median(a) if n else np.nan,
                     "Desv. estándar (N)": sd, "Varianza (N²)": sd**2,
                     "CV (%)": 100 * sd / abs(mean) if mean and np.isfinite(sd) else np.nan,
                     "Mínimo (N)": np.min(a) if n else np.nan, "Q1 (N)": np.percentile(a, 25) if n else np.nan,
                     "Q3 (N)": np.percentile(a, 75) if n else np.nan, "Máximo (N)": np.max(a) if n else np.nan})
    return pd.DataFrame(rows)


def dunn_holm(groups):
    # Dunn test with tie correction and Holm familywise adjustment.
    names = [g[0] for g in groups]
    vals = np.concatenate([g[1] for g in groups])
    ns = np.array([len(g[1]) for g in groups])
    ranks = stats.rankdata(vals)
    _, counts = np.unique(vals, return_counts=True)
    N = len(vals)
    tie_term = np.sum(counts**3 - counts) / (12 * (N - 1)) if N > 1 else 0
    base = N * (N + 1) / 12 - tie_term
    means = []
    pos = 0
    for n in ns:
        means.append(np.mean(ranks[pos:pos+n])); pos += n
    pairs = []
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            denom = np.sqrt(base * (1/ns[i] + 1/ns[j]))
            z = (means[i] - means[j]) / denom if denom > 0 else np.nan
            p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
            pairs.append((names[i], names[j], z, p))
    valid = [(i, row[3]) for i, row in enumerate(pairs) if np.isfinite(row[3])]
    adjusted = [np.nan] * len(pairs)
    last = 0
    for rank, (idx, p) in enumerate(sorted(valid, key=lambda v: v[1])):
        last = max(last, min(1.0, p * (len(valid) - rank)))
        adjusted[idx] = last
    return pd.DataFrame([{"Grupo A": a, "Grupo B": b, "z": z, "p sin ajuste": p, "p Holm": adjusted[i]}
                         for i, (a,b,z,p) in enumerate(pairs)])


def games_howell(groups):
    rows = []
    k = len(groups)
    for i in range(k):
        for j in range(i+1, k):
            a, b = groups[i][1], groups[j][1]
            n1,n2 = len(a),len(b)
            v1,v2 = np.var(a,ddof=1)/n1,np.var(b,ddof=1)/n2
            se2 = v1+v2
            df = se2**2/(v1**2/(n1-1)+v2**2/(n2-1)) if se2>0 else np.nan
            diff = np.mean(a)-np.mean(b)
            q = np.sqrt(2)*abs(diff)/np.sqrt(se2) if se2>0 else np.nan
            p = stats.studentized_range.sf(q,k,df) if np.isfinite(q) and np.isfinite(df) else np.nan
            rows.append({"Grupo A":groups[i][0],"Grupo B":groups[j][0],"Diferencia de medias (N)":diff,"gl Welch":df,"p ajustado":p})
    return pd.DataFrame(rows)


def fig_bytes(fig):
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=170,bbox_inches="tight");return buf.getvalue()


def add_fig(fig, name, figures):
    st.pyplot(fig)
    figures[name] = fig_bytes(fig)
    plt.close(fig)


with st.sidebar:
    st.header("Configuración")
    file = st.file_uploader("Excel o CSV", type=["xlsx","csv"])
    alpha = st.number_input("Nivel de significancia α", min_value=0.001,max_value=0.20,value=0.05,step=0.01,format="%.3f")
    st.caption("La independencia se verifica mediante el diseño experimental, no con un valor p.")

if file is None:
    st.write("Subí tu archivo para comenzar. Formatos admitidos: columnas por material o tabla larga con material y carga.")
    example = pd.DataFrame({"LDPE": [0.5,0.6,0.7],"HDPE":[1.2,1.3,1.4],"Bioplástico":[0.3,0.4,0.5]})
    st.download_button("Descargar plantilla de ejemplo (valores ilustrativos, N)", example.to_csv(index=False).encode(),"plantilla_polimeros.csv","text/csv")
    st.stop()

try:
    df=load_file(file)
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}");st.stop()
if df.empty:
    st.error("El archivo está vacío.");st.stop()

st.subheader("1. Datos y configuración de la medición")
st.dataframe(df.head(100),use_container_width=True)
format_type=st.radio("Estructura de datos",["Columnas por material (ancho)","Una columna de material y otra de medición (largo)"],horizontal=True)
if format_type.startswith("Columnas"):
    numeric_candidates=[c for c in df.columns if pd.to_numeric(df[c],errors="coerce").notna().sum()>0]
    cols=st.multiselect("Columnas de materiales",list(df.columns),default=numeric_candidates[:3])
    if len(cols)<2:st.warning("Seleccioná al menos dos materiales.");st.stop()
    groups=extract_wide(df,cols)
else:
    c1,c2=st.columns(2)
    group_col=c1.selectbox("Columna de material",list(df.columns))
    value_col=c2.selectbox("Columna de medición",list(df.columns),index=min(1,len(df.columns)-1))
    if group_col==value_col:st.error("Seleccioná columnas distintas.");st.stop()
    groups=extract_long(df,group_col,value_col)

mode=st.selectbox("¿Qué contienen los valores importados?",["Fuerza en newtons (N)","Masa total suspendida en gramos (g)","Masa añadida en gramos (g), sin portapesas","Intervalos de masa: último valor soportado y primer valor que rompe (formato largo)"])
if mode.startswith("Intervalos"):
    if not format_type.startswith("Una columna"):
        st.error("Para intervalos utilizá formato largo: material, última masa soportada y primera masa que rompe.");st.stop()
    c1,c2=st.columns(2)
    low_col=c1.selectbox("Última masa total soportada (g)",list(df.columns),index=0)
    high_col=c2.selectbox("Primera masa total que produjo ruptura (g)",list(df.columns),index=min(2,len(df.columns)-1))
    if len({group_col,low_col,high_col})<3:st.error("Las tres columnas deben ser diferentes.");st.stop()
    tmp=df[[group_col,low_col,high_col]].copy()
    for col in (low_col,high_col):tmp[col]=pd.to_numeric(tmp[col],errors="coerce")
    tmp=tmp.dropna()
    if (tmp[high_col]<=tmp[low_col]).any() or (tmp[[low_col,high_col]]<0).any():
        st.error("Cada límite superior debe ser mayor que el inferior y las masas no pueden ser negativas.");st.stop()
    st.warning("Se usa el punto medio de cada intervalo solo como aproximación exploratoria; no es una medición exacta de la fuerza de ruptura. Los resultados pueden cambiar al utilizar otros valores dentro de los intervalos.")
    tmp["Fuerza inferior (N)"]=tmp[low_col]*9.81/1000
    tmp["Fuerza superior (N)"]=tmp[high_col]*9.81/1000
    tmp["Estimación punto medio (N)"]=(tmp["Fuerza inferior (N)"]+tmp["Fuerza superior (N)"])/2
    st.dataframe(tmp,use_container_width=True)
    groups=[(str(name),sub["Estimación punto medio (N)"].to_numpy()) for name,sub in tmp.groupby(group_col,sort=False)]
else:
    if mode=="Masa añadida en gramos (g), sin portapesas":
        tare=st.number_input("Masa del portapesas y elementos suspendidos adicionales (g)",min_value=0.0,value=0.0,step=1.0)
        groups=[(n,(a+tare)*9.81/1000) for n,a in groups]
    elif mode=="Masa total suspendida en gramos (g)":
        groups=[(n,a*9.81/1000) for n,a in groups]

if len(groups)<2 or any(len(a)<3 for _,a in groups):
    st.error("Se requieren al menos dos grupos y tres observaciones válidas por grupo para este flujo.");st.stop()
if any(not np.isfinite(a).all() or (a<0).any() for _,a in groups):
    st.error("Hay mediciones negativas o no finitas.");st.stop()
if len(set(n for n,_ in groups))!=len(groups):
    st.error("Los nombres de los grupos deben ser únicos.");st.stop()

n_total=sum(len(a) for _,a in groups)
if n_total!=30 or len(groups)!=3 or any(len(a)!=10 for _,a in groups):
    st.warning(f"Diseño cargado: {len(groups)} grupos, {n_total} observaciones; el protocolo previsto es 3 grupos de 10.")

st.subheader("2. Estadística descriptiva")
summary=descriptives(groups)
st.dataframe(summary.style.format(precision=4),use_container_width=True)
figures={}

st.subheader("3. Visualización")
fig,ax=plt.subplots(figsize=(8,4.5))
ax.boxplot([a for _,a in groups],tick_labels=[n for n,_ in groups],showmeans=True)
ax.set_ylabel("Fuerza de ruptura estimada (N)");ax.grid(axis="y",alpha=.25)
add_fig(fig,"boxplot.png",figures)

fig,axes=plt.subplots(1,len(groups),figsize=(max(9,4*len(groups)),3.5),squeeze=False)
for ax,(name,a) in zip(axes[0],groups):
    ax.hist(a,bins="auto",edgecolor="white");ax.set_title(name);ax.set_xlabel("Fuerza (N)");ax.set_ylabel("Frecuencia")
fig.tight_layout();add_fig(fig,"histogramas.png",figures)

residuals=np.concatenate([a-np.mean(a) for _,a in groups])
fig,ax=plt.subplots(figsize=(6,4.5));stats.probplot(residuals,dist="norm",plot=ax);ax.set_title("Gráfico Q-Q de residuos")
add_fig(fig,"qq_residuos.png",figures)

st.subheader("4. Comprobación de supuestos")
st.caption("La independencia debe sustentarse en la selección de probetas, el orden aleatorio y el control del montaje.")
results=[]
def record(name,stat,p):
    results.append({"Prueba":name,"Estadístico":float(stat),"p":float(p)})
    st.write(f"**{name}:** estadístico = {stat:.5g}; p = {p:.5g}")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        sh_stat,sh_p=stats.shapiro(residuals)
        record("Shapiro-Wilk (residuos)",sh_stat,sh_p)
    except Exception as e:sh_p=np.nan;st.warning(f"Shapiro-Wilk no disponible: {e}")
    try:
        lev_stat,lev_p=stats.levene(*[a for _,a in groups],center="median")
        record("Levene (mediana)",lev_stat,lev_p)
    except Exception as e:lev_p=np.nan;st.warning(f"Levene no disponible: {e}")
    try:
        bart_stat,bart_p=stats.bartlett(*[a for _,a in groups])
        record("Bartlett",bart_stat,bart_p)
    except Exception as e:bart_p=np.nan;st.warning(f"Bartlett no disponible: {e}")
    for w in caught:
        st.warning(f"Advertencia numérica: {w.message}")

st.caption("p > α no demuestra que el supuesto sea verdadero; los gráficos y el diseño también deben revisarse. Bartlett es sensible a la falta de normalidad.")

st.subheader("5. Contrastes inferenciales")
all_a=[a for _,a in groups]
labels=np.concatenate([[name]*len(a) for name,a in groups])
values=np.concatenate(all_a)
valid_variance=all(len(a)>1 and np.var(a,ddof=1)>0 for a in all_a)
if not valid_variance:
    st.error("Algún grupo tiene varianza cero. No se puede aplicar de forma fiable el flujo automático habitual.");st.stop()

normal_ok=np.isfinite(sh_p) and sh_p>alpha
variance_ok=np.isfinite(lev_p) and lev_p>alpha
suggestion = (
    "ANOVA clásico"
    if normal_ok and variance_ok
    else "ANOVA de Welch"
    if normal_ok
    else "Kruskal-Wallis"
)
st.info(f"Ruta orientativa: **{suggestion}**. No es una decisión automática definitiva; considerá diseño, Q-Q, tamaños de muestra y naturaleza de la variable.")

method=st.selectbox("Prueba principal para interpretar",["ANOVA clásico","ANOVA de Welch","Kruskal-Wallis"],index=["ANOVA clásico","ANOVA de Welch","Kruskal-Wallis"].index(suggestion))
posthoc=None
try:
    if method=="ANOVA clásico":
        stat,p=stats.f_oneway(*all_a)
        record("ANOVA clásico F",stat,p)
        df1=len(groups)-1;df2=n_total-len(groups)
        grand=np.mean(values)
        ss_between=sum(len(a)*(np.mean(a)-grand)**2 for a in all_a)
        ss_total=np.sum((values-grand)**2)
        st.write(f"Grados de libertad: ({df1}, {df2}); η² = {ss_between/ss_total:.4f}" if ss_total>0 else "η² no definido")
        if p<alpha:
            tukey=pairwise_tukeyhsd(values,labels,alpha=alpha)
            posthoc=pd.DataFrame(tukey._results_table.data[1:],columns=tukey._results_table.data[0])
            st.write("**Tukey HSD:**")
    elif method=="ANOVA de Welch":
        w=anova_oneway(all_a,use_var="unequal",welch_correction=True)
        stat,p=float(w.statistic),float(w.pvalue)
        record("ANOVA de Welch F",stat,p)
        st.write(f"Grados de libertad: ({w.df[0]:.3f}, {w.df[1]:.3f})")
        if p<alpha:
            posthoc=games_howell(groups)
            st.write("**Games-Howell:**")
    else:
        stat,p=stats.kruskal(*all_a)
        record("Kruskal-Wallis H",stat,p)
        st.write("Kruskal-Wallis compara distribuciones; solo bajo condiciones adicionales puede interpretarse como comparación de medianas.")
        if p<alpha:
            posthoc=dunn_holm(groups)
            st.write("**Dunn con ajuste Holm:**")
    if posthoc is not None:st.dataframe(posthoc,use_container_width=True)
    st.success("Se rechaza H₀ al nivel α especificado: hay evidencia estadística de diferencias entre grupos." if p<alpha else "No se rechaza H₀: no hay evidencia estadística suficiente para afirmar diferencias entre grupos. Esto no demuestra igualdad.")
except Exception as e:
    st.error(f"No se pudo completar la prueba: {e}")

st.subheader("6. Exportar resultados")
result_df=pd.DataFrame(results)
output=io.BytesIO()
with pd.ExcelWriter(output,engine="openpyxl") as writer:
    summary.to_excel(writer,sheet_name="Descriptivos",index=False)
    result_df.to_excel(writer,sheet_name="Pruebas",index=False)
    pd.DataFrame({"Material":labels,"Fuerza estimada (N)":values}).to_excel(writer,sheet_name="Datos_analizados",index=False)
    if posthoc is not None:posthoc.to_excel(writer,sheet_name="Post_hoc",index=False)
st.download_button("📥 Descargar tablas Excel",output.getvalue(),"resultados_polimeros.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
zipbuf=io.BytesIO()
with zipfile.ZipFile(zipbuf,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for name,data in figures.items():z.writestr(name,data)
st.download_button("📥 Descargar gráficos PNG (ZIP)",zipbuf.getvalue(),"graficos_polimeros.zip",mime="application/zip")
st.caption("Informe exploratorio. Registrar masas verificadas, condiciones de ensayo, geometría y lugar de ruptura; no equiparar el montaje a un ensayo normalizado.")
