
import sys
from pathlib import Path       
import re                      
import numpy as np             
import pandas as pd            

sys.stdout.reconfigure(encoding="utf-8")       
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

CARPETA = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SALIDA = CARPETA / "resultados"
SALIDA.mkdir(exist_ok=True)


df = pd.read_csv(CARPETA / "dataset_raw.csv", encoding="utf-8")
filas_orig, columnas_orig = df.shape       
nulos_orig = int(df.isnull().sum().sum())  
filas_duplicadas = int(df.duplicated().sum())
ids_duplicados = int(df["usuarios_id"].duplicated().sum())

print(f"Personas encuestadas (filas): {filas_orig}")
print(f"Preguntas (columnas):         {columnas_orig}")
print(f"Celdas vacias en total:       {nulos_orig:,}")
print(f"Filas duplicadas:             {filas_duplicadas}")

print(f"Identificadores (usuarios_id) repetidos: {ids_duplicados}")

cols_texto = df.select_dtypes(exclude="number").columns.tolist()
print(f"Columnas de texto: {len(cols_texto)} de {columnas_orig}")
print(df["_38_c_mo_califica_su_desempe"].head(3).tolist())

for col in cols_texto:
    df[col] = df[col].str.strip()
print("2.1 Espacios sobrantes eliminados en todas las columnas de texto.")

def quitar_repetidos(valor):
    if not isinstance(valor, str) or "," not in valor:
        return valor
    partes = [p.strip() for p in valor.split(",")]
    vistas = []
    for p in partes:
        if p not in vistas:
            vistas.append(p)
    return ", ".join(vistas)


celdas_corregidas = 0
for col in cols_texto:
    nueva = df[col].apply(quitar_repetidos)
    celdas_corregidas += int(((nueva != df[col]) & df[col].notna()).sum())   
    df[col] = nueva
print(f"2.2 Celdas con opciones repetidas corregidas: {celdas_corregidas}")

df = df.drop_duplicates()
print(f"2.3 Filas duplicadas eliminadas: {filas_duplicadas}  (quedan {len(df)} filas)")

pct_nulos = df.isnull().mean() * 100
cols_no_aplica = [c for c in pct_nulos[pct_nulos > 40].index if c != "usuarios_id"]
cols_no_reportado = pct_nulos[(pct_nulos > 0) & (pct_nulos <= 40)].index.tolist()
cols_sin_nulos = int((pct_nulos == 0).sum())

for col in cols_no_aplica:
    df[col] = df[col].fillna("No aplica")
for col in cols_no_reportado:
    df[col] = df[col].fillna("No reportado")

print(f"2.4 Columnas 'No aplica' (>40 % vacias):     {len(cols_no_aplica)}")
print(f"    Columnas 'No reportado' (0-40 % vacias): {len(cols_no_reportado)}")
print(f"    Columnas sin ningun vacio:               {cols_sin_nulos}")
print(f"    Nulos restantes en el dataset:           {int(df.isnull().sum().sum())}")

df.to_csv(SALIDA / "dataset_clean.csv", index=False, encoding="utf-8")  
print(f"Dataset limpio: {df.shape[0]} filas x {df.shape[1]} columnas")
resumen_columnas = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "nulos_restantes": df.isnull().sum(),
    "valores_unicos": df.nunique(),
})
resumen_columnas.to_csv(SALIDA / "resumen_columnas.csv", encoding="utf-8")
print("Guardado en resultados/dataset_clean.csv y resultados/resumen_columnas.csv")


columnas_utiles = {
    "usuarios_id": "usuarios_id",
    "_12_sexo": "sexo",
    "_13_a_qu_grupo_poblacional": "grupo_poblacional",
    "_15_qu_nivel_de_perdida": "nivel_perdida_auditiva",
    "_20_seleccione_el_estrato": "estrato",
    "_21_cu_l_fue_su_ltimo_nivel": "nivel_educativo",
    "_35_a_qu_edad_adquiri_o": "edad_adquisicion_lsc",
    "_36_a_qu_edad_adquiri_o": "edad_adquisicion_oral",
    "_37_a_qu_edad_adquiri_o": "edad_adquisicion_escrito",
    "_38_c_mo_califica_su_desempe": "desempeno_lsc",
    "_39_c_mo_califica_su_desempe": "desempeno_oral",
    "_40_c_mo_califica_su_desempe": "desempeno_escrito",
    "_92_con_qu_frecuencia_se": "frecuencia_uso_1",
    "_93_con_qu_frecuencia_se": "frecuencia_uso_2",
    "_94_con_qu_frecuencia_se": "frecuencia_uso_3",
    "_95_con_qu_frecuencia_se": "frecuencia_uso_4",
}
datos = df[list(columnas_utiles.keys())].rename(columns=columnas_utiles)
print(f"3.1 Variables seleccionadas: {datos.shape[1]} (de {df.shape[1]} columnas)")

ORDINALES = [
    "nivel_perdida_auditiva", "nivel_educativo",
    "edad_adquisicion_lsc", "edad_adquisicion_oral", "edad_adquisicion_escrito",
    "desempeno_lsc", "desempeno_oral", "desempeno_escrito",
    "frecuencia_uso_1", "frecuencia_uso_2", "frecuencia_uso_3", "frecuencia_uso_4",
]
NOMINALES = ["sexo", "grupo_poblacional"]
SIN_DATO = {"No reportado", "No aplica", "No sabe"}
MANUAL = {"edad_adquisicion_escrito": {"Ninguna de las anteriores": 5}} 

def primera_opcion(valor):
    if not isinstance(valor, str):
        return valor
    return valor.split(",")[0].strip()


multiples = {c: int(datos[c].apply(lambda v: isinstance(v, str) and "," in v).sum())
             for c in ORDINALES + NOMINALES + ["estrato"]}
print("3.2 Filas con mas de una opcion marcada (se toma la primera):")
for c, n in multiples.items():
    if n:
        print(f"      {c:28s} {n}")

def a_numero(valor, columna):
    v = primera_opcion(valor)
    if not isinstance(v, str) or v in SIN_DATO or "No sabe" in v:  
        return np.nan
    if v in MANUAL.get(columna, {}):
        return MANUAL[columna][v]
    m = re.match(r"^(\d+)\.", v)
    return int(m.group(1)) if m else np.nan


def a_estrato(valor):
    v = primera_opcion(valor)
    if not isinstance(v, str) or v in SIN_DATO:
        return np.nan
    try:
        return int(v)
    except ValueError:
        return np.nan


for c in ORDINALES:
    datos[c + "_num"] = datos[c].apply(lambda v, col=c: a_numero(v, col))
datos["estrato_num"] = datos["estrato"].apply(a_estrato)
NUM = [c + "_num" for c in ORDINALES] + ["estrato_num"]
print("3.3 Respuestas convertidas a numeros (columnas nuevas terminadas en _num).")

sexo = datos["sexo"].apply(primera_opcion)
print(sexo.value_counts())
print(f"Sexo sin dato: {int((sexo == 'No reportado').sum())} de {len(datos)}")

perdida = datos["nivel_perdida_auditiva"].apply(primera_opcion)
print(perdida.value_counts())
print(f"Sin dato (No reportado): {int((perdida == 'No reportado').sum())}")
print(f"'No sabe':               {int(perdida.str.contains('No sabe').sum())}")
print(f"Respuestas validas:      {int(datos['nivel_perdida_auditiva_num'].notna().sum())}")

filas = []
for c in NUM:
    s = datos[c].dropna()                       
    filas.append({
        "variable": c.replace("_num", ""),
        "n_validos": len(s),
        "n_faltantes": int(datos[c].isna().sum()),
        "media": round(s.mean(), 2),
        "mediana": s.median(),
        "moda": s.mode().iloc[0],
        "minimo": s.min(),
        "maximo": s.max(),
        "rango": s.max() - s.min(),
        "varianza": round(s.var(ddof=1), 3),
        "desv_std": round(s.std(ddof=1), 3),
    })
estadisticas = pd.DataFrame(filas)
estadisticas.to_csv(SALIDA / "estadisticas_descriptivas.csv", index=False, encoding="utf-8")

print("Escala de desempeno: 1 = Alto, 2 = Medio, 3 = Bajo, 4 = Nulo (numero menor = mejor)\n")
CLAVE = ["nivel_perdida_auditiva", "desempeno_lsc", "desempeno_oral", "desempeno_escrito"]
print("Variables clave (diapositivas 17 y 18):")
print(estadisticas[estadisticas["variable"].isin(CLAVE)].to_string(index=False))
print("\nTodas las variables:")
print(estadisticas.to_string(index=False))
print()
for c in NOMINALES:
    conteo = datos[c].apply(primera_opcion).value_counts()
    pct = (conteo / conteo.sum() * 100).round(1)
    print(f"--- {c} ---")
    for cat, n in conteo.items():
        print(f"   {cat}: {n} ({pct[cat]} %)")
    print(f"   Moda: {conteo.idxmax()}\n")
print("Distribucion del desempeno (% de respuestas validas):")
etiquetas = {1: "Alto", 2: "Medio", 3: "Bajo", 4: "Nulo"}
for c in ["desempeno_lsc", "desempeno_oral", "desempeno_escrito"]:
    dist = (datos[c + "_num"].value_counts(normalize=True).sort_index() * 100).round(1)
    print(f"   {c:18s} " + " | ".join(f"{etiquetas[int(k)]} {v}%" for k, v in dist.items()))

print("\nValores atipicos (IQR = Q3 - Q1; atipico si esta a mas de 1.5*IQR de los cuartiles):")
for c in NUM:
    s = datos[c].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    atipicos = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
    if len(atipicos):
        print(f"   {c.replace('_num', ''):28s} {len(atipicos):4d} atipicos (valor {sorted(atipicos.unique())})")


print("5.1 Sexo vs nivel de perdida auditiva (% dentro de cada sexo):")
validos = datos["nivel_perdida_auditiva_num"].notna()
tabla_sexo = pd.crosstab(sexo[validos], perdida[validos], normalize="index").round(3) * 100
print(tabla_sexo.to_string())

print("\n5.2 Desempeno PROMEDIO por nivel de perdida (1 leve ... 4 profunda; menor = mejor):")
por_perdida = datos.groupby("nivel_perdida_auditiva_num")[
    ["desempeno_lsc_num", "desempeno_oral_num", "desempeno_escrito_num"]].mean().round(2)
por_perdida.index = ["1. Leve", "2. Moderada", "3. Severa", "4. Profunda"]
por_perdida.columns = ["LSC", "Oral", "Escrito"]
print(por_perdida.to_string())

print("\n    % que califica su desempeno en LSC como 'Alto', por nivel de perdida:")
lsc_alto = (datos[validos & datos["desempeno_lsc_num"].notna()]
            .groupby("nivel_perdida_auditiva_num")["desempeno_lsc_num"]
            .apply(lambda s: (s == 1).mean() * 100).round(1))
lsc_alto.index = por_perdida.index
print(lsc_alto.to_string())

print("\n    % que califica su desempeno ORAL como 'Nulo', por nivel de perdida:")
oral_nulo = (datos[validos & datos["desempeno_oral_num"].notna()]
             .groupby("nivel_perdida_auditiva_num")["desempeno_oral_num"]
             .apply(lambda s: (s == 4).mean() * 100).round(1))
oral_nulo.index = por_perdida.index
print(oral_nulo.to_string())

print("\n5.3 Nivel educativo promedio por estrato:")
educ = datos.groupby("estrato_num")["nivel_educativo_num"].agg(["mean", "count"]).round(2)
educ.columns = ["nivel_educativo_promedio", "n"]
print(educ.to_string())


print("\n5.4 Correlacion de Pearson entre pares clave:")
pares = [
    ("edad_adquisicion_oral_num", "desempeno_oral_num", "Edad de adquisicion oral vs desempeno oral"),
    ("nivel_perdida_auditiva_num", "desempeno_oral_num", "Nivel de perdida vs desempeno oral"),
    ("edad_adquisicion_lsc_num", "desempeno_lsc_num", "Edad de adquisicion LSC vs desempeno LSC"),
    ("estrato_num", "nivel_educativo_num", "Estrato vs nivel educativo"),
]
correlaciones = []
for a, b, nombre in pares:
    r = datos[[a, b]].dropna().corr().iloc[0, 1]
    correlaciones.append({"par": nombre, "r": round(r, 3)})
    print(f"   {nombre:46s} r = {r:.3f}")
pd.DataFrame(correlaciones).to_csv(SALIDA / "correlaciones.csv", index=False, encoding="utf-8")


e = estadisticas.set_index("variable")
print(f"Dataset: {len(df)} filas x {df.shape[1]} columnas, sin nulos | {len(columnas_utiles)} variables de estudio")
print(f"Sexo: {int((sexo != 'No reportado').sum())} respuestas con dato "
      f"({(sexo == '2. Masculino').mean() * 100:.1f} % masculino, {(sexo == '1. Femenino').mean() * 100:.1f} % femenino)")
for v, nombre in [("desempeno_lsc", "LSC"), ("desempeno_oral", "Oral"), ("desempeno_escrito", "Escrito"),
                  ("nivel_perdida_auditiva", "Perdida auditiva")]:
    print(f"{nombre:17s} N={int(e.loc[v, 'n_validos'])}  media={e.loc[v, 'media']}  mediana={e.loc[v, 'mediana']}  "
          f"moda={e.loc[v, 'moda']}  varianza={e.loc[v, 'varianza']}  desv={e.loc[v, 'desv_std']}  rango={e.loc[v, 'rango']}")
print(f"Escrito nivel medio: {(datos['desempeno_escrito_num'] == 2).sum() / datos['desempeno_escrito_num'].notna().sum() * 100:.1f} % "
      f"de las respuestas validas")
print("\nArchivos generados en la carpeta 'resultados':")
for f in sorted(SALIDA.iterdir()):
    print("   -", f.name)
