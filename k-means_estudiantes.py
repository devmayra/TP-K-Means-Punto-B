"""
TP 9 — K-Means, Punto B: Segmentación de Estudiantes Universitarios
Agrupa estudiantes en 4 segmentos según edad, horas de estudio semanales
y promedio académico.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------------------------------------------------------
# Datos de la consigna
# ------------------------------------------------------------
estudiantes = [
    {'edad': 42, 'horas_estudio': 2, 'promedio_academico': 8.87},
    {'edad': 26, 'horas_estudio': 8, 'promedio_academico': 9.11},
    {'edad': 56, 'horas_estudio': 5, 'promedio_academico': 7.29},
    {'edad': 29, 'horas_estudio': 14, 'promedio_academico': 7.27},
    {'edad': 43, 'horas_estudio': 8, 'promedio_academico': 8.5},
    {'edad': 34, 'horas_estudio': 10, 'promedio_academico': 4.45},
    {'edad': 51, 'horas_estudio': 3, 'promedio_academico': 6.12},
    {'edad': 22, 'horas_estudio': 12, 'promedio_academico': 4.33},
    {'edad': 47, 'horas_estudio': 6, 'promedio_academico': 7.0},
    {'edad': 38, 'horas_estudio': 9, 'promedio_academico': 4.75},
    {'edad': 60, 'horas_estudio': 2, 'promedio_academico': 9.0},
    {'edad': 31, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 45, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 3.2},
    {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 36, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 40, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 24, 'horas_estudio': 15, 'promedio_academico': 3.0},
    {'edad': 55, 'horas_estudio': 3, 'promedio_academico': 9.0},
    {'edad': 33, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 48, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 28, 'horas_estudio': 12, 'promedio_academico': 2.0},
    {'edad': 52, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 46, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 35, 'horas_estudio': 9, 'promedio_academico': 4.0},
    {'edad': 41, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 23, 'horas_estudio': 14, 'promedio_academico': 1.0},
    {'edad': 53, 'horas_estudio': 3, 'promedio_academico': 8.0},
    {'edad': 32, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 49, 'horas_estudio': 4, 'promedio_academico': 8.0},
    {'edad': 25, 'horas_estudio': 13, 'promedio_academico': 2.0},
    {'edad': 54, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 37, 'horas_estudio': 8, 'promedio_academico': 4.0},
    {'edad': 39, 'horas_estudio': 7, 'promedio_academico': 5.0},
    {'edad': 21, 'horas_estudio': 15, 'promedio_academico': 9.0},
    {'edad': 58, 'horas_estudio': 2, 'promedio_academico': 9.0},
    {'edad': 44, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 29, 'horas_estudio': 12, 'promedio_academico': 2.0},
    {'edad': 50, 'horas_estudio': 5, 'promedio_academico': 7.0},
    {'edad': 34, 'horas_estudio': 9, 'promedio_academico': 3.0},
    {'edad': 42, 'horas_estudio': 7, 'promedio_academico': 6.0},
    {'edad': 26, 'horas_estudio': 14, 'promedio_academico': 3.0},
    {'edad': 57, 'horas_estudio': 3, 'promedio_academico': 9.0},
    {'edad': 30, 'horas_estudio': 11, 'promedio_academico': 2.0},
    {'edad': 47, 'horas_estudio': 4, 'promedio_academico': 7.0},
    {'edad': 31, 'horas_estudio': 10, 'promedio_academico': 3.0},
    {'edad': 45, 'horas_estudio': 6, 'promedio_academico': 6.0},
    {'edad': 27, 'horas_estudio': 13, 'promedio_academico': 1.0},
    {'edad': 55, 'horas_estudio': 5, 'promedio_academico': 8.0},
]

df = pd.DataFrame(estudiantes)
variables = ['edad', 'horas_estudio', 'promedio_academico']

print(f"Cantidad de estudiantes: {len(df)}\n")
print("Estadísticas de los datos originales:")
print(df[variables].describe().round(2), "\n")

# ------------------------------------------------------------
# 1. Estandarizar los datos
# ------------------------------------------------------------
# Las variables tienen escalas muy distintas (edad: 21-60, horas: 2-15,
# promedio: 1-9.11). K-Means usa distancias euclídeas, así que sin
# estandarizar la edad dominaría el agrupamiento. StandardScaler lleva
# cada variable a media 0 y desvío estándar 1.
scaler = StandardScaler()
X_std = scaler.fit_transform(df[variables])

print("Datos estandarizados (primeras 5 filas):")
print(pd.DataFrame(X_std, columns=variables).head().round(2), "\n")

# ------------------------------------------------------------
# 2. Agrupar a los estudiantes en 4 segmentos
# ------------------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_std)

# Centroides llevados de nuevo a la escala original para poder interpretarlos
centroides = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_), columns=variables
)

# Se ordenan los clusters por edad promedio para que la numeración sea estable
orden = centroides['edad'].sort_values().index
renumerar = {viejo: nuevo + 1 for nuevo, viejo in enumerate(orden)}
df['segmento'] = df['cluster'].map(renumerar)
centroides.index = centroides.index.map(renumerar)
centroides = centroides.sort_index()


# Etiqueta descriptiva automática: usa el centroide estandarizado (z-score).
# Si está a más de 0.5 desvíos por debajo de la media -> bajo,
# a más de 0.5 por encima -> alto, y si no -> medio.
def nivel(z, bajo, medio, alto):
    if z < -0.5:
        return bajo
    if z > 0.5:
        return alto
    return medio


def describir(c):
    z = scaler.transform(pd.DataFrame([c], columns=variables))[0]
    edad = nivel(z[0], "jóvenes", "edad media", "mayores")
    horas = nivel(z[1], "poco estudio", "estudio moderado", "mucho estudio")
    prom = nivel(z[2], "promedio bajo", "promedio medio", "promedio alto")
    return f"{edad.capitalize()}, {horas}, {prom}"


etiquetas = {s: f"Segmento {s}: {describir(centroides.loc[s])}" for s in centroides.index}

# ------------------------------------------------------------
# 4. Informe de resultados
# ------------------------------------------------------------
print("=" * 70)
print("INFORME DE RESULTADOS")
print("=" * 70)

print("\nCentroides (valores promedio de cada segmento, escala original):")
print(centroides.round(2))

print("\nCantidad de estudiantes por segmento:")
print(df['segmento'].value_counts().sort_index())

for s in centroides.index:
    grupo = df[df['segmento'] == s]
    print(f"\n{etiquetas[s]}  ({len(grupo)} estudiantes)")
    print(f"  Edad:     {grupo['edad'].min()}–{grupo['edad'].max()} años "
          f"(media {grupo['edad'].mean():.1f})")
    print(f"  Horas:    {grupo['horas_estudio'].min()}–{grupo['horas_estudio'].max()} hs "
          f"(media {grupo['horas_estudio'].mean():.1f})")
    print(f"  Promedio: {grupo['promedio_academico'].min()}–{grupo['promedio_academico'].max()} "
          f"(media {grupo['promedio_academico'].mean():.2f})")

print("\nCorrelaciones entre variables:")
print(df[variables].corr().round(2))

# ------------------------------------------------------------
# 3. Visualizar en 2D (edad vs promedio)
# ------------------------------------------------------------
colores = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green', 4: 'tab:red'}

plt.figure(figsize=(11, 7))
for s in centroides.index:
    grupo = df[df['segmento'] == s]
    plt.scatter(grupo['edad'], grupo['promedio_academico'], s=80,
                color=colores[s], edgecolor='black', alpha=0.8, label=etiquetas[s])
    plt.scatter(centroides.loc[s, 'edad'], centroides.loc[s, 'promedio_academico'],
                s=350, marker='X', color=colores[s], edgecolor='black', linewidth=1.5)

plt.scatter([], [], s=150, marker='X', color='gray', edgecolor='black', label='Centroide')
plt.xlabel('Edad (años)')
plt.ylabel('Promedio académico')
plt.title('Segmentación de estudiantes universitarios (K-Means, k=4)\nEdad vs Promedio académico')
plt.legend(fontsize=9, loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('grafico_estudiantes.png', dpi=120)
plt.show()