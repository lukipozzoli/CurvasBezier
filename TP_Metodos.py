from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

print("Generando curva de Bézier cuadrática")

# Definir los puntos de control
p0 = [0, 0]
p1 = [1, 2]
p2 = [2, 0]

# Definir las funciones de Bézier f0, f1 y g
def f0(t, points):
    p0, p1 = points
    return (1 - t) * np.array(p0) + t * np.array(p1)

def f1(t, points):
    p1, p2 = points
    return (1 - t) * np.array(p1) + t * np.array(p2)

def g(t, points):
    p0, p1, p2 = points
    f0_t = f0(t, [p0, p1])
    f1_t = f1(t, [p1, p2])
    return (1 - t) * f0_t + t * f1_t

# Generar los valores de t
t_values = np.linspace(0, 1, 100)

# Calcular las curvas
f0_curve = np.array([f0(t, [p0, p1]) for t in t_values])
f1_curve = np.array([f1(t, [p1, p2]) for t in t_values])
g_curve = np.array([g(t, [p0, p1, p2]) for t in t_values])

# Graficar las curvas f0(t), f1(t) y g(t)
plt.figure(figsize=(8, 6))
plt.plot(f0_curve[:, 0], f0_curve[:, 1], label='f0(t) (entre p0 y p1)', linestyle='--', color='blue')
plt.plot(f1_curve[:, 0], f1_curve[:, 1], label='f1(t) (entre p1 y p2)', linestyle='--', color='green')
plt.plot(g_curve[:, 0], g_curve[:, 1], label='g(t) (Curva de Bézier cuadrática)', color='orange')
plt.scatter([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], c='red')
plt.text(p0[0], p0[1], 'p0', fontsize=12)
plt.text(p1[0], p1[1], 'p1', fontsize=12)
plt.text(p2[0], p2[1], 'p2', fontsize=12)
plt.title('Curva de Bézier cuadrática y sus interpolaciones')
plt.legend()
plt.grid()
plt.show()

# Explicación de la relación entre f0(t), f1(t) y g(t)
print("\n### Explicación ###")
print("La curva g(t) es una combinación de las curvas f0(t) y f1(t).")
print("f0(t) es una interpolación lineal entre p0 y p1, mientras que f1(t) es una interpolación lineal entre p1 y p2.")
print("g(t) se construye usando estas dos funciones intermedias, lo que resulta en una curva suave que se ajusta a p0, p1 y p2.")
print("Para una curva cúbica de Bézier, el proceso sería similar, utilizando tres funciones intermedias f0(t), f1(t) y f2(t), combinadas en dos pasos adicionales.")


##PUNTO 2###
print("Generando curva de Bézier cúbica")

# Definir los puntos de control
p0 = [0, 0]
p1 = [1, 3]
p2 = [3, 3]
p3 = [4, 0]

# Definir la función que calcula la curva de Bézier cúbica
def h(t, points):
    p0, p1, p2, p3 = points
    g1_t = (1 - t)**2 * np.array(p0) + 2 * t * (1 - t) * np.array(p1) + t**2 * np.array(p2)
    g2_t = (1 - t)**2 * np.array(p1) + 2 * t * (1 - t) * np.array(p2) + t**2 * np.array(p3)
    return (1 - t) * g1_t + t * g2_t

# Definir la función que calcula los coeficientes de la curva de Bézier cúbica
def coeficientes_bezier_cubica(t):
    c0 = (1 - t) ** 3
    c1 = 3 * t * (1 - t) ** 2
    c2 = 3 * t ** 2 * (1 - t)
    c3 = t ** 3
    return c0, c1, c2, c3

# Generar valores de t entre 0 y 1
t_values = np.linspace(0, 1, 100)

# Calcular la curva h(t) para la gráfica
h_curve = np.array([h(t, [p0, p1, p2, p3]) for t in t_values])

# Graficar la curva de Bézier cúbica
plt.figure(figsize=(8, 6))
plt.plot(h_curve[:, 0], h_curve[:, 1], label='h(t)')
plt.scatter([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]], c='red')
plt.text(p0[0], p0[1], 'p0')
plt.text(p1[0], p1[1], 'p1')
plt.text(p2[0], p2[1], 'p2')
plt.text(p3[0], p3[1], 'p3')
plt.title('Curva de Bézier cúbica')
plt.legend()
plt.grid()
plt.show()

# Calcular los coeficientes para cada valor de t
coef_c0, coef_c1, coef_c2, coef_c3 = [], [], [], []

for t in t_values:
    c0, c1, c2, c3 = coeficientes_bezier_cubica(t)
    coef_c0.append(c0)
    coef_c1.append(c1)
    coef_c2.append(c2)
    coef_c3.append(c3)

# Graficar los coeficientes de los puntos de control en función de t
plt.figure(figsize=(8, 6))
plt.plot(t_values, coef_c0, label='Coeficiente de p0: (1-t)^3')
plt.plot(t_values, coef_c1, label='Coeficiente de p1: 3t(1-t)^2')
plt.plot(t_values, coef_c2, label='Coeficiente de p2: 3t^2(1-t)')
plt.plot(t_values, coef_c3, label='Coeficiente de p3: t^3')
plt.title('Coeficientes de los puntos de control en función de t')
plt.xlabel('t')
plt.ylabel('Coeficientes')
plt.legend()
plt.grid()
plt.show()

# Calcular la suma de los coeficientes para t = 0.3, 0.5, 0.8
t_values_specific = [0.3, 0.5, 0.8]

print("Suma de los coeficientes para valores específicos de t:")
for t in t_values_specific:
    c0, c1, c2, c3 = coeficientes_bezier_cubica(t)
    suma_coeficientes = c0 + c1 + c2 + c3
    print(f"t = {t}: Suma de coeficientes = {suma_coeficientes:.2f}")

# Explicación de los resultados
print("\nConclusión: La suma de los coeficientes para cualquier valor de t siempre es 1, lo cual es una propiedad fundamental de las curvas de Bézier. Esto asegura que las curvas estén contenidas dentro del polígono formado por sus puntos de control.")


##PUNTO 3###


def plot_convex_hull(points):
    hull = ConvexHull(points)
    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

# Ejemplo con tres puntos en R2
points = np.array([[0, 0], [1, 2], [2, 0]])
plot_convex_hull(points)
plt.show()
