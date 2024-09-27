from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

print("Generando curva de Bézier cuadrática")

# Definir los puntos de control
p0 = np.array([0, 0]) #usamos np.array para poder hacer operaciones vectoriales con facilidad
p1 = np.array([1, 2])
p2 = np.array([2, 0])

#Funcion para ecuaciones de bezier
def bezier_f0(t, p0, p1):
    return (1-t)*p0 + t*p1

def bezier_f1(t, p1, p2):
    return (1-t)*p1 + t*p2

def bezier_g(t,f1, f2):
    return (1-t)*f1 + t*f2

# Generar la curva de Bezier para valores de t en [0, 1]
t_values = np.linspace(0, 1, 100) #nos permite generar 100 valores de t asi la curva se grafica de manera mas suave
#calculamos los puntos de la curva de bezier
bezier_curve = [bezier_g(t, bezier_f0(t,p0,p1), bezier_f1(t,p1,p2)) for t in t_values]
bezier_curve = np.array(bezier_curve)

puntos_de_control = np.array([p0, p1, p2])

#grafico
plt.figure(figsize=(8, 6))
plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], label='g(t)')
# Graficar los puntos de control y líneas entre ellos
plt.plot(puntos_de_control[:, 0], puntos_de_control[:, 1], 'ro--', label='Puntos de control', markersize=8)
# Establecer el título y etiquetas
plt.title('Curva de Bézier Cuadrática')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

# Despues lo saco para el informe, Explicación de la relación entre f0(t), f1(t) y g(t)
print("\n### Explicación ###")
print("La curva g(t) es una combinación de las curvas f0(t) y f1(t).")
print("f0(t) es una interpolación lineal entre p0 y p1, mientras que f1(t) es una interpolación lineal entre p1 y p2.")
print("g(t) se construye usando estas dos funciones intermedias, lo que resulta en una curva suave que se ajusta a p0, p1 y p2.")
print("Para una curva cúbica de Bézier, el proceso sería similar, utilizando tres funciones intermedias f0(t), f1(t) y f2(t), combinadas en dos pasos adicionales.")

##PUNTO 2###
print("Generando curva de Bézier cúbica")

# Para hacer la funcion cubica agregamos un nuevo punto de control
p0 = np.array([0, 0]) #usamos np.array para poder hacer operaciones vectoriales con facilidad
p1 = np.array([1, 3])
p2 = np.array([3, 3])
p3 = np.array([4, 0])

def bezier_cubic(t, p0, p1, p2, p3):
    return (1-t)**3 * p0 + 3 * (1-t)**2 * t * p1 + 3 * (1-t) * t**2 * p2 + t**3 * p3

# Generar la curva de Bezier para valores de t en [0, 1]
t_values = np.linspace(0, 1, 100) #nos permite generar 100 valores de t asi la curva se grafica de manera mas suave
#calculamos los puntos de la curva de bezier
bezier_curve2 = np.array([bezier_cubic(t,p0, p1, p2, p3) for t in t_values])
puntos_de_control2 = np.array([p0, p1, p2, p3])


# Graficar la curva de Bézier cúbica
plt.figure(figsize=(8, 6))
plt.plot(bezier_curve2[:, 0], bezier_curve2[:, 1],label='Curva de Bézier cúbica')
plt.scatter(puntos_de_control2[:,0], puntos_de_control2[:,1], c='red', s=80)
plt.text(p0[0], p0[1], 'p0')
plt.text(p1[0], p1[1], 'p1')
plt.text(p2[0], p2[1], 'p2')
plt.text(p3[0], p3[1], 'p3')
plt.title('Curva de Bézier cúbica')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()


# Calcular los coeficientes para cada valor de t en t_values
t_values = np.linspace(0, 1, 100)
coef_c0, coef_c1, coef_c2, coef_c3 = [], [], [], []
for t in t_values:
    c0, c1, c2, c3 = (1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3
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

# Calcular la suma de los coeficientes para valores específicos de t
t_values_specific = [0.3, 0.5, 0.8]
print("Suma de los coeficientes para valores específicos de t:")

# Calcular y mostrar la suma de los coeficientes para cada valor específico de t
for t in t_values_specific:
    c0, c1, c2, c3 = (1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3
    suma_coeficientes = c0 + c1 + c2 + c3
    print(f"t = {t}: Suma de coeficientes = {suma_coeficientes:.2f}")


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
