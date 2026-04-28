"""
dominio.py

Módulo de construcción del dominio 2D y solución analítica para el problema
de conducción de calor estacionario en placa plana.

La solución analítica se obtiene por separación de variables (Salsa, 2016):

    T(x, y) = (T2 - T1) * H(x, y) + T1
    H(x, y) = (2/pi) * sum_{n impar} (2/n) + sinh(n*pi*x/b) / 
              sinh(n*pi*a/b) * sin(n*pi*y/b)
"""

import numpy as np

def crear_malla(a, b, dx, dy):
    """
    Crea la malla uniforme 2D del dominio [0, a] x [0, b].

    Parámetros:
    a: float -> Longitud en x [m].
    b: float -> Longitud en y [m].
    dx: float -> Paso espacial en x [m].
    dy: float -> Paso espacial en y [m].

    Retorna:
    x: ndarray (Nx, ) -> Vector de posiciones en x.
    y: ndarray (Ny, ) -> Vector de posiciones en y.
    XX: ndarray (Ny, Nx) -> Malla en 2D en x (para graficar).
    YY: ndarray (Ny, Nx) -> Malla en 2D en y (para graficar).
    Nx, Ny: int -> Número de nodos en cada dirección.
    """

    # Paso 00: crear vectores nodales
    x = np.arange(0, a + dx, dx)
    y = np.arange(0, b + dy, dy)
    Nx = len(x)
    Ny = len(y)

    # Paso 01: crear malla 2D para graficar y evaluar solución analítica
    XX, YY = np.meshgrid(x, y)

    return x, y, XX, YY, Nx, Ny

def solucion_analitica(XX, YY, a, b, T1, T2, N_terms=101):
    """
    Evalúa la solución analítica de la ecuación de Laplace mediante la
    serie de Fourier (Salsa, 2016).
    T(x, y) = (T2 - T1) * H(x, y) + T1

    donde H(x, y) = (2/pi) * sum_{n impar} (2/n) *
                    sinh(n*pi*x/b) / sinh(n*pi*a/b) * sin(n*pi*y/b)
    
    Parámetros:
    XX: ndarray (Ny, Nx) -> Coordenadas x de la malla.
    YY: ndarray (Ny, Nx) -> Coordenadas y de la malla.
    a: float -> Longitud en x [m].
    b: float -> Longitud en y [m].
    T1: float -> Temperatura baja [K].
    T2: float -> Temperatura alta [K].
    N_terms: int -> Número de términos de la serie (solo impares).

    Retorna:
    T_analitica: ndarray (Ny, Nx) -> Campo de temperatura analítico [K].
    """

    # Paso 00: inicializar función H
    H = np.zeros_like(XX, dtype=float)

    # Paso 01: sumar términos impares de la serie de Fourier
    for n in range(1, N_terms + 1, 2):    # solo n impares: 1, 3, 5, ...
        # Argumento del seno hiperbólico (puede crecer mucho para n grandes)
        arg_num = n * np.pi * XX / b
        arg_den = n * np.pi * a / b

        # Paso 02: evitar overflow en sinh paa argumentos grandes
        # usando la identidad: sinh(x)/sinh(c) = exp(x - c) para x, c >> 1
        with np.errstate(over='ignore', invalid='ignore'):
            sinh_num = np.sinh(arg_num)
            sinh_den = np.sinh(arg_den)
            ratio = np.where(
                arg_den > 500,
                np.exp(arg_num - arg_den),   # aproximación estable
                sinh_num / sinh_den
            )

        H += (2.0/n) * ratio * np.sin(n * np.pi * YY / b)

    H *= (2.0 / np.pi)

    # Paso 03: recuperar temperatura real
    T_analitica = (T2 - T1) * H + T1

    return T_analitica
