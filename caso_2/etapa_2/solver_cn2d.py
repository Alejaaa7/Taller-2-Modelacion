"""
solver_cn2d.py

Módulo de solución numérica de la ecuación de calor 2D transitoria con 
fuente interna, mediante el esquema de Crank-Nicolson (semi-implícito).

Ecuación gobernante:
    dT/dt = alpha * nabla^2 T + G(x,y,t)

Condiciones de frontera:
    - Borde izquierdo  (x=0): Neumann dT/dx = 0  (aislado)
    - Borde derecho    (x=a): Dirichlet T = 50*sin(omega*t) + 130 [°C]
    - Bordes inf/sup (y=0,b): Neumann dT/dy = 0  (aislados)

Esquema Crank-Nicolson:
    A * T^{n+1} = B * T^n + g^{n+1/2}

donde A y B son constantes en el tiempo -> A_inv se calcula una sola vez.
"""

import numpy as np
from scipy import linalg



# Término fuente gaussiano oscilante


def fuente_G(XX, YY, t, G0, x0, y0, sigma, omega):
    """
    Evalúa el término fuente gaussiano oscilante G(x,y,t).

    G(x,y,t) = G0 * [(r^2 + 2sigma^2) / (2π sigma^6)] * exp(-r^2/(2sigma^2)) * sin(ω t)
    donde r^2 = (x-x0)^2 + (y-y0)^2

    Parámetros:
    
    XX, YY: ndarray -> Mallas de coordenadas.
    t: float -> Tiempo actual [s].
    G0: float  -> Amplitud de la fuente.
    x0, y0: float -> Centro de la gaussiana [m].
    sigma: float -> Ancho característico [m].
    omega: float -> Frecuencia de oscilación [rad/s].

    Retorna:
    
    G: ndarray -> Campo de la fuente en el instante t.
    """
    r2    = (XX - x0)**2 + (YY - y0)**2
    espacial = G0 * (r2 + 2 * sigma) / (2 * np.pi * sigma**6) * \
               np.exp(-r2 / (2 * sigma**2))
    G = espacial * np.sin(omega * t)
    return G



# Construcción del sistema Crank-Nicolson


def construir_matrices_cn(Nx, Ny, dx, dy, alpha, dt):
    """
    Construye las matrices A (implícita) y B (explícita) del esquema
    Crank-Nicolson para la ecuación de calor 2D.

    Condiciones de frontera:
      - Neumann en x=0, y=0, y=b -> nodos fantasma: T_{-1} = T_{1}
      - Dirichlet en x=a -> se incorpora en el vector g en cada paso

    Convención de índice global: k = j*Nx + i
    (incluye TODOS los nodos, las fronteras Dirichlet se manejan en g)

    Parámetros:
    
    Nx, Ny: int -> Nodos totales en x y en y.
    dx, dy: float -> Pasos espaciales [m].
    alpha: float -> Difusividad térmica [m^2/s].
    dt: float -> Paso temporal [s].

    Retorna:
    
    A: ndarray (N,N) -> Matriz implícita.
    B: ndarray (N,N) -> Matriz explícita.
    A_inv: ndarray (N,N) -> Inversa de A (calculada una sola vez).
    """
    N  = Nx * Ny
    rx = alpha * dt / (2.0 * dx**2)
    ry = alpha * dt / (2.0 * dy**2)

    A = np.zeros((N, N))
    B = np.zeros((N, N))

    for j in range(Ny):
        for i in range(Nx):
            k = j * Nx + i

            # Borde derecho: Dirichlet (se maneja en g, no en A/B)
            if i == Nx - 1:
                A[k, k] = 1.0
                # B queda en cero -> este nodo no depende del pasado
                continue

            # Coeficientes base (nodo interior)
            A[k, k] =  1.0 + 2*rx + 2*ry
            B[k, k] =  1.0 - 2*rx - 2*ry

            # Vecino izquierdo (i-1)
            if i > 0:
                A[k, k-1] = -rx
                B[k, k-1] =  rx
            else:
                # Neumann x=0: nodo fantasma T_{-1} = T_{1} -> coef x2
                A[k, k+1] += -rx    # suma al vecino derecho
                B[k, k+1] +=  rx

            # Vecino derecho (i+1)
            if i < Nx - 2:
                A[k, k+1] += -rx
                B[k, k+1] +=  rx
            else:
                # Vecino es borde Dirichlet -> pasa a g en cada paso
                pass

            # Vecino inferior (j-1)
            if j > 0:
                A[k, k-Nx] = -ry
                B[k, k-Nx] =  ry
            else:
                # Neumann y=0: nodo fantasma T_{j-1} = T_{j+1} -> coef x2
                A[k, k+Nx] += -ry
                B[k, k+Nx] +=  ry

            # Vecino superior (j+1)
            if j < Ny - 1:
                A[k, k+Nx] += -ry
                B[k, k+Nx] +=  ry
            else:
                # Neumann y=b: nodo fantasma T_{j+1} = T_{j-1} -> coef x2
                A[k, k-Nx] += -ry
                B[k, k-Nx] +=  ry

    # Calcular inversa de A una sola vez
    A_inv = linalg.inv(A)

    return A, B, A_inv


def vector_g(T_flat, Nx, Ny, dx, alpha, dt, omega, t_medio, G_medio):
    """
    Construye el vector de términos independientes g^{n+1/2} en cada
    paso de tiempo. Incluye la condición Dirichlet del borde derecho y el
    término fuente.

    Parámetros:
    
    T_flat: ndarray -> Vector de temperatura actual (aplanado).
    Nx, Ny: int -> Nodos totales.
    dx: float -> Paso espacial en x [m].
    alpha: float -> Difusividad térmica [m^2/s].
    dt: float -> Paso temporal [s].
    omega: float -> Frecuencia [rad/s].
    t_medio: float -> Tiempo en el punto medio del intervalo [s].
    G_medio: ndarray -> Término fuente en t_medio (aplanado).

    Retorna:
    
    g: ndarray -> Vector de términos independientes.
    """
    N  = Nx * Ny
    rx = alpha * dt / (2.0 * dx**2)
    g  = np.zeros(N)

    # Paso 00: término fuente en el punto medio
    g += dt * G_medio

    # Paso 01: condición Dirichlet borde derecho en t^{n+1}
    T_der_nuevo = 50.0 * np.sin(omega * t_medio) + 130.0
    for j in range(Ny):
        k     = j * Nx + (Nx - 2)   # nodo adyacente al borde derecho
        k_der = j * Nx + (Nx - 1)   # nodo del borde derecho
        g[k]      += rx * T_der_nuevo   # contribución implícita
        g[k_der]   = T_der_nuevo        # imponer valor en borde

    return g



# Integrador temporal completo


def resolver_transitorio(x, y, XX, YY, Nx, Ny, dx, dy,
                          alpha, dt, omega,
                          G0, x0, y0, sigma,
                          T_inicial, sensores):
    """
    Resuelve la ecuación de calor 2D transitoria con Crank-Nicolson.

    Parámetros:
    
    x, y: ndarray -> Vectores nodales.
    XX, YY: ndarray -> Mallas 2D.
    Nx, Ny: int -> Número de nodos.
    dx, dy: float -> Pasos espaciales [m].
    alpha: float -> Difusividad térmica [m^2/s].
    dt: float -> Paso temporal [s].
    omega: float -> Frecuencia de oscilación [rad/s].
    G0, x0, y0, sigma: float -> Parámetros del término fuente.
    T_inicial: ndarray -> Campo inicial T(x,y,0) [°C].
    sensores: list[tuple] -> Lista de (ix, iy) de nodos sensor.

    Retorna:
    
    t_hist: ndarray -> Vector de tiempos guardados.
    T_snapshots: list[ndarray] -> Campos T en instantes clave.
    T_sensores: ndarray -> Evolución temporal en sensores.
    T_prom_neu: ndarray -> Temperatura promedio en fronteras Neumann.
    """
    # Paso 00: tiempo final = 4pi/omega
    t_final = 4.0 * np.pi / omega
    t_vec = np.arange(0, t_final + dt, dt)
    Nt = len(t_vec)

    # Paso 01: construir matrices (solo una vez)
    print(f"   Construyendo matrices CN ({Nx*Ny} x {Nx*Ny})...")
    _, B, A_inv = construir_matrices_cn(Nx, Ny, dx, dy, alpha, dt)

    # Paso 02: inicializar campo
    T = T_inicial.copy()
    T_flat = T.flatten()

    # Paso 03: estructuras para guardar resultados
    n_sens      = len(sensores)
    T_sensores  = np.zeros((n_sens, Nt))
    T_prom_neu  = np.zeros(Nt)
    T_snapshots = []
    t_guardar   = np.linspace(0, t_final, 9).tolist()
    t_hist      = []

    # Guardar t = 0
    for s, (ix, iy) in enumerate(sensores):
        T_sensores[s, 0] = T[iy, ix]
    T_prom_neu[0] = _temp_prom_neumann(T, Nx, Ny)
    T_snapshots.append(T.copy())
    t_hist.append(0.0)

    # Paso 04: bucle temporal
    for n in range(Nt - 1):
        t_n     = t_vec[n]
        t_n1    = t_vec[n + 1]
        t_medio = 0.5 * (t_n + t_n1)

        # Término fuente en el punto medio
        G_med   = fuente_G(XX, YY, t_medio, G0, x0, y0, sigma, omega)
        G_flat  = G_med.flatten()

        # Vector g
        g = vector_g(T_flat, Nx, Ny, dx, alpha, dt,
                     omega, t_n1, G_flat)

        # Resolver: T^{n+1} = A_inv * (B * T^n + g)
        rhs    = B @ T_flat + g
        T_flat = A_inv @ rhs

        # Reconstruir matriz 2D
        T = T_flat.reshape((Ny, Nx))

        # Guardar sensores y promedio Neumann
        for s, (ix, iy) in enumerate(sensores):
            T_sensores[s, n+1] = T[iy, ix]
        T_prom_neu[n+1] = _temp_prom_neumann(T, Nx, Ny)

        # Guardar snapshots
        for tg in t_guardar:
            if tg > 0 and abs(t_n1 - tg) < dt / 2:
                T_snapshots.append(T.copy())
                t_hist.append(t_n1)

    return np.array(t_vec), t_hist, T_snapshots, T_sensores, T_prom_neu


def _temp_prom_neumann(T, Nx, Ny):
    # Calcula temperatura promedio sobre los tres bordes con Neumann.
    borde_izq = T[:, 0]
    borde_inf = T[0, :]
    borde_sup = T[-1, :]
    todos     = np.concatenate([borde_izq, borde_inf, borde_sup])
    return np.mean(todos)