"""
main.py

Script principal - Caso 2, Etapa 2:
Conducción de calor 2D transitoria con fuente interna gaussiana oscilante.

Ecuación gobernante:
    dT/dt = alpha * nabla^2 T + G(x, y, t)

Método numérico: Crank-Nicolson 2D

Condiciones de frontera:
    - Neumann (x=0, y=0, y=b): dT/dn = 0
    - Dirichlet (x=a): T = 50 * sin(omega * t) + 130 [°C]

Para correr:
    python main.py
"""

import numpy as np
import sys
sys.path.append("..") # para que las encuentre fácilmente

from etapa_1.dominio import crear_malla
from solver_cn2d import resolver_transitorio
from graficas import (graficar_efecto_alpha, graficar_prom_neumann, 
                      graficar_sensores, graficar_snapshots, 
                      graficar_snapshots_saturado)


# PARTE 1: Parámetros fijos del problema

a = 1.0 # longitud en x [m]
b = 1.0 # longitud en y [m]
dx = 0.05 # paso espacial en x [m]
dy = 0.05 # paso espacial en y [m]

# Parámetros del término fuente (dados en el taller)
G0 = 2.5
x0 = 0.3
y0 = 0.7
sigma = 0.07

# Difusividad térmica base (aluminio)
alpha_base = 8.418e-5   # m²/s

# Temperatura inicial uniforme
T_ini_val = 27.0

# Sensores: ubicados FUERA de la zona de influencia de la fuente
# La fuente actúa fuertemente en x=0.30, y rn [0.55, 0.85] entonces se 
# eligen puntos en bordes y en zona no afectada
sensores_pos = [
    (0,  10),   # borde izquierdo, y=0.50
    (10,  0),   # borde inferior,  x=0.50
    (10, 20),   # borde superior,  x=0.50
    (18, 10),   # cerca borde derecho, x=0.90, y=0.50
    (10,  4),   # interior alejado de fuente, x=0.50, y=0.20
]
sensores_labels = [
    "Borde izq. (x=0.00, y=0.50)",
    "Borde inf. (x=0.50, y=0.00)",
    "Borde sup. (x=0.50, y=1.00)",
    "Interior (x=0.90, y=0.50)",
    "Interior (x=0.50, y=0.20)",
]

print("—" * 65)
print("  CASO 2 — Etapa 2: Conducción Transitoria con Fuente Interna")
print("—" * 65)
print(f"  Dominio: {a} m x {b} m  |  dx=dy={dx} m")
print(f"  Fuente: G0={G0}, centro=({x0},{y0}), sigma={sigma}")
print()



# PARTE 2: Búsqueda del valor de omega
# Condición: 27°C ≤ T_prom_Neumann(t) ≤ 50°C

print(">> Búsqueda del valor de omega adecuado...")
print("   Condición: 27°C ≤ T_prom_Neumann ≤ 50°C\n")

omega_candidatos = [0.5, 1.0, 2.0, 4.0]
dt_busqueda      = 0.05
resultados_omega = []

for omega_c in omega_candidatos:
    x, y, XX, YY, Nx, Ny = crear_malla(a, b, dx, dy)
    T0 = np.full((Ny, Nx), T_ini_val)

    t_vec, _, _, _, T_prom = resolver_transitorio(
        x, y, XX, YY, Nx, Ny, dx, dy,
        alpha_base, dt_busqueda, omega_c,
        G0, x0, y0, sigma,
        T0, sensores_pos
    )

    T_max_obs = T_prom.max()
    T_min_obs = T_prom.min()
    cumple    = (T_min_obs >= 27.0) and (T_max_obs <= 50.0)

    resultados_omega.append({
        "omega": omega_c,
        "T_max": T_max_obs,
        "T_min": T_min_obs,
        "cumple": cumple
    })

    print(f"   omega = {omega_c:.2f} rad/s  |  "
          f"T_min = {T_min_obs:.2f}°C  |  T_max = {T_max_obs:.2f}°C  |  "
          f"{'CUMPLE' if cumple else 'NO CUMPLE'}")

# Seleccionar omega
omega_ok = next((r["omega"] for r in resultados_omega if r["cumple"]), None)
if omega_ok is None:
    omega_ok = min(resultados_omega,
                   key=lambda r: max(0,r["T_max"]-50)+max(0,27-r["T_min"]))["omega"]
    print(f"\n   Ninguno cumple. Usando omega = {omega_ok:.2f} rad/s (más cercano)")
else:
    print(f"\n   >> omega seleccionado: {omega_ok:.2f} rad/s")



# PARTE 3: Simulación completa con omega seleccionado

print(f"\n>> Simulación completa con omega = {omega_ok:.2f} rad/s ...")
dt_final = 0.02

x, y, XX, YY, Nx, Ny = crear_malla(a, b, dx, dy)
T0 = np.full((Ny, Nx), T_ini_val)

t_vec, t_hist, T_snaps, T_sens, T_prom = resolver_transitorio(
    x, y, XX, YY, Nx, Ny, dx, dy,
    alpha_base, dt_final, omega_ok,
    G0, x0, y0, sigma,
    T0, sensores_pos
)

print(f"   Instantes guardados: {[f'{t:.2f}' for t in t_hist]}")

graficar_snapshots(XX, YY, T_snaps[::2], t_hist[::2], omega_ok,
                   nombre_figura="caso2_etapa2_snapshots")

tiempos_interes = [0.0, 0.79, 2.36, 3.93, 5.50]
graficar_snapshots_saturado(XX, YY, T_snaps[1::2], t_hist[1::2], omega_ok, 
                            T_sat_min=27.0, T_sat_max=180.0, 
                            nombre_figura="caso2_etapa2_snapshots_saturado")
graficar_sensores(t_vec, T_sens, sensores_labels, omega_ok,
                  nombre_figura="caso2_etapa2_sensores")
cumple, T_min_f, T_max_f = graficar_prom_neumann(
    t_vec, T_prom, omega_ok,
    nombre_figura="caso2_etapa2_prom_neumann"
)
print(f"   T_min={T_min_f:.2f}°C | T_max={T_max_f:.2f}°C | "
      f"{'CUMPLE' if cumple else 'NO CUMPLE'}")



# PARTE 4: Efecto de la difusividad térmica alpha

print("\n>> Experimento: efecto de alpha ...")

alpha_lista = [1e-5, 8.418e-5, 5e-4, 2e-3]
resultados_alpha = []

for alpha_i in alpha_lista:
    xi, yi, XXi, YYi, Nxi, Nyi = crear_malla(a, b, dx, dy)
    T0i = np.full((Nyi, Nxi), T_ini_val)
    t_vi, _, _, _, T_pi = resolver_transitorio(
        xi, yi, XXi, YYi, Nxi, Nyi, dx, dy,
        alpha_i, dt_busqueda, omega_ok,
        G0, x0, y0, sigma, T0i, sensores_pos
    )
    resultados_alpha.append({"alpha": alpha_i, "t_vec": t_vi, "T_prom": T_pi})
    print(f"   alpha = {alpha_i:.2e} m²/s  |  completado")

graficar_efecto_alpha(resultados_alpha,
                      nombre_figura="caso2_etapa2_efecto_alpha")

print()
print("—" * 65)
print("  Etapa 2 finalizada. Figuras generadas:")
print("    - caso2_etapa2_snapshots.png")
print("    - caso2_etapa2_sensores.png")
print("    - caso2_etapa2_prom_neumann.png")
print("    - caso2_etapa2_efecto_alpha.png")
print("—" * 65)