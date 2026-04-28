"""
graficas_caso2.py

Módulo de visualización para el Caso 2: conducción de calor en placa plana.
Genera figuras para Etapa 1 (estacionario) y Etapa 2 (transitorio).

Taller 02 - Modelación Matemática
Universidad Nacional de Colombia
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors


# para magma pero bonito
colors = plt.cm.magma(np.linspace(0.05, 1, 256)) 
new_magma = mcolors.ListedColormap(colors)

def graficar_campo_temperatura(XX, YY, T, titulo, nombre_figura,
                                T_min=None, T_max=None):
    """
    Genera mapa de color 2D del campo de temperatura con isolíneas.

    Parámetros:
    XX, YY: ndarray -> Mallas de coordenadas.
    T: ndarray -> Campo de temperatura [K].
    titulo: str -> Título de la figura.
    nombre_figura: str -> Nombre para guardar.
    T_min, T_max: float -> Límites de la escala de color.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    T_min = T_min if T_min is not None else T.min()
    T_max = T_max if T_max is not None else T.max()

    # Paso 00: mapa de color relleno
    cf = ax.contourf(XX, YY, T, levels=50, cmap=new_magma,
                     vmin=T_min, vmax=T_max)

    # Paso 01: isolíneas con etiquetas
    cs = ax.contour(XX, YY, T, levels=10, colors="white",
                    linewidths=0.8, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    # Paso 02: barra de color
    cbar = fig.colorbar(cf, ax=ax, shrink=0.9)
    cbar.set_label("T [K]", fontsize=11)

    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{nombre_figura}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >> Figura guardada: {nombre_figura}.png")


def graficar_comparacion(XX, YY, T_num, T_analitica, nombre_figura):
    """
    Figura comparativa lado a lado: solución numérica vs analítica.

    Parámetros:
    XX, YY: ndarray -> Mallas de coordenadas.
    T_num: ndarray -> Campo numérico [K].
    T_analitica: ndarray -> Campo analítico [K].
    nombre_figura: str -> Nombre para guardar.
    """
    T_min = min(T_num.min(), T_analitica.min())
    T_max = max(T_num.max(), T_analitica.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    titulos = ["Solución Numérica (CFD)", "Solución Analítica", "Error absoluto |T_num - T_anal|"]
    campos  = [T_num, T_analitica, np.abs(T_num - T_analitica)]
    cmaps   = [new_magma, new_magma, "plasma"]

    for idx, (ax, campo, tit, cm) in enumerate(zip(axes, campos, titulos, cmaps)):
        if idx < 2:
            cf = ax.contourf(XX, YY, campo, levels=50, cmap=cm,
                             vmin=T_min, vmax=T_max)
            cs = ax.contour(XX, YY, campo, levels=8, colors="white",
                            linewidths=0.7, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
        else:
            cf = ax.contourf(XX, YY, campo, levels=50, cmap=cm)

        cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
        cbar.set_label("T [K]" if idx < 2 else "|ΔT| [K]", fontsize=9)
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel("y [m]", fontsize=10)
        ax.set_title(tit, fontsize=10, fontweight="bold")
        ax.set_aspect("equal")

    plt.suptitle("Comparación solución numérica vs. analítica — Etapa 1",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{nombre_figura}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >> Figura guardada: {nombre_figura}.png")


def graficar_curvas_error(dx_lista, errores_max, errores_L2, nombre_figura):
    """
    Curvas de error vs tamaño de malla en escala log-log.
    Permite estimar el orden de convergencia del esquema.

    Parámetros:
    dx_lista: list -> Valores de Δx usados
    errores_max: list -> Error máximo para cada Δx [K]
    errores_L2: list -> Error L2 para cada Δx [K]
    nombre_figura: str -> Nombre para guardar
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(dx_lista, errores_max, color="dodgerblue", marker="o", linewidth=2,
              markersize=7, label="Error máximo")
    ax.loglog(dx_lista, errores_L2,  color="hotpink", marker="o", linewidth=2,
              markersize=7, label="Error $L_2$")

    # Paso 00: línea de referencia de orden 2 para comparar
    dx_ref = np.array([min(dx_lista), max(dx_lista)])
    ref    = errores_L2[-1] * (dx_ref / dx_lista[-1])**2
    ax.loglog(dx_ref, ref, "--k", linewidth=1.2, alpha=0.7,
              label="Orden 2 (referencia)")

    ax.set_xlabel("$\\Delta x$ [m]", fontsize=11)
    ax.set_ylabel("Error [K]", fontsize=11)
    ax.set_title("Convergencia del esquema CFD — Etapa 1",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{nombre_figura}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >> Figura guardada: {nombre_figura}.png")