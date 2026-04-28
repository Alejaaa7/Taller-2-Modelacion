"""
graficas.py

Módulo de visualización para el Caso 1: sistema de transporte advectivo.
Genera todas las figuras requeridas.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def graficar_perfiles(x, u_hist, v_hist, t_hist, nombre_figura="perfiles_uvt"):
    """
    Genera la figura comparativa de perfiles u y v vs x para los instantes
    guardados.

    Parámetros:
    x: ndarray -> Vector de posiciones nodales.
    u_hist: list[ndarray] -> Historial de u.
    v_hist: list[ndarray] -> Historial de v.
    t_hist: list[float] -> Tiempos correspondientes.
    nombre_figura: str -> Nombre base para guardar la figura.
    """

    n_instantes = len(t_hist)
    fig, axes = plt.subplots(1, n_instantes, figsize=(14, 4), sharey=True)

    for idx, ax in enumerate(axes):
        ax.plot(x, u_hist[idx], color="hotpink", linewidth=2.0, label=f"U(x, {t_hist[idx]:.0f})")
        ax.plot(x, v_hist[idx], color="dodgerblue",  linewidth=2.0, label=f"V(x, {t_hist[idx]:.0f})")
        ax.set_title(f"$t = {t_hist[idx]:.0f}$ s", fontsize=11)
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel("u, v", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.set_xlim([x[0], x[-1]])
 
    plt.suptitle("Perfiles espaciales de $u$ y $v$ para distintos instantes de tiempo",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{nombre_figura}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >> Figura guardada: {nombre_figura}.png")
 
 
def graficar_efecto_malla(resultados_malla, t_objetivo, L):
    """
    Compara perfiles de u para distintos tamaños de malla Δx
    en un instante dado. Permite visualizar el efecto de refinamiento.
 
    Parámetros:
    resultados_malla: list[dict] -> Lista de dicts con keys: dx, x, u, v, t.
    t_objetivo: float -> Instante de tiempo a comparar [s].
    L: float -> Longitud del dominio [m].
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colores = ["#7EC9F8", "#ffa837", "#6cee6c", "#ED2D60"]
    estilos = ["-", "-", "-", "-."]
 
    for idx, res in enumerate(resultados_malla):
        color = colores[idx % len(colores)]
        estilo = estilos[idx % len(estilos)]

        label = f"$\\Delta x = {res['dx']:.4f}$ m"
        axes[0].plot(res["x"], res["u"], linestyle=estilo, color=color,
                     linewidth=1.8, label=label)
        axes[1].plot(res["x"], res["v"], linestyle=estilo, color=color,
                     linewidth=1.8, label=label)
 
    for ax, var in zip(axes, ["u", "v"]):
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel(var, fontsize=11)
        ax.set_title(f"Perfil de ${var}$ — $t = {t_objetivo:.0f}$ s", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.set_xlim([0, L])
 
    plt.suptitle("Efecto del tamaño de malla $\\Delta x$ sobre la solución",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("efecto_malla.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  >> Figura guardada: efecto_malla.png")
 
 
def graficar_efecto_cfl(resultados_cfl, t_objetivo, L):
    """
    Compara perfiles de u para distintos números de Courant (CFL)
    en un instante dado. Permite visualizar inestabilidades numéricas.
 
    Parámetros
    ----------
    resultados_cfl : list[dict] - Lista de dicts con keys: cfl, dt, x, u, v
    t_objetivo     : float      - Instante de tiempo a comparar [s]
    L              : float      - Longitud del dominio [m]
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colores = ["#55ffe2", "#37023a", "#ff0051", "#6f00ff"]
    estilos = ["-", "--", ":", "-"]
 
    for idx, res in enumerate(resultados_cfl):
        color = colores[idx % len(colores)]
        estilo = estilos[idx % len(estilos)]

        label = f"CFL = {res['cfl']:.3f}  ($\\Delta t = {res['dt']:.4f}$ s)"
        axes[0].plot(res["x"], res["u"], linestyle=estilo, color=color,
                     linewidth=1.8, label=label)
        axes[1].plot(res["x"], res["v"], linestyle=estilo, color=color,
                     linewidth=1.8, label=label)
 
    for ax, var in zip(axes, ["u", "v"]):
        ax.set_xlabel("x [m]", fontsize=10)
        ax.set_ylabel(var, fontsize=11)
        ax.set_title(f"Perfil de ${var}$ — $t = {t_objetivo:.0f}$ s", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
        ax.set_xlim([0, L])
 
    plt.suptitle("Efecto del número de Courant (CFL) sobre la estabilidad",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("efecto_cfl.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  >> Figura guardada: efecto_cfl.png")