#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import seed, multivariate_normal
from numpy import sqrt, exp, mean, histogram, pi
import numpy as np
import time
import sys
import re
import getopt
import os

__author__ = 'Isaías Rodríguez <isurwars@gmail.com>'


def uso(nombre_programa):
    texto = "\nuso: " + nombre_programa + """ [OPCIONES] archivo_car

    DESCRIPCION
        Programa que genera un conjunto de velocidades que corresponden a la
        distribucion de probabilidad de Maxwell-Boltzmann.

        La distribución de probabilidades de Maxwell-boltzmann es funcion de la
        masa de los atomos, por lo que deben indicarse los tipos de atomos que
        se encuentran en la distribución. Debe darse el nombre del archivo
        donde se en cuentra una lista que contenga los atomos de los cuales se
        desea una velocidad aleatoria que cumpla con la distribucion de
        Maxwell-Boltzmann.

        Por ejemplo, si son 3 atomos de silicio y dos de germanio, el archivo
        debera listarlos como sigue:

                Si
                Si
                Si
                Ge
                Ge

        Las velocidades que se generan apareceran en ese mismo orden.
        El archivo que se proporciona puede ser un archivo tipo car o cualquier
        otro en el que aparezca al principio de la linea un simbolo atomico.

    OPCIONES
        -h --help
                Muestra esta ayuda y sale.

        -s --semilla=<numero>
                Establece la semilla para el generador de numeros aleatorios.
                El numero puede ser cualquier real positivo o negativo. Si no
                se especifica se utiliza el reloj del sistema con la función
                time.time().

        -t --temperatura=<numero>
                Temperatura del sistema en kelvin. Por defecto sera 300 K.

        -o --archivo_salida=<archivo>
            Nombre del archivo de salida, por defecto se usa la extension dpmb
            Si no se da un nombre el nombre por defecto sera el nombre base del
            archivo de atomos.

        -m --unidad_ms6
            Por defecto las unidades de velocidad en DMol3 (2016) son:
            (amu*bohr^2/ Hartree)^1/2, si se desea usar DMol3 (6.1) las
            unidades eran (bohr/atu).

        -c --castep
            Fuerza el modo de salida a CASTEP, aunque el de entrada sea un CAR.

    """
    print(texto)


masas_atomicas = {
    'H' :   1.0079, 'He':   4.0026, 'Li':   6.9410, 'Be':   9.0122,
    'B' :  10.8110, 'C' :  12.0107, 'N' :  14.0067, 'O' :  15.9994,
    'F' :  18.9984, 'Ne':  20.1797, 'Na':  22.9898, 'Mg':  24.305 ,
    'Al':  26.9815, 'Si':  28.0855, 'P' :  30.9738, 'S' :  32.065 ,
    'Cl':  35.453 , 'Ar':  39.948 , 'K' :  39.0983, 'Ca':  40.078 ,
    'Sc':  44.9559, 'Ti':  47.867 , 'V' :  50.9415, 'Cr':  51.9961,
    'Mn':  54.9381, 'Fe':  55.845 , 'Co':  58.9333, 'Ni':  58.6934,
    'Cu':  63.546 , 'Zn':  65.409 , 'Ga':  69.723 , 'Ge':  72.64  ,
    'As':  74.9216, 'Se':  78.96  , 'Br':  79.904 , 'Kr':  83.798 ,
    'Rb':  85.4678, 'Sr':  87.62  , 'Y' :  88.9059, 'Zr':  91.224 ,
    'Nb':  92.9064, 'Mo':  95.94  , 'Tc':  98     , 'Ru': 101.07  ,
    'Rh': 102.9055, 'Pd': 106.42  , 'Ag': 107.8682, 'Cd': 112.411 ,
    'In': 114.818 , 'Sn': 118.71  , 'Sb': 121.76  , 'Te': 127.6   ,
    'I' : 126.9045, 'Xe': 131.293 , 'Cs': 132.9055, 'Ba': 137.327 ,
    'La': 138.9055, 'Ce': 140.116 , 'Pr': 140.9077, 'Nd': 144.242 ,
    'Pm': 145.    , 'Sm': 150.36  , 'Eu': 151.964 , 'Gd': 157.25  ,
    'Tb': 158.9254, 'Dy': 162.5   , 'Ho': 164.9303, 'Er': 167.259 ,
    'Tm': 168.9342, 'Yb': 173.04  , 'Lu': 174.967 , 'Hf': 178.49  ,
    'Ta': 180.9479, 'W' : 183.84  , 'Re': 186.207 , 'Os': 190.23  ,
    'Ir': 192.217 , 'Pt': 195.084 , 'Au': 196.9666, 'Hg': 200.59  ,
    'Tl': 204.3833, 'Pb': 207.2   , 'Bi': 208.9804, 'Po': 209.    ,
    'At': 210.    , 'Rn': 222.    , 'Fr': 223.    , 'Ra': 226.    ,
    'Ac': 227.    , 'Th': 232.0381, 'Pa': 231.0359, 'U' : 238.0289,
    'Np': 237.    , 'Pu': 244.    , 'Am': 243.    , 'Cm': 247.    ,
    'Bk': 247.    , 'Cf': 251.    , 'Es': 252.    , 'Fm': 257.    ,
    'Md': 258.    , 'No': 259.    , 'Lr': 262.    , 'Rf': 261.    ,
    'Db': 262.    , 'Sg': 266.}

# ------------------------------------------------------------------------------
# constante de boltzmann
k   = 1.3806e-23        # J/K
# Unidad de Masa Atómica
uma = 1.6605e-27        # kg
# Radio de Bohr
a0 = 5.2917e-11         # m
# Energia de Hartree
Ha = 4.3597e-18         # J
# Unidad de Tiempo Atomico
uta = 1.0327e-15        # s


def norma(vector):
    return sqrt((vector ** 2).sum())


def descomponer_vector(vector):
    norm = norma(vector)
    if norm == 0:
        return 0.0, vector
    return norm, vector / norm


def DPMB_momento(T, m, momento):
    """Funcion de distribucion de probabilidad de
     momento de Maxwell-Boltzmann."""
    P2 = momento[0]**2. + momento[1]**2. + momento[2]**2.
    aux = 2. * m * uma * k * T
    return (pi * aux)**(-1.5) * exp(-P2 / aux)


def momento_mas_probable(T, m):
    """Moda de la distribucion de Maxwell-Boltzmann de momentos"""
    return sqrt(2 * k * T * uma * m)


def multi_momentos_MB(temperatura, masas):
    """"Genera una lista de velocidades aleatorias, las cuales se ajustan a la
    distribucion de Maxwell-Boltzmann"""
    mean = [0.0, 0.0, 0.0]              # Centro de gravedad
    cov = [[1, 0, 0],                   # Matriz de covarianza
           [0, 1, 0],
           [0, 0, 1]]
    num_atomos = len(masas)
    
    # Genera vectores aleatorios con distribucion normal estandar
    lista_normales = multivariate_normal(mean, cov, num_atomos)
    
    # Genera velocidades con la varianza física correcta para cada átomo
    lista_velocidades = []
    for i in range(num_atomos):
        sigma = sqrt(temperatura * k / (uma * masas[i]))
        v = lista_normales[i] * sigma
        lista_velocidades.append(v)
    
    lista_velocidades = np.array(lista_velocidades)
    
    # Corregir el centro de masas si el numero de atomos es mayor a 10
    if num_atomos > 10:
        v_cm = np.zeros(3)
        for i in range(num_atomos):
            v_cm += masas[i] * lista_velocidades[i]
        v_cm /= sum(masas)
        for i in range(num_atomos):
            lista_velocidades[i] -= v_cm

    # Cambiar de velocidades a rapidez y dirección
    prueba_T = 0.
    lista_rapidez = []
    lista_unitarios = []
    for i in range(num_atomos):
        rapidez, unitario = descomponer_vector(lista_velocidades[i])
        prueba_T += (uma * masas[i] * rapidez ** 2.) / (3. * k)
        lista_rapidez.append(rapidez)
        lista_unitarios.append(unitario)
        
    prueba_T /= num_atomos
    
    # Rescalar las velocidades para obtener la temperatura exacta solicitada
    if prueba_T > 0:
        alpha = sqrt(temperatura / prueba_T)
        for i in range(num_atomos):
            lista_rapidez[i] *= alpha
            
    return lista_rapidez, lista_unitarios


def nombre_base(nombre_archivo):
    """Da el nombre base del sistema."""
    return os.path.splitext(nombre_archivo)[0]


def atomos(nombre_archivo):
    """Lee los nombres de los atomos del archivo. Devuelve una lista."""
    lista_atomos = []

    try:
        with open(nombre_archivo, "r", encoding="utf-8") as archivo_entrada:
            # patron_atomo: coincide con el símbolo de un átomo al inicio de la línea (puede tener espacios al inicio)
            patron_atomo = re.compile(r'^\s*([A-Z][a-z]?)(?=[\d\s]|$|[\+\-])')
            for linea in archivo_entrada:
                patron_encontrado = re.match(patron_atomo, linea)
                if patron_encontrado is not None:
                    atomo = patron_encontrado.group(1)
                    # Solo agregarlo si es un elemento válido en masas_atomicas
                    if atomo in masas_atomicas:
                        lista_atomos.append(atomo)
    except FileNotFoundError:
        print(f"Error: El archivo '{nombre_archivo}' no existe.", file=sys.stderr)
        sys.exit(1)

    return lista_atomos


def main():
    """Función principal."""
    # argumento, ver documentación del módulo 'getopt'
    try:
        opciones, argumentos = getopt.getopt(sys.argv[1:],
                                             "hct:o:s:",
                                             ["help",
                                              "castep",
                                              "temperatura=",
                                              "archivo_salida=",
                                              "semilla="])
    except getopt.GetOptError as msg:
        print(msg)
        print("Para ayuda use --help")
        sys.exit(2)

    # si se elije la opción de ayuda
    for (opcion, valor) in opciones:
        if opcion in ("-h", "--help"):
            uso(sys.argv[0])
            sys.exit(0)

    # valores por defecto
    castep_mode = False
    nombre_archivo_salida = None
    temperatura           = 300.0         # 300 K
    semilla               = time.time()   # tiempo del sistema
    factor = uta / a0     # factor de m/s a bohr/fs. (MS 2016)
    factor_castep = 1e-2  # factor de m/s a Å/ps (CASTEP)

    # procesamiento de opciones
    for (opcion, valor) in opciones:
        if opcion in ("-o", "--archivo_salida"):
            nombre_archivo_salida  = valor
        elif opcion in ("-t", "--temperatura"):
            temperatura = float(valor)
        elif opcion in ("-s", "--semilla"):
            semilla = float(valor)
        elif opcion in ("-c", "--castep"):
            castep_mode = True
        else:
            assert False, "Opcion inmanejable: " + opcion

    if not argumentos:
        print("Error: Debe especificar un archivo de entrada.", file=sys.stderr)
        uso(sys.argv[0])
        sys.exit(1)

    # procesamiento de argumentos
    nombre_archivo = argumentos[0]
    lista_atomos   = atomos(nombre_archivo)

    if not lista_atomos:
        print(f"Error: No se encontraron átomos válidos en el archivo '{nombre_archivo}'.", file=sys.stderr)
        sys.exit(1)

    if nombre_archivo_salida is None:
        nombre_archivo_salida = nombre_base(nombre_archivo)

    # establece la semilla del generador de números aleatorios
    seed(int(semilla))

    masas = []
    for atomo in lista_atomos:
        masa = masas_atomicas[atomo]
        masas += [masa]

    rapideces, vec_aleatorios = multi_momentos_MB(temperatura, masas)
    cmp = np.zeros(3)
    prueba_T = 0.
    for i in range(len(masas)):
        cmp += masas[i] * rapideces[i] * vec_aleatorios[i]
        prueba_T += (uma * masas[i] * rapideces[i] ** 2.) / (3. * k)
    prueba_T /= len(masas)

    # Calcular velocidad de centro de masas en m/s
    cmp /= sum(masas)

    if castep_mode:
        # Formato CASTEP
        with open(nombre_archivo_salida + ".cell", 'w', encoding="utf-8") as archivo_salida:
            archivo_salida.write("# Velocidades según la distribución de Maxwell-Boltzmann\n")
            archivo_salida.write(f"# Semilla : {semilla}\n")
            archivo_salida.write(f"# Número de átomos : {len(masas)}\n")
            archivo_salida.write(f"# Temperatura solicitada : {temperatura:.4f} K\n")
            archivo_salida.write(f"# Velocidad del cm: [{cmp[0] * factor_castep:.8e} {cmp[1] * factor_castep:.8e} {cmp[2] * factor_castep:.8e}] A/ps\n")
            archivo_salida.write(f"# Prueba de Temperatura : {prueba_T:.4f} K\n")
            archivo_salida.write("%BLOCK IONIC_VELOCITIES\n")
            for i in range(len(masas)):
                vx = rapideces[i] * vec_aleatorios[i][0] * factor_castep
                vy = rapideces[i] * vec_aleatorios[i][1] * factor_castep
                vz = rapideces[i] * vec_aleatorios[i][2] * factor_castep
                archivo_salida.write(f"  {lista_atomos[i]:4s}  {vx:15.8f}  {vy:15.8f}  {vz:15.8f}\n")
            archivo_salida.write("%ENDBLOCK IONIC_VELOCITIES\n")
    else:
        # Formato DMol3 (.dpmb)
        # Convertir velocidad de centro de masas a unidades de salida
        cmp *= factor
        media = mean(rapideces)
        variance = sqrt(temperatura * k / (uma * max(masas)))
        histo, bandas = histogram(rapideces, bins="auto")

        with open(nombre_archivo_salida + ".dpmb", 'w', encoding="utf-8") as archivo_salida:
            archivo_salida.write("# Velocidades según la distribución de Maxwell-Boltzmann\n")
            archivo_salida.write(f"# Semilla : {semilla}\n")
            archivo_salida.write(f"# Número de átomos : {len(masas)}\n")
            archivo_salida.write(f"# Temperatura solicitada : {temperatura:8.4f}K\n")
            archivo_salida.write(f"# Velocidad del cm: [{cmp[0]:.8e} {cmp[1]:.8e} {cmp[2]:.8e}]\n")
            archivo_salida.write(f"# Prueba de Temperatura : {prueba_T:8.4f}K\n")
            archivo_salida.write(f"# Media de la distribucion : {media * factor:8.4f}\n")
            archivo_salida.write(f"# Media ideal de Maxwell-Boltzmann: {sqrt(8 / 3.1415926) * variance * factor:8.4f}\n")
            archivo_salida.write("# Histograma de Velocidades\n")
            archivo_salida.write("# ------------------------------------------------- #\n")
            archivo_salida.write("      velocidad         cuentas\n")
            for i in range(len(histo)):
                archivo_salida.write(f"{bandas[i] * factor:15.6f} {histo[i]:15d}\n")
            archivo_salida.write("# ------------------------------------------------- #\n")
            archivo_salida.write("MD_Velocity User_Defined\n")

            for i in range(len(masas)):
                archivo_salida.write(
                    f"{i + 1:4d} {rapideces[i] * factor:15.6f} "
                    f"{vec_aleatorios[i][0]:15.6f} {vec_aleatorios[i][1]:15.6f} {vec_aleatorios[i][2]:15.6f} "
                    f"#{lista_atomos[i]}\n"
                )


if __name__ == '__main__':
    main()