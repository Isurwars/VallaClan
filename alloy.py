#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program produce a random array of atoms in a provided structure
it requieres a initial structure in CAR file format.

You may provide an optional concentration file, for details please
check the use function, or call the -h option.
"""


import re
import getopt
import sys
import time
import numpy as np
import os

__author__ = "Isaías Rodriguez <isurwars@gmail.com>"


# ------------------------------------------------------------------------------
class Atom:
    """
    A class to represent basic atom characteristics.

    Attributes:
    ----------
    name : str
        The name of the atom.
    number : int
        The atomic number.
    x : float
        The x-coordinate of the atom.
    y : float
        The y-coordinate of the atom.
    z : float
        The z-coordinate of the atom.
    residue : str, optional
        The residue name (default is "XXXX").
    res_num : int, optional
        The residue number (default is 1).
    potential : str, optional
        The potential type (default is "xx").
    element : str, optional
        The element symbol (default is "H").
    charge : float, optional
        The charge of the atom (default is 0.0).
    """

    def __init__(
        self,
        name,
        number,
        x,
        y,
        z,
        residue="XXXX",
        res_num=1,
        potential="xx",
        element="H",
        charge=0.0,
    ):
        """
        Initialize the Atom class with attributes.

        Parameters:
        ----------
        name : str
            The name of the atom.
        number : int
            The atomic number.
        x : float
            The x-coordinate of the atom.
        y : float
            The y-coordinate of the atom.
        z : float
            The z-coordinate of the atom.
        residue : str, optional
            The residue name (default is "XXXX").
        res_num : int, optional
            The residue number (default is 1).
        potential : str, optional
            The potential type (default is "xx").
        element : str, optional
            The element symbol (default is "H").
        charge : float, optional
            The charge of the atom (default is 0.0).
        """
        self.name = name
        self.number = number
        self.x = x
        self.y = y
        self.z = z
        self.residue = residue
        self.res_num = res_num
        self.potential = potential
        self.element = element
        self.charge = charge

    def __str__(self) -> str:
        """
        Return a string representation of the Atom object.

        Returns:
        -------
        str
            A string representation of the atom's characteristics.
        """
        return (
            f"Atom: {self.name}, "
            f"Number: {self.number}, "
            f"Coordinates: ({self.x}, {self.y}, {self.z}), "
            f"Residue: {self.residue}, "
            f"Residue Number: {self.res_num}, "
            f"Potential: {self.potential}, "
            f"Element: {self.element}, "
            f"Charge: {self.charge}"
        )


# ------------------------------------------------------------------------------
class Cell:
    """
    A class to represent a unit cell in a crystal structure.

    Attributes:
    ----------
    atoms : list
        A list of Atom objects within the cell.
    a : float
        The length of the cell along the x-axis.
    b : float
        The length of the cell along the y-axis.
    c : float
        The length of the cell along the z-axis.
    alfa : float
        The angle between b and c axes in degrees.
    beta : float
        The angle between a and c axes in degrees.
    gama : float
        The angle between a and b axes in degrees.
    """

    def __init__(self, atoms, a, b, c, alfa, beta, gama):
        """
        Initialize the Cell class with its dimensions and atoms.

        Parameters:
        ----------
        atoms : list
            A list of Atom objects within the cell.
        a : float
            The length of the cell along the x-axis.
        b : float
            The length of the cell along the y-axis.
        c : float
            The length of the cell along the z-axis.
        alfa : float
            The angle between b and c axes in degrees.
        beta : float
            The angle between a and c axes in degrees.
        gama : float
            The angle between a and b axes in degrees.
        """
        self.atoms = atoms
        self.a = a
        self.b = b
        self.c = c
        self.alfa = alfa
        self.beta = beta
        self.gama = gama

    def volume(self) -> float:
        """
        Calculate the volume of the unit cell.

        Returns:
        -------
        float
            The volume of the unit cell.
        """
        alfa_rad = np.radians(self.alfa)
        beta_rad = np.radians(self.beta)
        gama_rad = np.radians(self.gama)

        volume = (
            self.a
            * self.b
            * self.c
            * np.sqrt(
                1
                - np.cos(alfa_rad) ** 2
                - np.cos(beta_rad) ** 2
                - np.cos(gama_rad) ** 2
                + 2 * np.cos(alfa_rad) * np.cos(beta_rad) * np.cos(gama_rad)
            )
        )
        return volume

    def __str__(self) -> str:
        """
        Return a string representation of the Cell object.

        Returns:
        -------
        str
            A string representation of the cell's characteristics.
        """
        return (
            f"Cell: a={self.a}, b={self.b}, c={self.c}, "
            f"alfa={self.alfa}, beta={self.beta}, gama={self.gama}, "
            f"Number of atoms: {len(self.atoms)}"
        )


# ------------------------------------------------------------------------------
def use(nombre_programa: str) -> None:
    """
    Display the usage message for the program.

    Parameters:
    ----------
    nombre_programa : str
        The name of the program to be displayed in the usage message.
    """
    texto = (
        f"\nuso: {nombre_programa} [OPCIONES] archivo_car\n\n"
        "DESCRIPCION\n"
        "    Programa para generar una aleacion a partir de un archivo car\n"
        "    que contiene las posiciones átomicas.\n\n"
        "OPCIONES\n"
        "    -h --help\n"
        "            Muestra esta ayuda y sale.\n\n"
        "    -o --archivo-salida=<nombre_del_archivo_de_salida>\n"
        "            Establece el nombre del archivo de salida, si esta\n"
        "            opción no se utiliza el nombre del archivo por defecto\n"
        "            sera el nombre base del archivo car.\n\n"
        "    -e --archivo_concentraciones=<archivo>\n"
        "            En este archivo se incluiran los tipos de átomos\n"
        "            presentes en la aleación y la concentración requerida.\n"
        "            Se deben proporcionar como porcentajes, como numero de\n"
        "            átomos de cada tipo, o comoconcentración quimica\n"
        "            deseada, pero no mezcladas entre si.\n\n"
        "            #Porcentajes:\n"
        "                Si  65\n"
        "                Mg  23\n"
        "                C   12\n\n"
        "            #Concentraciones:\n"
        "                Si 0.65\n"
        "                Mg 0.23\n"
        "                C  0.12\n\n"
        "            # Numero de átomos de cada tipo:\n"
        "                Si 166\n"
        "                Mg  59\n"
        "                C   31\n"
    )

    print(texto)


# ------------------------------------------------------------------------------
def read_car_file(file_name) -> Cell:
    """
    Read a CAR file to populate a Cell class
    """
    # Abre el archivo ARC en modo de sólo lectura.
    with open(file_name, "r", encoding="utf-8") as input_file:
        lines = input_file.readlines()

    # Expresión regular para un número real
    # num_real = '([+-]?\.?\d+\.?\d*(?:[eE][+-]?\d+)?)'
    # Expresión regular para un número entero
    # '([+-]?\d+)'
    # Expresión regular para encontrar un elemento químico seguido de un número
    # y tres números reales (coordenadas) separados por espacios.
    # Esta expresión regular generará 5 grupos de búsqueda:
    #      grupo 1: nombre del elemento químico
    #      grupo 2: número del elemento químico
    #      grupo 3: coordenada x
    #      grupo 4: coordenada y
    #      grupo 5: coordenada z
    patron_ele_num_xyz = re.compile(
        r"^([A-Z][a-z]?)([0-9A-Z]+)"
        + r"\s+([-+]?\.?\d+\.?\d*(?:[eE][-+]?\d+)?|\*+)"
        + r"\s+([-+]?\.?\d+\.?\d*(?:[eE][-+]?\d+)?|\*+)"
        + r"\s+([-+]?\.?\d+\.?\d*(?:[eE][-+]?\d+)?|\*+)"
        + r"\s+([a-zA-Z0-9]{1,4})"
        + r"\s+([+-]?\d+)"
        + r"\s+([a-zA-Z0-9]{1,4})"
        + r"\s+([A-Z][a-z]?)"
        + r"\s+([+-]?\.?\d+\.?\d*(?:[eE][+-]?\d+)?|\*+)"
    )

    # El patrón anterior contempla la explosión de la celda, en tal caso
    # un(os) asterisco(s) deberá(n) aparecer en lugar de la coordenada

    # patrón encontrar los parámetros de la celda, a, b, c, alfa, beta, gama
    patron_parametros_celda = re.compile(
        r"^!?PBC"
        + r"\s+([-+]?\.?\d+\.?\d*(?:[eE][-+]?\d+)?|\*+)"
        + r"\s+([-+]?\.?\d+\.?\d*(?:[eE][-+]?\d+)?|\*+)"
        + r"\s+([-+]?\.?\d+\.?\d*(?:[eE][-+]?\d+)?|\*+)"
        + r"\s+([-+]?\.?\d+\.?\d*(?:[eE][-+]?\d+)?|\*+)"
        + r"\s+([-+]?\.?\d+\.?\d*(?:[eE][-+]?\d+)?|\*+)"
        + r"\s+([-+]?\.?\d+\.?\d*(?:[eE][-+]?\d+)?|\*+)"
    )
    # Default Cell
    cell = Cell([], 1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
    # Lee una por una las líneas del archivo de entrada
    for line in lines:
        # busca el patron EleNumXYZ en la línea
        matching_patron = re.match(patron_ele_num_xyz, line)
        # si se encuentra el patrón en la línea
        if matching_patron is not None:
            cell.atoms.append(
                Atom(
                    matching_patron.group(1),
                    matching_patron.group(2),
                    matching_patron.group(3),
                    matching_patron.group(4),
                    matching_patron.group(5),
                    matching_patron.group(6),
                    matching_patron.group(7),
                    matching_patron.group(8),
                    matching_patron.group(9),
                    matching_patron.group(10),
                )
            )

        # si el patrón no se encontró en la línea
        else:
            cell_parameters = re.match(patron_parametros_celda, line)
            # si se encuentran los parámetros de la celda.
            if cell_parameters is not None:
                cell.a = float(cell_parameters.group(1))
                cell.b = float(cell_parameters.group(2))
                cell.c = float(cell_parameters.group(3))
                cell.alfa = float(cell_parameters.group(4))
                cell.beta = float(cell_parameters.group(5))
                cell.gama = float(cell_parameters.group(6))
    print("CAR file read...")
    return cell


# ------------------------------------------------------------------------------
def parse_options() -> dict:
    """Function to read and parse arguments"""
    try:
        options, arguments = getopt.getopt(
            sys.argv[1:],
            "ho:e:",
            [
                "help",
                "out_file=",
                "concentrations_file=",
                "concentration_file=",
                "archivo-salida=",
                "archivo_concentraciones=",
            ],
        )
    except getopt.error as msg:
        print(msg, file=sys.stderr)
        print("For help use --help", file=sys.stderr)
        sys.exit(2)

    # Check for help option
    for option, value in options:
        if option in ("-h", "--help"):
            use(sys.argv[0])
            sys.exit(0)

    if not arguments:
        print("Error: Must specify an input file.", file=sys.stderr)
        use(sys.argv[0])
        sys.exit(1)

    file_name = arguments[0]
    out_file_name = None
    atomic_concentrations = {}
    concentrations = []

    # Check for the other options
    for option, value in options:
        if option in ("-o", "--out_file", "--archivo-salida"):
            out_file_name = value
        elif option in (
            "-e",
            "--concentration_file",
            "--concentrations_file",
            "--archivo_concentraciones",
        ):
            try:
                with open(value, "r", encoding="utf-8") as concentration_file:
                    for line in concentration_file:
                        line_stripped = line.strip()
                        if not line_stripped or line_stripped.startswith("#"):
                            continue
                        read_aux = line_stripped.split()
                        if len(read_aux) < 2:
                            continue
                        concentration = float(read_aux[1])
                        atomic_concentrations[read_aux[0]] = concentration
                        concentrations.append(concentration)
            except FileNotFoundError:
                print(f"Error: Concentrations file '{value}' not found.", file=sys.stderr)
                sys.exit(1)
        else:
            assert False, "Unknown option: " + option

    if out_file_name is None:
        out_file_name = f"{os.path.splitext(file_name)[0]}.car"

    return {
        "file_name": file_name,
        "out_file_name": out_file_name,
        "concentrations": concentrations,
        "atomic_concentrations": atomic_concentrations,
    }


# ------------------------------------------------------------------------------
def main() -> None:
    """Main function."""
    params = parse_options()
    cell = read_car_file(params["file_name"])
    total_num_atoms = len(cell.atoms)
    num_atoms = total_num_atoms
    type_of_concentrations = sum(params["concentrations"])
    atomic_concentrations = params["atomic_concentrations"]
    
    if type_of_concentrations == total_num_atoms:
        pass
    elif type_of_concentrations == 100:
        sorted_elements = sorted(atomic_concentrations.items(), key=lambda x: x[1])
        for idx, (element, concen) in enumerate(sorted_elements):
            if idx == len(sorted_elements) - 1:
                num_aux = num_atoms
            else:
                num_aux = round((total_num_atoms * concen / 100.0), 0)
                num_aux = min(num_aux, num_atoms)
            atomic_concentrations[element] = num_aux
            num_atoms -= num_aux
    elif type_of_concentrations <= 1.0:
        sorted_elements = sorted(atomic_concentrations.items(), key=lambda x: x[1])
        for idx, (element, concen) in enumerate(sorted_elements):
            if idx == len(sorted_elements) - 1:
                num_aux = num_atoms
            else:
                num_aux = round((total_num_atoms * concen), 0)
                num_aux = min(num_aux, num_atoms)
            atomic_concentrations[element] = num_aux
            num_atoms -= num_aux
    else:
        print(f"Error: The sum of concentrations or atom counts ({type_of_concentrations}) "
              f"does not match the total number of atoms ({total_num_atoms}), "
              "and does not represent percentages (sum to 100) or chemical fractions (sum <= 1.0).", file=sys.stderr)
        sys.exit(1)

    elements = []
    for element, num_element in atomic_concentrations.items():
        for i in range(int(num_element)):
            elements += [element]
            
    # Mezclar la lista para garantizar aleatoriedad
    np.random.shuffle(elements)
    
    count_aux = 0
    for i in cell.atoms:
        i.element = elements[count_aux]
        i.name = i.element + i.number
        count_aux += 1

    with open(params["out_file_name"], "w", encoding="utf-8") as out_file:
        out_file.write("!BIOSYM archive 3\n")
        out_file.write("PBC=ON\n")
        out_file.write("Alloy Generated CAR File\n")
        aux_str = f"!DATE {str(time.asctime(time.localtime(time.time())))}\n"
        out_file.write(aux_str)
        aux_str = (
            f"PBC   {cell.a:.4f}   "
            f"{cell.b:.4f}   "
            f"{cell.c:.4f}   "
            f"{cell.alfa:.4f}   "
            f"{cell.beta:.4f}   "
            f"{cell.gama:.4f} (P1)\n"
        )
        out_file.write(aux_str)
        for i in cell.atoms:
            aux_str = (
                f"{i.name:5s}  "
                f"{float(i.x): 13.9f}  "
                f"{float(i.y): 13.9f}  "
                f"{float(i.z): 13.9f} XXXX 1      xx      "
                f"{format(i.element):4s}  0.000\n"
            )
            out_file.write(aux_str)
        out_file.write("end\n")
        out_file.write("end\n")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(t2 - t1, "s")