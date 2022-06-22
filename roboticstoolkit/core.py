import sympy as sp

# Define some handy constants
x_vec = sp.Matrix([1, 0, 0])
y_vec = sp.Matrix([0, 1, 0])
z_vec = sp.Matrix([0, 0, 1])


# Define some general-purpose functions
def print_equations_dict(equations_dict, keys=None):
    """
    Print equations from a dictionary

    Args:
        equation_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
            Is printed as 'symbolic_name == expression'.
            Each expression can be a single expression or a list of expressions.
            If it is a list, the entries will be printed as separate equations.
        keys - A list of keys (symbolic names) to print. If omitted, all equations are printed
    """

    # Print all equations by default
    if keys is None:
        keys = equations_dict.keys()
    
    # Print each of the given equations
    for name in keys:
        equations = equations_dict[name]
        if isinstance(equations, list):
            for i, eq in enumerate(equations):
                print(f'{name}{i} == {eq}')
        else:
            print(f'{name} == {equations}')