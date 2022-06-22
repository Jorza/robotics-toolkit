import sympy as sp

# Define some handy constants
x_vec = sp.Matrix([1, 0, 0])
y_vec = sp.Matrix([0, 1, 0])
z_vec = sp.Matrix([0, 0, 1])


# Define some general-purpose functions
def diff_total(expr, diffby, diffmap):
    """
    Take the total derivative with respect to a variable.

    Based on https://robotfantastic.org/total-derivatives-in-sympy.html

    Args:
        expr - expression to differentiate
        diffby - differentiate with respect to this variable
        diffmap - dictionary mapping all variables that depend on diffby to their symbolic derivative
    Return:
        A sympy expression

    
    Example:

        theta, t, theta_dot = symbols("theta t theta_dot")
        diff_total(cos(theta), t, {theta: theta_dot})

    returns

        -theta_dot*sin(theta)
    """
    # Create functional forms of all symbols in the diffmap
    funcmap = {v:sp.Function(v)(diffby) for v in diffmap}
    # Replace symbols in the expression by their functional forms
    fnexpr = expr.subs(funcmap)
    # Do the differentiation
    diffexpr = sp.diff(fnexpr, diffby)
    # Replace the Derivatives with the variables in diffmap
    derivmap = {sp.Derivative(funcmap[v], diffby):dv for v, dv in diffmap.items()}
    finaldiff = diffexpr.subs(derivmap)
    # Replace the functional forms with their original form
    return finaldiff.subs({funcmap[v]:sp.Symbol(f'{v}') for v in diffmap})


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