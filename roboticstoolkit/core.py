import sympy as sp
import copy

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


def print_latex(expr):
    print(f'$${sp.latex(expr)}$$')


def print_equation(left_expr, right_expr, latex=False):
    if latex:
        print(f'$${sp.latex(left_expr)} = {sp.latex(right_expr)}$$')
    else:
        print(f'{left_expr} == {right_expr}')


def print_equations_dict(equations_dict, keys=None, latex=False):
    """
    Print equations from a dictionary

    Args:
        equations_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
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
            variables = sp.symbols(f'{name}:{len(equations)}')
            for i in range(len(equations)):
                print_equation(variables[i], equations[i], latex=latex)
        else:
            print_equation(sp.symbols(f'{name}'), equations, latex=latex)


def substitute_equations_dict(equations_dict, subs_map):
    """
    Substitute into all equations in a dictionary

    Args:
        equations_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
            Each expression can be a single expression or a list of expressions.
        subs_map - A dictionary of substitutions, as you would use for expr.subs(subs_map)
    Return:
        An equation dict with all values substituted
    """

    # Generate output dict that is different from the input one
    equations_dict = copy.deepcopy(equations_dict)

    # Substitute each of the given equations
    for name, equations in equations_dict.items():
        if isinstance(equations, list):
            equations_dict[name] = [eq.subs(subs_map) for eq in equations]
        else:
            equations_dict[name] = equations_dict[name].subs(subs_map)

    return equations_dict


def free_symbols_equations_dict(equations_dict):
    """
    Get all free symbols from a dictionary of equations

    Args:
        equations_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
            Each expression can be a single expression or a list of expressions.
    Return:
        A set of free symbols
    """

    # Prepare the output dict
    free_symbols = set()

    # Get the symbols from each of the given equations
    for name, equations in equations_dict.items():
        if isinstance(equations, list):
            for eq in equations:
                free_symbols.update(eq.free_symbols)
        else:
            free_symbols.update(equations_dict[name].free_symbols)

    return free_symbols


def flatten_equations_dict(equations_dict):
    """
    Remove all lists from the equation dict, replace them with their own key entries

    Args:
        equations_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
            Each expression can be a single expression or a list of expressions.
    Return:
        A flattened dictionary of equations
    """

    # Prepare the output dict
    flat_dict = dict()

    # Substitute each of the given equations
    for name, equations in equations_dict.items():
        if isinstance(equations, list):
            for i, eq in enumerate(equations):
                flat_dict[f'{name}{i}'] = eq
        else:
            flat_dict[name] = equations

    return flat_dict
