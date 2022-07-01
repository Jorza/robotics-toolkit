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
    # Convert expression to symbol if needed
    if isinstance(expr, str):
        expr = sp.symbols(expr)
    # Print as latex
    print(f'$${sp.latex(expr)}$$')


def print_equation(left_expr, right_expr, latex=False):
    # Convert expressions to symbols if needed
    if isinstance(left_expr, str):
        left_expr = sp.symbols(left_expr)
    if isinstance(right_expr, str):
        right_expr = sp.symbols(right_expr)

    # Print the equation
    if latex:
        print(f'$${sp.latex(left_expr)} = {sp.latex(right_expr)}$$')
    else:
        print(f'{left_expr} == {right_expr}')


def func_equations_dict(equations_dict, func, *args, output=True, keys=None, **kwargs):
    """
    Apply a function to all equations in a dictionary

    Args:
        equations_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
            Each expression can be a single expression or a list of expressions.
        func - A function to apply to each entry of the dict. Is called like func(name, equation, *args **kwargs).
            The 'name' is the key where the equation is stored, or the concatenation of key + index if the equation is in a list.
        args - Additional arguments for func
        output - If set to False, then the function will not return anything.
            Increases efficiency if no return needed.
        keys - A list of keys to operate on. If omitted, all keys are operated on
        kwargs - Additional keyword arguments for func
    Return:
        A new equation dict with the same structure as the original, with the outputs of func
    """

    if output:
        # Generate output dict that is different from the input one
        equations_dict = copy.deepcopy(equations_dict)

    if keys is None:
        # Operate on all equations by deafult
        keys = equations_dict.keys()

    for name in keys:
        # Apply the given function to each of the equations
        equations = equations_dict[name]
        if isinstance(equations, list):
            func_output = [func(f'{name}{i}', eq, *args, **kwargs) for i, eq in enumerate(equations)]
        else:
            func_output = func(name, equations_dict[name], *args, **kwargs)
        if output:
            equations_dict[name] = func_output

    if output:
        return equations_dict


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
    func_equations_dict(equations_dict, print_equation, output=False, keys=keys, latex=latex)


def substitute_equations_dict(equations_dict, subs_map, keys=None):
    """
    Substitute into all equations in a dictionary

    Args:
        equations_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
            Each expression can be a single expression or a list of expressions.
        subs_map - A dictionary of substitutions, as you would use for expr.subs(subs_map)
        keys - A list of keys to substitute. If omitted, all equations are substituted
    Return:
        An equation dict with all values substituted
    """
    return func_equations_dict(equations_dict, lambda _, eq: eq.subs(subs_map), keys=keys)


def free_symbols_equations_dict(equations_dict, keys=None):
    """
    Get all free symbols from a dictionary of equations

    Args:
        equations_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
            Each expression can be a single expression or a list of expressions.
        keys - A list of keys for equations to get the symbols from. If omitted, all free symbols are collected
    Return:
        A set of free symbols
    """
    free_symbols = set()
    func_equations_dict(equations_dict, lambda _, eq: free_symbols.update(eq.free_symbols), output=False, keys=keys)
    return free_symbols


def round_equations_dict(equations_dict, num_sig_figs, keys=None):
    """
    Round all equations in a dictionary

    Args:
        equations_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
            Each expression can be a single expression or a list of expressions.
        num_sig_figs - Number of significant figures to round to, as you would use for sp.N(expr, num_sig_figs)
        keys - A list of keys to round. If omitted, all equations are rounded
    Return:
        An equation dict with all values rounded
    """
    return func_equations_dict(equations_dict, lambda _, eq: sp.N(eq, num_sig_figs), keys=keys)


def flatten_equations_dict(equations_dict, keys=None):
    """
    Remove all lists from the equation dict, replace them with their own key entries.

    Args:
        equations_dict - A dictionary of equations. Has entries of the form 'symbolic_name': expression.
            Each expression can be a single expression or a list of expressions.
        keys - A list of keys to put into the new flattened dict. If omitted, all equations are taken
    Return:
        A flattened dictionary of equations
    """
    flat_dict = dict()
    func_equations_dict(equations_dict, lambda name, eq: flat_dict.__setitem__(name, eq), output=False, keys=keys)
    return flat_dict
