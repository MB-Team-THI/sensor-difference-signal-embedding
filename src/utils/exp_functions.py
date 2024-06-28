

def inverse_exp_function(signal, params):
    # https://www.desmos.com/calculator/3fisjexbvp?lang=de

    a = params['a']
    b = params['b']
    d = params['d']
    shift_x = params['shift_x']
    shift_y = params['shift_y']

    signal_out = []
    for x in signal:
        sgn = x / abs(x)

        y = 2* (a*(-b**(-d*( abs(x) - shift_x)))+shift_y)

        signal_out.append(sgn*y)

    return signal_out


def log_function():
    pass