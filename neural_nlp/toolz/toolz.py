import cytoolz.curried as z


@z.curry
def unzip(seq):
    """
    The implementation of this using the standard library is zip(*seq). This is a lazy implementation. For example,
    this function will a tule (a, b) where list(a) == [1, 3, 5] and list(b) == [2, 4, 6]. a and b are iterables

    a, b = unzip([(1, 2), (3, 4), (5, 6)])

    :param seq:
    :rtype: tuple
    :return: tuple of iterables
    """
    l = z.pipe(seq, z.first, tuple, len)
    t = z.pipe(range(l), z.map(lambda i: z.pipe(seq, z.pluck(i), iter)), tuple)
    return t


def piped_map(seq, *funcs):
    """
    If we have piped_map([a, b], f, g, h) then this will return [g(f(a)), g(f(b))]. In other words, we pipe
    each value of seq through *funcs.

    :param iterable seq: a sequence of values
    :param vararg funcs: callable functions
    :rtype: iterable
    :return: sequence of values piped through the provided functions
    """
    return z.map(lambda v: z.pipe(v, *funcs), seq)


@z.curry
def split(delim, s):
    """
    This is a functional helper for str.split, which may be called like

    dash_split = split('-')
    vals = dash_split('a-b-c')
    # vals == ['a', 'b', 'c']

    :param str delim: split delimiter
    :param str s: raw text string
    :rtype: list
    :return: split string
    """
    return s.split(delim)

