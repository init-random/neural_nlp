import cytoolz.curried as z


@z.curry
def unzip(seq):
    l = z.pipe(seq, z.first, tuple, len)
    t = z.pipe(range(l), z.map(lambda i: z.pipe(seq, z.pluck(i), iter)), tuple)
    return t

