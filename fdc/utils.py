def transform(X, transform_list):
    Xnew = np.copy(X)
    for f, c in transform_list:
        Xnew = f(Xnew[c])
    return Xnew

    