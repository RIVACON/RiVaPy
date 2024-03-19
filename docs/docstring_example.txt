# Here we demonstrate the rivapy's best practice docstring ;-)

def dummy_function(a: np.ndarray, b: str)->float:
    """This dummy function does dummy values

    Args:
        a (np.ndarray): Describe all arguments. To cite external packages, see the documentation of the argument below._
        b (str): This variable defines b, the super argument in this method. See :ext:py:`pandas.date_range` for a clearer definition.

    Returns:
        float: A description of the return values.
    """
    pass