
class DocStringExample:
    def __init__(self, a: int, b: float):
        """Example for a docstring. Here a short description

        Here you may write a longer description of class functionality. It may include formulas such as

        .. math:
            a^2+b^2=c^2
        
        Args:
            a (int): _description_
            b (float): _description_
            
        .. note::
           Here you can insert soem note, such as: Be careful, it may be dangerous, see :class:`rivapy.marketdata.DiscountCurve`.

        .. seealso::
           Refer to other places where one may find this class or refer to classes simliar to this one (e.g. :class:`rivapy.marketdata.DiscountCurve`).

        Example:
            >>> depp = DocStringExample(1.0,2.0)

        
        """