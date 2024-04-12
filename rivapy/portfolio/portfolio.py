"""
Defines a portfolio class. 
"""

class PortfolioConstituent:
    def __init__(self):
        self.__financial_data = None

    @property
    def financial_data(self):
        if self.__financial_data is None:
            pass
            # yf.load()
        return self.__financial_data

class Portfolio:
    
    def __init__(self, constitutents: PortfolioConstituent):
        self.__constitutents = constitutents
        
        self.__supported_specifications = None #[EuropeanVanillaSpecification]

        # todo market data dict
        self.__market_data = {}
        # todo: as a first step: make sure, that all constitutents are stocks.
        # todo: optionally pass yahoo finance symbols to donwload market data.

    @property
    def value(self):
        return np.sum([ c.value for c in self.__constituents ])
    
    @property
    def nominal(self):
        return np.sum([ c.nominal for c in self.__constituents ])

    @property
    def size(self):
        return len(self.__constitutents)

    @property
    def constituents(self):
        return self.__constitutents
    
    def add_constituent(self, const):
        # if any(isinstance(...))
        pass