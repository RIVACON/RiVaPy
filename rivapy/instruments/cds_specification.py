from datetime import datetime
from typing import List


class CDSSpecification:
    def __init__(self,  premium: float, 
                premium_pay_dates: List[datetime], 
                protection_start: datetime,
                notional: float = 1.0, 
                expiry: datetime=None, 
                recovery: float = None, 
                issuer: str = '', 
                cash_settled: bool = True):
        """Constructor for credit default swap

        Args:
            premium (float): The premium as fraction of notional paid at each premium date.
            premium_pay_dates (List[datetime]): List of dates for premium payments.
            protection_start (datetime): Date when protection starts
            notional (foat): Notional
            expiry (datetime, optional): [description]. Defaults to None.
            recovery (float, optional): The protection is only paid for the real loss (notional minus recovery). If recovery is not specified, it is assumed that  recovery as specified in contract. If no fixed recovery is specified[description]. Defaults to None.
            issuer (str, optional): [description]. Defaults to ''.
            cash_settled (bool, optional): Flag indicating o instrument is physical settled (the protection buyer )

        """
        self.expiry = expiry
        self.premium = premium
        self.premium_pay_dates = premium_pay_dates
        self.protection_start = protection_start
        self.notional = notional
        if expiry is None:
            self.expiry = premium_pay_dates[-1]
        self.recovery = recovery
        self.issuer = issuer
        self.cash_settled = cash_settled
        self.validate()

    def validate(self):
        """Some simple validation
        """
        if len(self.premium_pay_dates) == 0:
            raise Exception('Premium payment dates must not be empty.')
        
        
        