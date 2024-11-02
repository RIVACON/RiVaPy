from rivapy.tools import SimpleSchedule
import rivapy.tools.interfaces as interfaces

class EnergyFutureSpecifications(interfaces.FactoryObject):
    def __init__(self, schedule:SimpleSchedule, price:float, name:str) -> None:
        self.schedule = schedule
        self.price = price
        self.name = name

    def get_schedule(self):
        return self.schedule.get_schedule()
    
    def get_price(self):
        return self.price
    
    def get_start(self):
        return self.schedule.start#get_schedule()[0]
    
    def get_end(self):
        return self.schedule.end#get_schedule()[-1]
    
    def get_start_end(self):
        return (self.get_start(), self.get_end())

    def _to_dict(self)->dict:
        self.to_dict()

    

