
class ValidationEpochDivergenceStop():

    def __init__(self, epochs=2, criteria=min):
        self.epochs = epochs
        self.criteria = criteria

        self.previous = list()
    
    def doStop(self, value):
        try:
            if self.criteria(self.previous[0], value) == self.previous[0]: # if value do not meet criteria, increase list
                self.previous.append(value)
            else: # if value do meet criteria, reset with it
                self.previous = [value]
        except:
            self.previous = [value]
        
        return len(self.previous) > self.epochs
