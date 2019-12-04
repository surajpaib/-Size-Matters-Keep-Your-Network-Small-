from abc import ABCMeta, abstractmethod

class BaseClass(metaclass=ABCMeta):
    @abstractmethod
    def set_model(self, model):
        pass
        
    @abstractmethod
    def set_optimizer(self, optim):
        pass

    @abstractmethod
    def set_optimizer_model(self, optim, model):
        pass

    @abstractmethod
    def set_percentage(self, perc):
        pass

    @abstractmethod
    def define_strategy(self):
        pass

    @abstractmethod
    def apply_strategy(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_optimizer(self):
        pass

    @abstractmethod
    def get_optimizer_model(self):
        pass



    
if __name__ == "__main__":
    p = BaseClass()
    p.get_graph()