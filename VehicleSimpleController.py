import numpy as np

class VehicleSimpleController :
    def __init__ (self,**kwargs) :
        self.k1 = kwargs.get('k1', 1)
        self.k2 = kwargs.get('k2', 1)
        self.lf = kwargs.get('lf', 1)
        self.lr = kwargs.get('lr', 1)
    def control(self, t, x) :
        u = np.zeros(2)
        u[0] = np.clip(self.k1*(np.sqrt(x[0]**2 + x[1]**2)) - self.k2*x[3],-20,20)
        u[1] = np.clip(np.arctan2(np.sin(-x[2]+np.pi+np.arctan2(x[1],x[0])),self.lr), -np.pi/3, np.pi/3)
        return u
