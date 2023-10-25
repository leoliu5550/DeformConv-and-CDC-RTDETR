from reg import Registers

class Model:
    pass

# @是装饰器，等价于Registers.model.register(Model)
@Registers.model.register
class Model1(Model):
    pass

@Registers.model.register
class Model2(Model):
    pass

@Registers.model.register
class Model3(Model):
    pass
