
class Transition:
    def __init__(self, f):
        print(f"registering {f}")
        self.f = f


def sideeffect(caller=None):
    print(f"Side {caller=}")


class Moo:
    def __init__(self):
        pass

    def selfeffect(self, caller=None, **kwargs):
        print(f"I am moo {self=}, {caller=}")

    def __set_name__(self, owner, name):
        print("set name", self, owner, name)


m = Moo()


class Boo:
    m2 = Moo()

    def __init__(self, moo_inst):
        self.trans3.f = moo_inst.selfeffect

    def selfeffect(self, caller=None, **kwargs):
        print(f"I am boo {self=}, {caller=}")

    trans1 = Transition(selfeffect)
    trans2 = Transition(sideeffect)
    trans3 = Transition(lambda: 1)
    trans4 = Transition(m2.selfeffect)


b = Boo(m)
b.selfeffect()
b.trans1.f(b)
b.trans2.f(b)
b.trans3.f(b)
print("t4")
b.trans4.f(b)