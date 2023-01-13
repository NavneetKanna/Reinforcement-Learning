from test1 import a


class b(a):
    def __init__(self) -> None:
        print("in b init")
        super().__init__()

    def foo2(self):
        print("foo2")
        print(self)
        print(self.f)
        self.k = 2
        print("in b foo2")
        a.foo1(self)
    
    def step(self):
        print(f"in b step")
        return 4

b().foo2()