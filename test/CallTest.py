class Person:
    val = 3

    def __init__(self, val1):
        self.val = val1

    def __call__(self, name):
        print("__call__: hello" + name)

    def hello(self, name):
        print("hello" + name)


person = Person(5)
person("Alice")
person.hello("Bob")
print(person.val)
