
from SDL_FSM import FSM_base, FSM_STATES, Transition, event
from SDL_FSM.message import message, Message


def test_binding():
    class Foo(FSM_base):
        class states(FSM_STATES):
            A = "AAAA"

        def __init__(self, name:str):
            self.state = self.states.A
            self.name = name
            FSM_base.__init__(self)

        def __repr__(self):
            return self.name

        @message
        def stuff(self):
            return self.name

        poke = Message(payload="FIXED")

    a = Foo("A")
    b = Foo("B")
    print(a.poke())
    print(b.poke())

    print(a.stuff())
    print(b.stuff())