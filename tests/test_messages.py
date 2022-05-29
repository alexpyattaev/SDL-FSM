
from SDL_FSM import FSM_base, FSM_STATES
from SDL_FSM.message import message, Message


def test_binding():
    """
    Tests that the messages are bound to FSM instances when appropriate.
    """
    class Foo(FSM_base):
        class states(FSM_STATES):
            A = "AAAA"

        def __init__(self, name: str):
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
    assert a.poke() == "FIXED"
    assert b.poke() == "FIXED"

    assert a.stuff() == "A"
    assert b.stuff() == "B"
