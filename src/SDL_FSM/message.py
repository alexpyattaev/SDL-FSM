from typing import Callable, Optional

from .statemachine import FSM_base


class Message:
    """Message instance attached to an FSM instance."""
    def __init__(self,  payload=None, fn: Callable = None):
        self.fn = fn
        self.payload = payload
        self.fsm: Optional[FSM_base] = None
        if self.fn is not None:
            self.name = self.fn.__name__

    def __set_name__(self, owner, name):
        print("set_name", owner, name)
        self.name = name

    def __repr__(self):
        if self.fsm is None:
            return f"<Message {self.name} (unbound) >"
        else:
            return f"<Message {self.name} from {self.fsm}>"

    def __get__(self, instance, owner):
        """Returns self, but bound to instance of fsm"""
        assert issubclass(owner, FSM_base)
        self.fsm = instance
        return self

    def __call__(self, *args, **kwargs):
        """Produce the actual message proper"""
        assert self.fsm is not None
        if self.payload is not None:
            return self.payload
        else:
            return self.fn(self.fsm)

    def subscribe(self, *args, **kwargs):
        pass

    def unsubscribe(self, *args, **kwargs):
        pass


def message(f: Callable):
    """Decorator that turns methods into message sources.
    :returns a decorator for method
    """
    return Message(fn=f)




