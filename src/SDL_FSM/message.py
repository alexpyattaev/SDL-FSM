import sys
from typing import Callable, Optional, Any

from .statemachine import FSM_base

if sys.version_info < (3, 11, 0):
    from typing_extensions import Self
else:
    from typing import Self


class Message:
    """Message instance attached to an FSM instance."""
    def __init__(self, payload=None, fn: Callable[[Self], None] = None):
        self.fn = fn
        if fn is not None:
            self.__name__ = fn.__name__
            self.__qualname__ = fn.__qualname__
        self.payload = payload
        self.fsm: Optional[FSM_base] = None
        if self.fn is not None:
            self.name = self.fn.__name__

    def __repr__(self):
        if self.fsm is None:
            return f"<Message {self.__name__} (unbound) >"
        else:
            return f"<Message {self.__name__} from {self.fsm}>"

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

    def send(self):
        pass

    def send_later(self):
        pass

    def subscribe(self, *args, **kwargs):
        pass

    def unsubscribe(self, *args, **kwargs):
        pass


def message(f: Callable[[Self], Any]) -> Message:
    """Decorator that turns methods into message sources.
    :returns a decorator for method
    """
    msg = Message(fn=f)
    return msg

