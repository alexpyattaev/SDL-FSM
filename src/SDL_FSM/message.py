import functools
from typing import Callable, Optional

from .statemachine import FSM_base


class Message:
    """Message instance attached to an FSM instance."""
    def __init__(self,  payload=None, fn: Callable = None):
        self.fn = fn
        if fn is not None:
            self.__name__ = fn.__name__
            self.__qualname__ = fn.__qualname__
        self.payload = payload
        self.fsm: Optional[FSM_base] = None
        if self.fn is not None:
            self.name = self.fn.__name__

    # def __set_name__(self, owner, name):
    #     print("set_name", owner, name)
    #     self.__name__ = name

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


def message(f: Callable) -> Message:
    """Decorator that turns methods into message sources.
    :returns a decorator for method
    """
    msg = Message(fn=f)
    #functools.update_wrapper(msg, f, assigned=functools.WRAPPER_ASSIGNMENTS, updated=functools.WRAPPER_UPDATES)

    #'__module__', '__name__', '__qualname__', '__doc__',
    #'__annotations__'
    return msg




