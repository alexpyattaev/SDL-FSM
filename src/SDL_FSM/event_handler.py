from typing import TYPE_CHECKING, Optional, Callable, Sequence, Any
from .base_types import FSM_STATES
from .transition import Transition
if TYPE_CHECKING:
    from .statemachine import FSM_base


class Event_Handler:
    """Actual event handler that gets attached to FSM classes. Acts as a Python Descriptor,
     i.e. it will bind to appropriate FSM entity when called."""
    ReturnType = Optional[Transition]

    def __init__(self, fn: Callable, states: set[FSM_STATES], fsm: Optional['FSM_base'] = None):
        self.states = states
        self.fn = fn
        self.fsm = fsm
        self.__name__ = self.fn.__name__

    def __repr__(self):
        if self.fsm is None:
            return f"<EventHandler {self.__name__} (unbound) states:{self.states}>"
        else:
            return f"<EventHandler {self.__name__} of {self.fsm} states:{self.states}>"

    def __get__(self, instance, owner):
        """Returns self, but bound to instance of fsm"""
        self.fsm = instance
        return self

    def check_state(self, fsm: 'FSM_base'):
        if not self.states:
            return

        if fsm.state not in self.states:
            raise RuntimeError(f"FSM is in incorrect state {fsm.state}, expected one of {self.states}")

    def __call__(self, *args, **kwargs):
        assert self.fsm is not None
        print(f"Handling event name {self.fn.__name__} on {self.fsm}, {args} {kwargs}")
        try:
            self.fsm._is_handling_event = True
            self.check_state(self.fsm)
            trans = self.fn(self.fsm, *args, **kwargs)
            if trans is None:
                return
            assert isinstance(trans, Transition)
            trans()
        finally:
            self.fsm._is_handling_event = False


def event(state: Optional[FSM_STATES] = None, states: Sequence[FSM_STATES] = tuple()) -> Callable[[Callable[..., Any]], Event_Handler]:
    """Decorator that turns methods into event handlers.
    state and states arguments will be merged into a single set.

    :param state: state in which this event can be handled
    :param states: sequence of states in which this event can be handled
    :returns a decorator for event
    """
    states = set(states)
    if state is not None:
        states.add(state)

    def wrapper(f: Callable[['FSM_base', ...], Event_Handler.ReturnType]) -> Event_Handler:
        """Wraps method f as event handler"""
        print(f"Creating event from {f}")
        return Event_Handler(f, states)

    return wrapper
