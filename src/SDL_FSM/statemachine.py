import dataclasses
from copy import deepcopy
from enum import Enum
from types import MethodType
from typing import Callable, Optional, Sequence, Any


class _MetaTransition(type):
    def __getitem__(self, item):
        print(f"meta: {item=}")
        inst = self()
        if item is None or item is not CALLABLE:
            del item.__call__

        return self


class STATE:
    pass


CALLABLE = "CALLABLE"


class FSM_base:
    _current_state = None
    _is_handling_event = False
    _is_valid = True

    @property
    def valid(self):
        return self._is_valid

    @property
    def state(self):
        return self._current_state

    @state.setter
    def state(self, v):
        print(f"Transitioning {self._current_state}-> {v}")
        self._current_state = v

    def _can_transition(self, src):
        if not self._is_valid:
            raise RuntimeError("Invalid FSM!")

        if src != self._current_state:
            raise RuntimeError("Invalid source state")

    def __init__(self):
        self.transitions = {}
        self.events = {}
        for i in dir(self):
            v = getattr(self, i)
            if isinstance(v, Transition):
                tr = v._bind(self)
                self.transitions[i] = tr
                setattr(self, i, tr)
            if isinstance(v, Event_Handler):
                self.events[i] = v

    def make_graph(self):
        import networkx as nx
        g = nx.DiGraph


@dataclasses.dataclass
class Transition():  # metaclass=_MetaTransition):
    """
    Abstraction of state transition of FSM
    """
    src: STATE = None  # source state
    dst: STATE = None  # destination state
    side_effects: list = dataclasses.field(default_factory=list)  # list of side effects (in order)
    _self_effects: list = dataclasses.field(default_factory=list) # internal
    _fsm_ref: FSM_base = None # internal

    def side_effect(self, f: Callable):
        """Registers callable f as side effect of transition. f can not be a method on the FSM itself.
        :returns Transition instance
        """
        self.side_effects.append(f)
        return self

    def self_effect(self, m: Callable):
        """
        Registers class method m as side effect of transition.
        :param m: method to register. m must be a method on the FSM class (i.e. not bound).
        :returns Transition instance
        """
        assert self._fsm_ref is None, "Can not register self_effect after Transition is bound to FSM"
        self._self_effects.append(m)
        return self

    def _bind(self, fsm: FSM_base):
        """Binds Transition to fsm.
         Creates a copy of Transition instance such that each FSM can have its own customized Transitions.
         Also binds all self_effects. """
        x = deepcopy(self)
        x._fsm_ref = fsm
        for e in x._self_effects:
            # bin all self-effect functions
            x.side_effects.append(MethodType(e, fsm))
        return x

    # noinspection PyProtectedMember
    def __call__(self, *args, **kwargs) -> tuple[Any]:
        """
        Allows Transitions to be called as events, thus forcing FSM to switch states.

        Use in cases where events would be overkill for your needs.
        :param args: passed to side_effect functions
        :param kwargs: passed to side_effect functions
        :return: tuple of return values from side_effects, in order
        """
        self._fsm_ref._can_transition(self.src)
        try:
            rv = tuple(se(*args, FSM=self._fsm_ref, **kwargs) for se in self.side_effects)
            self._fsm_ref.state = self.dst
            return rv
        except Exception:
            self._fsm_ref._is_valid = False
            raise


class Event_Handler:
    """Actual event handler that gets attached to FSM classes. Acts as a Python Descriptor,
     i.e. it will bind to appropriate FSM entity when called."""
    def __init__(self, fn: Callable, states: set[STATE], fsm: Optional[FSM_base] = None):
        self.states = states
        self.fn = fn
        self.fsm = fsm
        self.name = self.fn.__name__

    def __repr__(self):
        if self.fsm is None:
            return f"<EventHandler {self.name} (unbound) states:{self.states}>"
        else:
            return f"<EventHandler {self.name} of {self.fsm} states:{self.states}>"

    def __get__(self, instance, owner):
        """Returns self, but bound to instance of fsm"""
        assert issubclass(owner, FSM_base)
        self.fsm = instance
        return self

    def check_state(self, fsm: FSM_base):
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


def event(state: Optional[STATE] = None, states: Sequence[STATE] = tuple()):
    """Decorator that turns methods into event handlers.
    state and states arguments will be merged into a single set.

    :param state: state in which this event can be handled
    :param states: sequence of states in which this event can be handled
    :returns a decorator for event
    """
    states = set(states)
    if state is not None:
        states.add(state)

    def wrapper(f: Callable):
        """Wraps method f as event handler"""
        print(f"Creating event from {f}")
        return Event_Handler(f, states)

    return wrapper



