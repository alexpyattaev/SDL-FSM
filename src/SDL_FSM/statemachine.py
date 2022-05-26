import copy
import dataclasses
import enum
from copy import deepcopy
from enum import Enum
from types import MethodType
from typing import Callable, Optional, Sequence, Any, MutableSequence

CALLABLE = "CALLABLE"


class FSM_STATES(str, enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class FSM_base:
    _current_state = None
    _is_handling_event = False
    _is_valid = True

    class states(FSM_STATES):
        pass

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

    def make_networkx_graph(self):
        import networkx as nx
        g = nx.MultiDiGraph()
        g.add_nodes_from(self.states.__members__.keys())
        for n, t in self.transitions.items():
            g.add_edge(t.src.name, t.dst.name, label=n, effects=t.side_effects)
        return g

    def make_dot_graph(self, name:str=None):
        import pydot
        if name is None:
            name = self.__class__.__name__
        graph = pydot.Dot(name, graph_type='digraph')
        graph.prog = "/usr/bin/dot"
        for k, v in self.states.__members__.items():
            label = f"""<{k}. <BR/> {v.value}>"""
            graph.add_node(pydot.Node(k, label=label))

        for n, t in self.transitions.items():
            label = """<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">"""
            for f in t.side_effects:
                label += f"<TR><TD>{f.__name__}</TD></TR>"
            label += """</TABLE>>"""
            print(label)
            graph.add_edge(pydot.Edge(t.src.name, t.dst.name, label=label, color='blue'))
        return graph


@dataclasses.dataclass(frozen=True, slots=True)
class Transition:
    """
    Abstraction of state transition of FSM
    """
    src: FSM_STATES = None  # source state
    dst: FSM_STATES = None  # destination state
    side_effects: MutableSequence = dataclasses.field(default_factory=list)  # list of side effects (in order)
    frozen: bool = False  # prevents modification in runtime

    _self_effects: MutableSequence = dataclasses.field(default_factory=list)  # internal use, list of not-yet-bound methods
    _fsm_ref: FSM_base = None  # internal

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

        vals = dataclasses.asdict(self)
        vals['_fsm_ref'] = fsm
        side_effects = copy.copy(self.side_effects)
        # bin all self-effect functions
        side_effects.extend(MethodType(e, fsm) for e in self._self_effects)
        vals['_self_effects'] = tuple()
        vals['side_effects'] = tuple(side_effects)if self.frozen else side_effects
        return Transition(**vals)

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
    def __init__(self, fn: Callable, states: set[FSM_STATES], fsm: Optional[FSM_base] = None):
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


def event(state: Optional[FSM_STATES] = None, states: Sequence[FSM_STATES] = tuple()):
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



