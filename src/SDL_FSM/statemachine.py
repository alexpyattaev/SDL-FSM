import copy
import dataclasses
import enum
from types import MethodType
from typing import Callable, Optional, Sequence, Any, MutableSequence, Union, Self


class FSM_STATES(str, enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class FSM_base:
    _current_state = None
    _is_handling_event = False
    _invalidation_reason = None

    class states(FSM_STATES):
        pass

    def invalidate(self, reason):
        assert reason is not None
        self._invalidation_reason = reason
        # TODO: kill all subscriptions

    @property
    def valid(self):
        return self._invalidation_reason is None

    def __repr__(self):
        return f"<FSM instance {self.__class__.__name__} at {hex(id(self))}>"

    @property
    def state(self):
        return self._current_state

    @state.setter
    def state(self, v):
        print(f"Transitioning {self._current_state}-> {v}")
        self._current_state = v

    def _can_transition(self, src):
        if not self.valid:
            raise RuntimeError(f"Invalid FSM!, reason {self._invalidation_reason}")

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
            g.add_edge(t.src.name, t.dst.name, label=n, effects=t.effects)
        return g

    def make_dot_graph(self, name: str = None):
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
            for f in t.effects:
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
    effects: MutableSequence = dataclasses.field(default_factory=list)  # list of side effects (in order)
    frozen: bool = False  # prevents modification in runtime if set
    _fsm_ref: FSM_base = None  # internal use, reference to FSM instance used during binding

    def exec(self, f: Union[Callable[[FSM_base, ...], Any], Callable[[Self, ...], Any]]):
        """Registers callable f as side effect of transition.

        The callable f can either be a method on the FSM itself,
        or any function that will take FSM reference as first argument
        :param f: f function to register
        :returns Transition instance
        """
        self.effects.append(f)
        return self

    def send(self, msg):
        """
        Send a message msg to all its subscribers
        :param msg:
        :return: Transition instance
        """
        self.effects.append(msg.send)
        return self

    def send_later(self, msg):
        """
        Send msg using asyncio defer logic (call_later).

        This breaks up the call stack, but also allows the receiver to find sending FSM
         already in target state, rather than in transition
        :param msg:
        :return: Transition instance
        """
        self.effects.append(msg.send_async)
        return self

    def _bind(self, fsm: FSM_base):
        """Binds Transition to fsm.
         Creates a copy of Transition instance such that each FSM can have its own customized Transition.
         """
        vals = dataclasses.asdict(self)
        vals['_fsm_ref'] = fsm                        
        vals['effects'] = tuple(self.effects) if self.frozen else copy.copy(self.effects)
        return Transition(**vals)

    # noinspection PyProtectedMember
    def __call__(self, *args, **kwargs) -> None:
        """
        Allows Transitions to be called as events, thus forcing FSM to switch states.

        Use in cases where events would be overkill for your needs.
        :param args: passed to side_effect functions
        :param kwargs: passed to side_effect functions        
        """
        self._fsm_ref._can_transition(self.src)
        try:
            for se in self.effects:
                # effect is a bound method on my own FSM
                if hasattr(se, "__self__") and se.__self__ is self._fsm_ref:
                    se(*args, **kwargs)
                else:  # effect is some other kind of function
                    se(self._fsm_ref, *args,  **kwargs)
            self._fsm_ref.state = self.dst
            
        except Exception as e:
            self._fsm_ref.invalidate(e)
            raise


class Event_Handler:
    """Actual event handler that gets attached to FSM classes. Acts as a Python Descriptor,
     i.e. it will bind to appropriate FSM entity when called."""
    ReturnType = Optional[Transition]

    def __init__(self, fn: Callable, states: set[FSM_STATES], fsm: Optional[FSM_base] = None):
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

    def wrapper(f: Callable[[Self, ...], EvHandler_ReturnType]) -> Event_Handler:
        """Wraps method f as event handler"""
        print(f"Creating event from {f}")
        return Event_Handler(f, states)

    return wrapper
