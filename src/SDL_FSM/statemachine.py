import dataclasses
from copy import deepcopy
from enum import Enum
from types import MethodType
from typing import Callable, Optional, Sequence

import networkx as nx


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
                tr = v.bind(self)
                self.transitions[i] = tr
                setattr(self, i, tr)
            if isinstance(v, Event_Handler):
                self.events[i] = v

    def make_graph(self):
        g = nx.DiGraph




@dataclasses.dataclass
class Transition():  # metaclass=_MetaTransition):
    src: STATE = None
    dst: STATE = None
    side_effects: list = dataclasses.field(default_factory=list)
    _self_effects: list = dataclasses.field(default_factory=list)
    _fsm_ref: FSM_base = None

    def side_effect(self, f: Callable):
        self.side_effects.append(f)
        return self

    def self_effect(self, f: Callable):
        self._self_effects.append(f)
        return self

    def bind(self, fsm: FSM_base):
        x = deepcopy(self)
        x._fsm_ref = fsm
        for e in x._self_effects:
            # bin all self-effect functions
            x.side_effects.append(MethodType(e, fsm))
        return x

    # noinspection PyProtectedMember
    def __call__(self, *args, **kwargs):
        self._fsm_ref._can_transition(self.src)
        try:
            for se in self.side_effects:
                se(*args, FSM=self._fsm_ref, **kwargs)
            self._fsm_ref.state = self.dst
        except Exception:
            self._fsm_ref._is_valid = False
            raise


def moo_side_effect(FSM: FSM_base, moo_maybe: str = "", **_):
    print(f"MOOOOOO {FSM} {moo_maybe}")


class Event_Handler:
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
    states = set(states)
    if state is not None:
        states.add(state)

    def wrapper(f: Callable):
        print(f"Creating event from {f}")
        return Event_Handler(f, states)

    return wrapper


OBSCURE_EXTERNAL_CONDITION = False


class Panda(FSM_base):
    @dataclasses.dataclass
    class states:
        HUNGRY: STATE = "Hungry"
        ANGRY: STATE = "Angry"
        HAPPY: STATE = "Happy"

    def __init__(self, name: str):
        self.name = name
        self.state = self.states.HUNGRY
        FSM_base.__init__(self)

    def mutating_side_effect(self, time=10, **_):
        print(f"TIMER ARM {time}")

    eat = Transition(src=states.HUNGRY, dst=states.ANGRY).side_effect(moo_side_effect).self_effect(mutating_side_effect)
    shoot = Transition(src=states.ANGRY, dst=states.HAPPY)
    leave = Transition(src=states.HAPPY, dst=states.HUNGRY)

    @event(state=states.HUNGRY)
    def food_sighted(self, **_):
        print(f"{self.name} food_sighted")
        if OBSCURE_EXTERNAL_CONDITION:
            return self.eat
        else:
            return None

    def __repr__(self):
        return self.name

    @event(state=states.ANGRY)
    def target_sighted(self, **_):
        print(f"{self.name} target_sighted")
        if OBSCURE_EXTERNAL_CONDITION:
            return self.shoot
        else:
            return None

    @event(state=states.HAPPY)
    def all_done(self, **_):
        print(f"{self.name} all_done")
        return self.leave


Panda1 = Panda("Arnold")
OBSCURE_EXTERNAL_CONDITION = False
Panda1.food_sighted()
OBSCURE_EXTERNAL_CONDITION = True
Panda1.food_sighted()
Panda1.target_sighted()

Panda2 = Panda("Dolf")
Panda1.leave.side_effect(Panda2.food_sighted)
Panda1.all_done()
Panda2.shoot()
