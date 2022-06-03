from typing import TYPE_CHECKING

from SDL_FSM.base_types import FSM_STATES


from .transition import Transition
from .event_handler import Event_Handler


class FSM_base:
    _current_state = None
    _is_handling_event = False
    _invalidation_reason = None
    _transition_running = None

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

    def _assert_can_transition(self, src):
        if self._transition_running is not None:
            return False
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
                # noinspection PyProtectedMember
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
