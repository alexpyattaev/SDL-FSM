import copy
import dataclasses
from typing import MutableSequence, Union, Callable, TYPE_CHECKING, Any
from .base_types import FSM_STATES
import sys

if sys.version_info < (3, 11, 0):
    from typing_extensions import Self
else:
    from typing import Self

if TYPE_CHECKING:
    from .statemachine import FSM_base


@dataclasses.dataclass(frozen=True, slots=True)
class Transition:
    """
    Abstraction of state transition of FSM
    """
    src: FSM_STATES = None  # source state
    dst: FSM_STATES = None  # destination state
    effects: MutableSequence = dataclasses.field(default_factory=list)  # list of side effects (in order)
    frozen: bool = False  # prevents modification in runtime if set
    _fsm_ref: 'FSM_base' = None  # internal use, reference to FSM instance used during binding

    def exec(self, f: Union[Callable[[Self, ...], Any], Callable[['FSM_base', ...], Any]]):
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
        self.effects.append(msg.send_later)
        return self

    def _bind(self, fsm: 'FSM_base'):
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
        self._fsm_ref._assert_can_transition(self.src)
        self._fsm_ref._transition_running = self
        try:
            for se in self.effects:
                # effect is a bound method on my own FSM
                if hasattr(se, "__self__") and se.__self__ is self._fsm_ref:
                    se(*args, **kwargs)
                else:  # effect is some other kind of function
                    se(self._fsm_ref, *args, **kwargs)
            self._fsm_ref.state = self.dst
            self._fsm_ref._transition_running = None
        except Exception as e:
            self._fsm_ref.invalidate(e)
            raise
