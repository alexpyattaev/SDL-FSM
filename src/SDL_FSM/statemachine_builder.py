from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from random import random
from typing import Dict, Generic, ItemsView, List, Literal, NamedTuple, Optional, Set, Tuple, Type, TypeVar, Callable, Iterable, Union, overload
from enum import Enum, IntEnum


# TODO: Rename all internal classes to start with "_"
StatesEnum = TypeVar('StatesEnum', bound=IntEnum)  # Must start with 0 and be sequential (validated in constructor)
InputsEnum = TypeVar('InputsEnum', bound=IntEnum)
OutputsEnum = TypeVar('OutputsEnum', bound=IntEnum)
DataType = TypeVar('DataType')
# TODO: Refactor to aliases
# FsmDef = FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]


class NoOutputs(IntEnum):
    pass


class DefinitionError(Exception): 
    pass
class TriggerError(Exception):
    pass


class FsmDefinition(Generic[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, states: Type[StatesEnum], inputs: Type[InputsEnum], outputs: Type[OutputsEnum] = NoOutputs, \
            data: Optional[Type[DataType]] = None, initial_state: Optional[StatesEnum] = None, name: Optional[str] = None, \
            auto_finalize: bool = True, raise_on_invalid_input: bool = True) -> None:
        if self._validate_enum(states) is False:
            raise RuntimeError()
        self._states_type: Type[StatesEnum] = states
        if self._validate_enum(inputs) is False:
            raise RuntimeError()
        self._inputs_type: Type[InputsEnum] = inputs
        if outputs is not None and self._validate_enum(outputs) is False:
            raise RuntimeError()
        self._outputs_type: Type[OutputsEnum] = outputs
        self._data_type: Optional[Type[DataType]] = data

        self._inputs_in_state: List[Set[InputsEnum]] = [set() for _ in self._inputs_type]
        self._input_expression: List[Optional[Expression]] = [None for _ in self._inputs_type]
        self._input_expression_runner: List[Optional[ExpressionRunner]] = [None for _ in self._inputs_type]
        self._output_callbacks: List[Set[Callable]] = [set() for _ in self._outputs_type]

        self.initial_state: Optional[StatesEnum] = initial_state
        
        self._name: Optional[str] = name

        self._auto_finalize: bool = auto_finalize
        self._raise_on_invalid_input: bool = raise_on_invalid_input

        self._ready: bool = False

    @property
    def name(self) -> Optional[str]:
        return self._name

    # TODO: repr

    @property
    def states(self) -> Type[StatesEnum]:
        return self._states_type

    @property
    def inputs(self) -> Type[InputsEnum]:
        return self._inputs_type

    @property
    def outputs(self) -> Type[OutputsEnum]:
        return self._outputs_type

    def _validate_enum(self, enum: Type[IntEnum]) -> bool:
        for i, val in enumerate(enum):
            if i != val:
                return False
        return True
    
    def set_state(self, state: StatesEnum, inputs: Iterable[InputsEnum]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        if self._ready is True:
            raise DefinitionError(f"Cannot update a finalized Fsm Definition {self}")

        inputs_in_state = self._inputs_in_state[state]
        if len(inputs_in_state) > 0:
            inputs_in_state.clear()
        inputs_in_state.update(inputs)

        return self

    @overload
    def add_state_input(self, state: StatesEnum, input: InputsEnum) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...    
    @overload
    def add_state_input(self, state: StatesEnum, input: Iterable[InputsEnum]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    def add_state_input(self, state: StatesEnum, input: Union[InputsEnum, Iterable[InputsEnum]]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        if self._ready is True:
            raise DefinitionError(f"Cannot update a finalized Fsm Definition {self}")

        inputs_in_state = self._inputs_in_state[state]
        if isinstance(input, Iterable):
            inputs_in_state.update(input)
        else:
            inputs_in_state.add(input)
        
        return self

    @overload
    def remove_state_input(self, state: StatesEnum, input: InputsEnum) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    @overload
    def remove_state_input(self, state: StatesEnum, input: Iterable[InputsEnum]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    def remove_state_input(self, state: StatesEnum, input: Union[InputsEnum, Iterable[InputsEnum]]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        if self._ready is True:
            raise DefinitionError(f"Cannot update a finalized Fsm Definition {self}")

        inputs_in_state = self._inputs_in_state[state]
        if isinstance(input, Iterable):
            inputs_in_state.difference_update(input)
        else:
            inputs_in_state.discard(input)

        return self

    def set_input(self, input: InputsEnum, expr: Expression) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        if self._ready is True:
            raise DefinitionError(f"Cannot update a finalized Fsm Definition {self}")
        self._input_expression[input] = expr

        return self

    def set_output(self, output: OutputsEnum, callbacks: Iterable[Callable[[FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]], None]]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        output_callbacks = self._output_callbacks[output]
        if len(output_callbacks) > 0:
            output_callbacks.clear()
        output_callbacks.update(callbacks)

        return self

    @overload
    def add_output_callback(self, output: OutputsEnum, callback: Callable[[FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]], None]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    @overload
    def add_output_callback(self, output: OutputsEnum, callback: Iterable[Callable[[FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]], None]]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    def add_output_callback(self, output: OutputsEnum, callback: Union[Callable[[FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]], None], Iterable[Callable[[FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]], None]]]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        output_callbacks = self._output_callbacks[output]
        if isinstance(callback, Iterable):
            output_callbacks.update(callback)
        else:
            output_callbacks.add(callback)
        
        return self

    @overload
    def remove_output_callback(self, output: OutputsEnum, callback: Callable[[FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]], None]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    @overload
    def remove_output_callback(self, output: OutputsEnum, callback: Iterable[Callable[[FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]], None]]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    def remove_output_callback(self, output: OutputsEnum, callback: Union[Callable[[FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]], None], Iterable[Callable[[FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]], None]]]) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        output_callbacks = self._output_callbacks[output]
        if isinstance(callback, Iterable):
            output_callbacks.difference_update(callback)
        else:
            output_callbacks.discard(callback)
        
        return self

    def finalize(self) -> None:
        if self._ready is True:
            raise DefinitionError(f"Cannot finalize an already finalized Fsm Definition {self}")

        named_expressions: Dict[InputsEnum, Expression] = {}
        for input in self._inputs_type:
            expression = self._input_expression[input]
            if expression is None:
                raise DefinitionError(f"Cannot finalize Fsm Definition {self}, as Input {input} is not defined")
            else:
                named_expressions[input] = expression
        
        builder = CallableExpressionBuilder(self._states_type, self._inputs_type, self._outputs_type, self._data_type, named_expressions)
        try:
            builder.build()
        except ExpressionBuildError as err:
            raise DefinitionError(f"Cannot finalize Fsm Definition {self}, due to build error: {err}")
        
        named_expression_runners = builder.get_named_expression_runners()
        names: Set[InputsEnum] = set()
        for name, expression_runner in named_expression_runners:
            if name in names:
                raise DefinitionError(f"Cannot finalize Fsm Definition {self}, as duplicate Input {name} encountered after building expressions")
            else:
                names.add(name)
            
            self._input_expression_runner[name] = expression_runner
        
        if len(names ^ set(self._inputs_type)) > 0:
            self._input_expression_runner = [None for _ in self._inputs_type]

            raise DefinitionError(f"Cannot finalize Fsm Definition {self}, as built Inputs {names} do not coinside with defined Inputs {self._inputs_type}")

        self._ready = True

    def _trigger_output(self, fsm_implementation: FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType], output: OutputsEnum) -> None:
        for callback in self._output_callbacks[output]:
            callback(fsm_implementation)
        for callback in fsm_implementation._output_callbacks[output]:
            callback(fsm_implementation)

    def _trigger_state(self, fsm_implementation: FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType], state: Optional[StatesEnum]) -> None:
        if state is not None:
            fsm_implementation._state = state

    def _trigger_input(self, fsm_implementation: FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType], input: InputsEnum) -> bool:
        # TODO: Handle busy correctly, i.e. add queueable property or immediate or throw
        if fsm_implementation._busy is True:
            raise RuntimeError()
        fsm_implementation._busy = True

        if input not in self._inputs_in_state[fsm_implementation._state]:
            if self._raise_on_invalid_input is True:
                raise TriggerError()
            else:
                fsm_implementation._busy = False
                return False
        
        old_state = fsm_implementation._state

        expression_runner = self._input_expression_runner[input]
        # if expression_runner is None:
        #     raise RuntimeError()
        fsm_controller = FsmController(self, fsm_implementation)
        expression_runner(fsm_implementation._data, fsm_controller)  # type:ignore
        fsm_controller.invalidate()

        fsm_implementation._busy = False
        if fsm_implementation._state == old_state:
            return False
        else:
            return True

    def implement(self, initial_state: Optional[StatesEnum] = None, name: Optional[str] = None) -> FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        if self._ready is False:
            if self._auto_finalize is True:
                self.finalize()
            else:
                raise DefinitionError(f"Cannot implement a not finalized Fsm Definition {self}")

        if initial_state is not None:
            impl_state: StatesEnum = initial_state
        elif self.initial_state is not None:
            impl_state = self.initial_state
        else:
            raise DefinitionError(f"Cannot implement Fsm Definition {self} without an initial state")
        if name is not None:
            impl_name: str = name
        elif self._name is not None:
            impl_name = self._name
        else:
            raise DefinitionError(f"Cannot implement Fsm Definition {self} without a name")
        fsm_implementation = FsmImplementation(self, \
            self._states_type, self._inputs_type, self._outputs_type, \
            self._data_type, impl_state, impl_name)

        return fsm_implementation


class FsmImplementation(Generic[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, fsm_definition: FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType], \
            states: Type[StatesEnum], inputs: Type[InputsEnum], outputs: Type[OutputsEnum], \
            data: Optional[Type[DataType]], initial_state: StatesEnum, name: Optional[str] = None) -> None:
        self._fsm_definition: FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType] = fsm_definition
        
        self._state: StatesEnum = initial_state
        self._output_callbacks: List[Set[Callable]] = [set() for _ in outputs]
        if data is None:
            self._data: Optional[DataType] = None
        else:
            self._data = data()

        self._busy: bool = False

        self._name: Optional[str] = name

    @property
    def name(self) -> Optional[str]:
        return self._name

    # TODO: repr

    @property
    def state(self) -> StatesEnum:
        return self._state

    @property
    def busy(self) -> bool:
        return self._busy

    def get_definition(self) -> FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        return self._fsm_definition

    def set_output(self, output: OutputsEnum, callbacks: Iterable[Callable]) -> FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        output_callbacks = self._output_callbacks[output]
        if len(output_callbacks) > 0:
            output_callbacks.clear()
        output_callbacks.update(callbacks)

        return self

    @overload
    def add_output_callback(self, output: OutputsEnum, callback: Callable) -> FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    @overload
    def add_output_callback(self, output: OutputsEnum, callback: Iterable[Callable]) -> FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    def add_output_callback(self, output: OutputsEnum, callback: Union[Callable, Iterable[Callable]]) -> FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        output_callbacks = self._output_callbacks[output]
        if isinstance(callback, Iterable):
            output_callbacks.update(callback)
        else:
            output_callbacks.add(callback)
        
        return self

    @overload
    def remove_output_callback(self, output: OutputsEnum, callback: Callable) -> FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    @overload
    def remove_output_callback(self, output: OutputsEnum, callback: Iterable[Callable]) -> FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        ...
    def remove_output_callback(self, output: OutputsEnum, callback: Union[Callable, Iterable[Callable]]) -> FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        output_callbacks = self._output_callbacks[output]
        if isinstance(callback, Iterable):
            output_callbacks.difference_update(callback)
        else:
            output_callbacks.discard(callback)
        
        return self

    def trigger(self, input: InputsEnum) -> bool:
        return self._fsm_definition._trigger_input(self, input)


class ExpressionType(IntEnum):
    Default = 0
    Action = 1
    Output = 2
    State = 3
    Condition = 4
    Label = 5
    Jump = 6


class Expression(ABC, Generic[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, expression_type: ExpressionType) -> None:
        self._type: ExpressionType = expression_type

    @property
    def type(self) -> ExpressionType:
        return self._type
        

class ChainExpression(Expression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, expression_type: ExpressionType) -> None:
        self.next: Optional[Expression[StatesEnum, InputsEnum, OutputsEnum, DataType]] = None
        self._tail: Optional[ChainExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]] = None

        super().__init__(expression_type)

    def _extend_tail(self, expression: ChainExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
        # TODO: Somehow forbid intermediate next and tail assignment
        if self._tail is None:
            if self.next is not None:
                raise RuntimeError() # TODO: Some other kind of exception
            
            self.next = expression
            self._tail = expression
        else:
            if self._tail.next is not None:
                raise RuntimeError()
            
            self._tail.next = expression
            self._tail = expression

    def _finish_tail(self, expression: Expression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
        if self._tail is None:
            if self.next is not None:
                raise RuntimeError()
            
            self.next = expression
        else:
            if self._tail.next is not None:
                raise RuntimeError()
            
            self._tail.next = expression

    def Action(self, callable: Callable[[DataType], None]) -> ChainExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        self._extend_tail(Action(callable))

        return self
    
    def Output(self, output: OutputsEnum) -> ChainExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        self._extend_tail(Output(output))

        return self

    def Label(self, label: str) -> ChainExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        self._extend_tail(Label(label))

        return self

    def Condition(self, condition: Callable[[DataType], bool], next_true: Expression[StatesEnum, InputsEnum, OutputsEnum, DataType], next_false: Expression[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> ChainExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]:
        self._extend_tail(Condition(condition, next_true, next_false))
        
        return self

    def State(self, state: Optional[StatesEnum]) -> Expression:
        self._finish_tail(State(state))

        return self

    def Jump(self, target: str) -> Expression:
        self._finish_tail(Jump(target))

        return self


class Action(ChainExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    # TODO: Add parameters to this (and to Fsm trigger)
    def __init__(self, callable: Callable[[DataType], None]) -> None:
        super().__init__(expression_type=ExpressionType.Action)

        self.callable: Callable[[DataType], None] = callable
        self.next: Optional[Expression[StatesEnum, InputsEnum, OutputsEnum, DataType]] = None


class Output(ChainExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, output: OutputsEnum) -> None:
        super().__init__(expression_type=ExpressionType.Output)

        self.output: OutputsEnum = output
        self.next: Optional[Expression] = None


class Label(ChainExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, label: str) -> None:
        super().__init__(expression_type=ExpressionType.Label)

        self.label: str = label
        self.next: Optional[Expression[StatesEnum, InputsEnum, OutputsEnum, DataType]] = None


class Condition(ChainExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, condition: Callable[[DataType], bool], next_true: Expression[StatesEnum, InputsEnum, OutputsEnum, DataType], next_false: Expression[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> None:
        super().__init__(expression_type=ExpressionType.Condition)

        self.condition: Callable[[DataType], bool] = condition
        self.next_true: Expression[StatesEnum, InputsEnum, OutputsEnum, DataType] = next_true
        self.next_false: Expression[StatesEnum, InputsEnum, OutputsEnum, DataType] = next_false


# TODO: Decide class
# .Decide(some_callable, 
# {
#     1: bla,
#     2: foo,
#     3: boo,
# }
# )


class Jump(Expression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, target: str) -> None:
        super().__init__(expression_type=ExpressionType.Jump)

        self.target: str = target


class State(Expression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, state: Optional[StatesEnum]) -> None:
        super().__init__(expression_type=ExpressionType.State)

        self.state: Optional[StatesEnum] = state


class FsmController(Generic[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, fsm_definition: FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType], fsm_implementation: FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> None:
        self._fsm_definition: FsmDefinition[StatesEnum, InputsEnum, OutputsEnum, DataType] = fsm_definition
        self._fsm_implementation: FsmImplementation[StatesEnum, InputsEnum, OutputsEnum, DataType] = fsm_implementation
        self._valid = True

    # TODO: repr

    def trigger_output(self, output: OutputsEnum) -> None:
        if self._valid is True:
            self._fsm_definition._trigger_output(self._fsm_implementation, output)
        else:
            raise TriggerError(f"Fsm Controller {self} is invalid")

    def trigger_state(self, state: Optional[StatesEnum]) -> None:
        if self._valid is True:
            self._fsm_definition._trigger_state(self._fsm_implementation, state)
        else:
            raise TriggerError(f"Fsm Controller {self} is invalid")

    def invalidate(self) -> None:
        self._valid = False


class ExpressionRunner(ABC, Generic[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(self, data: DataType, fsm_controller: FsmController[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> None:
        pass


class ExpressionBuilder(ABC, Generic[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    @abstractmethod
    def __init__(self, states: Type[StatesEnum], inputs: Type[InputsEnum], outputs: Type[OutputsEnum], data: Optional[Type[DataType]], \
            named_expressions: Dict[InputsEnum, Expression]) -> None:
        pass

    @abstractmethod
    def build(self) -> bool:
        pass

    @abstractmethod
    def get_named_expression_runners(self) -> ItemsView[InputsEnum, ExpressionRunner]:
        pass


class ExpressionError(Exception):
    pass
class ExpressionBuildError(Exception):
    pass


class CallableExpressionPoint(NamedTuple):
    previous_callable_expression: Optional[CallableExpression]
    next_id: int
class ExpressionPoint(NamedTuple):
    previous_callable_expression_point: CallableExpressionPoint
    current_expression: Expression


class CallableExpressionBuilder(ExpressionBuilder[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, states: Type[StatesEnum], inputs: Type[InputsEnum], outputs: Type[OutputsEnum], data: Optional[Type[DataType]], \
            named_expressions: Dict[InputsEnum, Expression]) -> None:
        self._named_expressions: Dict[InputsEnum, Expression] = named_expressions
        self._named_expression_runners: Dict[InputsEnum, ExpressionRunner[StatesEnum, InputsEnum, OutputsEnum, DataType]] = {}

        self._labels: Dict[str, CallableExpression] = {}
        """Label name to built expression mapping. Collects all labels as they are met traversing raw expressions."""
        self._jumps: Dict[CallableExpressionPoint, str] = {}
        """Built expressions mapped to their next jump name that is not known yet. Will be connected in final pass."""
        self._shortcuts: Dict[InputsEnum, str] = {}
        """Stores names of raw expressions that have a jump as their head. Will be connected in final pass."""

        self._current_name: Optional[InputsEnum] = None
        self._pending_label: Optional[str] = None

    def _connect_callable_expressions(self, previous_callable_expression_point: CallableExpressionPoint, current_callable_expression: CallableExpression):
        if previous_callable_expression_point.previous_callable_expression is not None:
            if isinstance(previous_callable_expression_point.previous_callable_expression, (CallableAction, CallableOutput, CallableState)):
                previous_callable_expression_point.previous_callable_expression.next = current_callable_expression  # type:ignore
            elif isinstance(previous_callable_expression_point.previous_callable_expression, CallableCondition):
                if previous_callable_expression_point.next_id == 0:
                    previous_callable_expression_point.previous_callable_expression.next_true = current_callable_expression
                else:
                    previous_callable_expression_point.previous_callable_expression.next_false = current_callable_expression
            else:
                raise ExpressionError(f"Cannot assign next expression to callable expression {previous_callable_expression_point.previous_callable_expression}.")         
        else:
            if self._current_name is None:
                raise RuntimeError()
            self._named_expression_runners[self._current_name] = CallableExpressionRunner(current_callable_expression)

    def _build_expression(self, previous_callable_expression_point: CallableExpressionPoint, expression: Expression) -> Optional[CallableExpression]:
        if isinstance(expression, Label):
            if self._pending_label is not None:
                raise ExpressionError("Expressions cannot have multiple labels")
            if expression.label in self._labels:
                raise ExpressionError(f"Duplicate label {expression}")

            self._pending_label = expression.label

            return None
        elif isinstance(expression, Jump):
            if self._pending_label is not None:
                raise ExpressionError(f"Jump {expression} cannot be labeled")

            if previous_callable_expression_point.previous_callable_expression is None:
                if self._current_name is None:
                    raise RuntimeError()
                self._shortcuts[self._current_name] = expression.target

                return None
            else:
                try:
                    return self._labels[expression.target]
                except KeyError:
                    self._jumps[previous_callable_expression_point] = expression.target

                    return None

        if isinstance(expression, Action):
            callable_expression = CallableAction(expression.callable)
        elif isinstance(expression, Output):
            callable_expression = CallableOutput(expression.output)
        elif isinstance(expression, State):
            callable_expression = CallableState(expression.state)
        elif isinstance(expression, Condition):
            callable_expression = CallableCondition(expression.condition)
        else:
            raise ExpressionError(f"Unsupported type of expression {expression}")
        
        if self._pending_label is not None:
            self._labels[self._pending_label] = callable_expression
            self._pending_label = None

        return callable_expression

    def build(self):
        try:
            for name, head_expression in self._named_expressions.items():
                if head_expression is None:
                    raise ExpressionBuildError(f"Expression {name} is not defined")

                self._current_name = name
                self._pending_label = None

                visited_expressions = set((head_expression, ))
                def check_visited(expression: Expression):
                    if expression in visited_expressions:
                        raise ExpressionError(f"Expression loop detected at {expression}")
                    else:
                        visited_expressions.add(expression)
                to_visit_stack = [ExpressionPoint(CallableExpressionPoint(None, 0), head_expression)]

                while to_visit_stack:
                    expression_point = to_visit_stack.pop()
                    previous_callable_expression_point, current_expression = expression_point
                    
                    current_callable_expression = self._build_expression(previous_callable_expression_point, current_expression)
                    if current_callable_expression is not None:
                        self._connect_callable_expressions(previous_callable_expression_point, current_callable_expression)
                    
                    if isinstance(current_expression, (Action, Output)):
                        next_expression = current_expression.next
                        if next_expression is None:
                            raise ExpressionBuildError(f"Missing next expression for expression {current_expression}")
                        check_visited(next_expression)
                        to_visit_stack.append(ExpressionPoint(CallableExpressionPoint(current_callable_expression, 0), next_expression))
                    elif isinstance(current_expression, Label):
                        next_expression = current_expression.next
                        if next_expression is None:
                            raise ExpressionBuildError(f"Missing next expression for expression {current_expression}")
                        check_visited(next_expression)
                        to_visit_stack.append(ExpressionPoint(previous_callable_expression_point, next_expression))
                    elif isinstance(current_expression, Condition):
                        next_expression = current_expression.next_true
                        if next_expression is None:
                            raise ExpressionBuildError(f"Missing next expression for expression {current_expression}")
                        check_visited(next_expression)
                        to_visit_stack.append(ExpressionPoint(CallableExpressionPoint(current_callable_expression, 0), next_expression))
                        next_expression = current_expression.next_false
                        if next_expression is None:
                            raise ExpressionBuildError(f"Missing next expression for expression {current_expression}")
                        check_visited(next_expression)
                        to_visit_stack.append(ExpressionPoint(CallableExpressionPoint(current_callable_expression, 1), next_expression))
                    elif isinstance(current_expression, (State, Jump)):
                        if self._pending_label is not None:
                            raise ExpressionError(f"Hanging label {self._pending_label}")
                    else:
                        raise ExpressionError(f"Unsupported type of expression {current_expression}")

            for name, target in self._shortcuts.items():
                try:
                    resolved_target = self._labels[target]
                except KeyError:
                    raise ExpressionError(f"Unable to resolve label {target}")
                self._named_expression_runners[name] = CallableExpressionRunner(resolved_target)
            for callable_expression_point, target in self._jumps.items():
                try:
                    resolved_target = self._labels[target]
                except KeyError:
                    raise ExpressionError(f"Unable to resolve label {target}")
                self._connect_callable_expressions(callable_expression_point, resolved_target)
        except ExpressionError as error:
            raise ExpressionBuildError(f"Error while building expression {name}: {error}")  # TODO: name can be unbound error

    def get_named_expression_runners(self) -> ItemsView[InputsEnum, ExpressionRunner[StatesEnum, InputsEnum, OutputsEnum, DataType]]:
        return self._named_expression_runners.items()


# TODO: Making a Jump Table with arg unpack is faster a little bit
class CallableExpressionRunner(ExpressionRunner[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, expr: CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> None:
        self.chain: CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType] = expr
    
    def __call__(self, data: DataType, fsm_controller: FsmController[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> None:
        cur_expr = self.chain
        while cur_expr is not None:  
            cur_expr = cur_expr(data, fsm_controller)


class CallableExpression(ABC, Generic[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    @abstractmethod
    def __call__(self, data: DataType, fsm_controller: FsmController[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> Optional[CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]]:
        pass


class CallableAction(CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, callable: Callable[[DataType], None]) -> None:
        self._callable: Callable[[DataType], None] = callable
        self.next: Optional[CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]] = None

    def __call__(self, data: DataType, fsm_controller: FsmController[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> Optional[CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]]:
        self._callable(data)

        return self.next

    
class CallableOutput(CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, output: OutputsEnum) -> None:
        self._output: OutputsEnum = output
        self.next: Optional[CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]] = None

    def __call__(self, data: DataType, fsm_controller: FsmController[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> Optional[CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]]:
        fsm_controller.trigger_output(self._output)
        
        return self.next


class CallableState(CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, state: Optional[StatesEnum]) -> None:
        self._state: Optional[StatesEnum] = state
 
    def __call__(self, data: DataType, fsm_controller: FsmController[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> Optional[CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]]:
        fsm_controller.trigger_state(self._state)

        return None


class CallableCondition(CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]):
    def __init__(self, condition: Callable[[DataType], bool]) -> None:
        self._condition: Callable[[DataType], bool] = condition
        self.next_true: Optional[CallableExpression] = None
        self.next_false: Optional[CallableExpression] = None
    
    def __call__(self, data: DataType, fsm_controller: FsmController[StatesEnum, InputsEnum, OutputsEnum, DataType]) -> Optional[CallableExpression[StatesEnum, InputsEnum, OutputsEnum, DataType]]:
        if self._condition(data) is True:
            return self.next_true
        else:
            return self.next_false


# ------------------------------------------
# ------------------------------------------
# ------------------------------------------

class PandaState(IntEnum):
    Sleep = 0
    Awake = 1
    Happy = 2

class PandaInput(IntEnum):
    WakeUp = 0
    Feed = 1
    Rest = 2

class PandaOutput(IntEnum):
    Poop = 0

@dataclass
class PandaData:
    happiness: int = 0

def output_print(panda_impl: FsmImplementation[PandaState, PandaInput, PandaOutput, PandaData], output: PandaOutput):
    print(f"Output print: {output} in state {panda_impl.state}")

def action_print(data: PandaData, msg: str):
    print(f"Action print: {msg}")

def random_condition(data: PandaData):
    if random() > 0.5:
        return True
    else:
        return False

def make_happier(data: PandaData, val: int):
    data.happiness += val

def happiness_condition(data: PandaData):
    if data.happiness > 10:
        return True
    else:
        return False


panda_definition = FsmDefinition(PandaState, PandaInput, PandaOutput, PandaData, PandaState.Awake, "Panda")
panda_definition \
    .set_state(PandaState.Sleep, [PandaInput.WakeUp, ]) \
    .set_state(PandaState.Awake, [PandaInput.Feed, ]) \
    .set_state(PandaState.Happy, [PandaInput.Rest, ]) \
    .set_input(PandaInput.WakeUp,
        Action(partial(action_print, msg="WakeUp"))\
        .State(PandaState.Awake)
    ) \
    .set_input(PandaInput.Feed,
        Action(partial(action_print, msg="Feed"))
        .Action(partial(make_happier, val=3))
        .Condition(
            random_condition
            ,
            Action(partial(action_print, msg="Random True")) \
            .State(None)
            ,
            Action(partial(action_print, msg="Random False")) \
            .State(PandaState.Happy)
        )
    ) \
    .set_input(PandaInput.Rest, 
        Action(partial(action_print, msg="Rest"))\
        .Condition(
            happiness_condition
            ,
            Action(partial(action_print, msg="HAPPY!"))\
            .Output(PandaOutput.Poop)\
            .Jump("something")
            ,
            Action(partial(action_print, msg="Not happy :<"))\
            .Label("something")\
            .State(PandaState.Sleep)
        )
        # Action(partial(action_print, "Rest"))\
        # .Condition(
        #     random_condition
        #     ,
        #     Action(partial(action_print, "Random2 True"))\
        #     .Output(PandaOutput.Poop)
        #     ,
        #     Action(partial(action_print, "Random2 False"))
        # )\
        # .State(PandaState.Sleep)
    ) \
    .add_output_callback(PandaOutput.Poop, partial(output_print, output=PandaOutput.Poop))
panda_definition.finalize()

panda_fsm = panda_definition.implement()

print(f"Panda state is {panda_fsm.state.name}")
while panda_fsm.state is PandaState.Awake:
    panda_fsm.trigger(PandaInput.Feed)
    print(f"Panda state is {panda_fsm.state.name}")
panda_fsm.trigger(PandaInput.Rest)
print(f"Panda state is {panda_fsm.state.name}")
