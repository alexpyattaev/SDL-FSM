from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from functools import partial
from random import random
from typing import Dict, Generic, ItemsView, List, Literal, NamedTuple, Optional, Set, Type, TypeVar, Callable, Iterable, overload
from enum import Enum


StatesEnum = TypeVar('StatesEnum', bound=Enum)
InputsEnum = TypeVar('InputsEnum', bound=Enum)
OutputsEnum = TypeVar('OutputsEnum', bound=Enum)



# def some_method(cls, x: str) -> str:
#     return f"result {x}"

# class MyMeta(ABCMeta):
#     def __new__(mcs, *args, **kwargs):
#         cls = super().__new__(mcs, *args, **kwargs)
#         cls.some_method = classmethod(some_method)
#         return cls


# class MyABC(ABC):
#     @classmethod
#     def some_method(cls, x: str) -> str:
#         return x

# class MyClassWithSomeMethod(metaclass=MyMeta):
#     pass

# mc = MyClassWithSomeMethod()
# print(mc.some_method("asd"))

# exit()

# -----------------------

# from dataclasses import dataclass

# @dataclass
# class InventoryItem:
#     """Class for keeping track of an item in inventory."""
#     name: str
#     unit_price: float
#     quantity_on_hand: int = 0

#     def total_cost(self) -> float:
#         return self.unit_price * self.quantity_on_hand

# d = InventoryItem("asd", 10, 50)


# from collections import namedtuple
# Student = namedtuple('Student', ['name', 'age', 'DOB'])
# # Adding values
# S = Student('Nandini', '19', '2541997')
# # Access using index
# print("The Student age using index is : ", end="")
# print(S[1])
# # Access using name
# print("The Student name using keyname is : ", end="")
# print(S.name)


# class Base:
#     a: int = 3
#     b: str = 'abc'

# class Derived(Base):
#     pass

# print(Derived.__annotations__)


# # constructor
# def constructor(self, arg):
#     self.constructor_arg = arg
  
# # method
# def displayMethod(self, arg):
#     print(arg)
  
# # class method
# @classmethod
# def classMethod(cls, arg):
#     print(arg)

# cls_annotations = {'string_attribute': 'str'} ## Added
  
# # creating class dynamically
# Geeks = type("Geeks", (object, ), {
#     # annotations
#     "__annotations__": cls_annotations,
#     # constructor
#     "__init__": constructor,
#     # data members
#     "string_attribute": "Geeks 4 geeks !",
#     "int_attribute": 1706256,
#     # member functions
#     "func_arg": displayMethod,
#     "class_func": classMethod
# })

# # creating objects
# obj = Geeks("constructor argument")
# print(obj.constructor_arg)
# print(obj.string_attribute)
# print(obj.int_attribute)
# obj.func_arg("Geeks for Geeks")
# print(obj.__annotations__)
# Geeks.class_func("Class Dynamically Created !")

# exit()

#----------------

# class TestMeta(type):
#     def __new__(cls, name, bases, cls_dict):
#         class_ = super().__new__(cls, name, bases, cls_dict)
#         print("Meta")
#         return class_

# def test_decorator(func):
#     print("Decorator")
#     return func

# class Test(Generic[StatesEnum, OutputsEnum], metaclass=TestMeta):
#     @overload
#     def __init__(self, states: Type[StatesEnum]) -> None: ...
#     @overload
#     def __init__(self, states: Type[StatesEnum], outputs: Type[OutputsEnum]) -> None: ...
#     def __init__(self, states: Type[StatesEnum], outputs: Optional[Type[OutputsEnum]] = None) -> None:
#         print("Constructor")
#         self.states: Type[StatesEnum] = states
#         self.outputs: Optional[Type[OutputsEnum]] = outputs  
#         # if self.outputs is None:
#         #     self.print = self._not_print

#     def _not_print(self, val: StatesEnum):
#         self.print(val, None)

#     @test_decorator
#     def print(self, val: StatesEnum, val2: OutputsEnum):
#         print(f"{self.states}: {val}")
#         if self.outputs is not None:
#             print(f"{self.outputs}")
#         else:
#             print(f"None")

# class Color(Enum):
#     RED = 1
#     GREEN = 2
#     BLUE = 3

# class ColorNot(Enum):
#     REDNOT = 1
#     GREENNOT = 2
#     BLUENOT = 3

# a = Test(Color)
# a.print(Color.RED, None)

# b = Test(Color, ColorNot)
# b.print(Color.GREEN, ColorNot.BLUENOT)
# b.print(Color.GREEN, ColorNot.REDNOT)

# exit()

# --------------------------


class FsmDefinition(Generic[StatesEnum, InputsEnum, OutputsEnum]):
    def __init__(self, states: Type[StatesEnum], inputs: Type[InputsEnum], \
            outputs: Optional[Type[OutputsEnum]] = None, initial_state: Optional[StatesEnum] = None, name: Optional[str] = None, \
            auto_rebuild: bool = True, raise_on_invalid_input: bool = True) -> None:
        self._states_type: Type[StatesEnum] = states
        self._inputs_type: Type[InputsEnum] = inputs
        self._outputs_type: Optional[Type[OutputsEnum]] = outputs

        self._initial_state: Optional[StatesEnum] = initial_state
        
        # TODO: Hide definitions from public - make separate configurator objects and additional views for internal use
        self._states_definition: Dict[StatesEnum, StateDefinition[InputsEnum]] = {state : StateDefinition[InputsEnum]() for state in states}
        self._inputs_definition: Dict[InputsEnum, InputDefinition] = {input : InputDefinition() for input in inputs}
        if outputs is None:
            self._outputs_definition: Optional[Dict[OutputsEnum, OutputDefinition]] = None
        else:
            self._outputs_definition = {output : OutputDefinition() for output in outputs}

        self._ready: bool = False
        self._auto_rebuild: bool = auto_rebuild
        self._raise_on_invalid_input: bool = raise_on_invalid_input
        
        self._name: Optional[str] = name

    @property
    def name(self):
        return self._name

    def configure_state(self, state: StatesEnum) -> StateDefinition[InputsEnum]:
        try:
            return self._states_definition[state]
        except KeyError:
            raise KeyError()

    def configure_input(self, input: InputsEnum) -> InputDefinition:
        try:
            return self._inputs_definition[input]
        except KeyError:
            raise KeyError()

    def configure_output(self, output: OutputsEnum) -> OutputDefinition:
        if self._outputs_definition is None:
            raise KeyError()
        
        try:
            return self._outputs_definition[output]
        except KeyError:
            raise KeyError()

    def set_initial_state(self, state: StatesEnum):
        self._initial_state = state

    def build(self):
        self._ready = False

        named_expressions: Dict[object, Expression]  = {input: input_definition.expression for input, input_definition in self._inputs_definition.items()}  # type: ignore
        builder = ExpressionBuilder(named_expressions)

        try:
            builder.build()
        except ExpressionBuildError as err:
            raise RuntimeError(err)
        
        named_built_expressions = builder.get_built_expressions()
        names = set()
        for name, built_expression in named_built_expressions:
            if name in names:
                raise RuntimeError(f"Duplicate name {name} encountered after building expresions")
            else:
                names.add(name)
            
            try:
                input_definition: InputDefinition = self._inputs_definition[name]  # type:ignore  # TODO: Any other way to do this?
            except KeyError:
                raise RuntimeError(f"Unable to match built expression name {name} to input")
            input_definition.built_expression = built_expression
        
        if names ^ self._inputs_definition.keys():
            for input_definition in self._inputs_definition.values():
                input_definition.built_expression = None

            raise RuntimeError(f"Built names {names} do not coninside with inputs {self._inputs_definition.keys()}")

        self._ready = True

    def trigger(self, fsm_implementation: FsmImplementation, input: InputsEnum):
        if self._ready is False:
            if self._auto_rebuild is True:
                self.build()
            else:
                raise RuntimeError()
        
        # TODO: Handle busy correctly, i.e. add queueable property or immediate or throw
        if fsm_implementation.busy is True:
            raise RuntimeError()
        fsm_implementation.busy = True

        state_definition = self._states_definition[fsm_implementation.state]
        if input in state_definition.inputs:
            input_definition = self._inputs_definition[input]
        else:
            if self._raise_on_invalid_input is True:
                raise KeyError()
            else:
                return False
        
        for on_input in input_definition.on_input:
            on_input(self, fsm_implementation, input)
        for on_input in fsm_implementation.configure_input(input).on_input:
            on_input(self, fsm_implementation, input)

        runnable_expression = input_definition.built_expression(self, fsm_implementation)  # type:ignore  # TODO: why?
        while runnable_expression is not None:
            runnable_expression = input_definition.built_expression(self, fsm_implementation)  # type:ignore
        
        fsm_implementation.busy = False

        return True

    def implement(self, initial_state: Optional[StatesEnum] = None):        
        if initial_state is not None:
            fsm_implementation = FsmImplementation(self._states_type, self._inputs_type, self._outputs_type, initial_state)
        elif self._initial_state is not None:
            fsm_implementation = FsmImplementation(self._states_type, self._inputs_type, self._outputs_type, self._initial_state)
        else:
            raise RuntimeError()

        return Fsm(self, fsm_implementation)


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


class ExpressionBuilder:
    def __init__(self, named_expressions: Dict[object, Expression]) -> None:
        self._named_expressions: Dict[object, Expression] = named_expressions
        self._named_built_expressions: Dict[object, CallableExpression] = {}

        self._labels: Dict[str, CallableExpression] = {}
        """Label name to built expression mapping. Collects all labels as they are met traversing raw expressions."""
        self._jumps: Dict[CallableExpressionPoint, str] = {}
        """Built expressions mapped to their next jump name that is not known yet. Will be connected in final pass."""
        self._shortcuts: Dict[object, str] = {}
        """Stores names of raw expressions that have a jump as their head. Will be connected in final pass."""

        self._current_name: object = None
        self._pending_label: Optional[str] = None

    def _connect_callable_expressions(self, previous_callable_expression_point: CallableExpressionPoint, current_callable_expression: CallableExpression):
        if previous_callable_expression_point.previous_callable_expression is not None:
            if isinstance(previous_callable_expression_point.previous_callable_expression, (CallableAction, CallableOutput, CallableState)):
                previous_callable_expression_point.previous_callable_expression.next = current_callable_expression
            elif isinstance(previous_callable_expression_point.previous_callable_expression, CallableIf):
                if previous_callable_expression_point.next_id == 0:
                    previous_callable_expression_point.previous_callable_expression.next_true = current_callable_expression
                else:
                    previous_callable_expression_point.previous_callable_expression.next_false = current_callable_expression
            else:
                raise ExpressionError(f"Cannot assign next expression to callable expression {previous_callable_expression_point.previous_callable_expression}.")         
        else:
            self._named_built_expressions[self._current_name] = current_callable_expression

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
        elif isinstance(expression, If):
            callable_expression = CallableIf(expression.condition)
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

                visited_expressions: Set[Expression] = set((head_expression, ))
                def check_visited(expression: Expression):
                    if expression in visited_expressions:
                        raise ExpressionError(f"Expression loop detected at {expression}")
                    else:
                        visited_expressions.add(expression)
                to_visit_stack: List[ExpressionPoint] = [ExpressionPoint(CallableExpressionPoint(None, 0), head_expression)]

                while to_visit_stack:
                    expression_point = to_visit_stack.pop()
                    previous_callable_expression_point, current_expression = expression_point
                    
                    current_callable_expression = self._build_expression(previous_callable_expression_point, current_expression)
                    if current_callable_expression is not None:
                        self._connect_callable_expressions(previous_callable_expression_point, current_callable_expression)
                    
                    if isinstance(current_expression, (Action, Output)):
                        next_expression = current_expression.next
                        check_visited(next_expression)
                        to_visit_stack.append(ExpressionPoint(CallableExpressionPoint(current_callable_expression, 0), next_expression))
                    elif isinstance(current_expression, Label):
                        next_expression = current_expression.next
                        check_visited(next_expression)
                        to_visit_stack.append(ExpressionPoint(previous_callable_expression_point, next_expression))
                    elif isinstance(current_expression, If):
                        next_expression = current_expression.next_true
                        check_visited(next_expression)
                        to_visit_stack.append(ExpressionPoint(CallableExpressionPoint(current_callable_expression, 0), next_expression))
                        next_expression = current_expression.next_false
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
                self._named_built_expressions[name] = resolved_target
            for callable_expression_point, target in self._jumps:
                try:
                    resolved_target = self._labels[target]
                except KeyError:
                    raise ExpressionError(f"Unable to resolve label {target}")
                self._connect_callable_expressions(callable_expression_point, resolved_target)
        except ExpressionError as error:
            raise ExpressionBuildError(f"Error while building expression {name}: {error}")

    def get_built_expressions(self) -> ItemsView[object, CallableExpression]:
        return self._named_built_expressions.items()


class StateDefinition(Generic[InputsEnum]):
    def __init__(self) -> None:
        # TODO: How to add typing to this class? Based on FsmDefinition
        self.inputs: Set[InputsEnum] = set()
        self.on_enter: Set[Callable] = set()  # TODO: Specify Callable
        self.on_exit: Set[Callable] = set()

    def add_input(self, input: InputsEnum):
        if isinstance(input, Iterable):
            self.inputs.update(input)
        else:
            self.inputs.add(input)
        return self

    def remove_input(self, input: InputsEnum):
        if isinstance(input, Iterable):
            self.inputs.difference_update(input)
        else:
            self.inputs.discard(input)
        return self

    def add_on_enter(self, func: Callable):
        self.on_enter.add(func)
        return self

    def remove_on_enter(self, func: Callable):
        self.on_enter.discard(func)
        return self

    def add_on_exit(self, func: Callable):
        self.on_exit.add(func)
        return self

    def remove_on_exit(self, func: Callable):
        self.on_exit.discard(func)
        return self


class InputDefinition:
    def __init__(self) -> None:
        self.expression: Optional[Expression] = None
        self.built_expression: Optional[CallableExpression] = None
        self.on_input: Set[Callable] = set()

    def set_expression(self, expr: Expression):
        self.expression = expr
        return self

    def add_on_input(self, func: Callable):
        self.on_input.add(func)
        return self

    def remove_on_input(self, func: Callable):
        self.on_input.discard(func)
        return self

class OutputDefinition:
    def __init__(self) -> None:
        self.on_output: Set[Callable] = set()

    def add_on_output(self, func: Callable):
        self.on_output.add(func)
        return self

    def remove_on_output(self, func: Callable):
        self.on_output.discard(func)
        return self


class FsmImplementation(Generic[StatesEnum, InputsEnum, OutputsEnum]):
    def __init__(self, states: Type[StatesEnum], inputs: Type[InputsEnum], outputs: Optional[Type[OutputsEnum]], initial_state: StatesEnum) -> None:
        self.state: StatesEnum = initial_state

        self.busy: bool = False

        self._states_implementaion: Dict[StatesEnum, StateImplementation] = {state : StateImplementation() for state in states}
        self._inputs_implementaion: Dict[InputsEnum, InputImplementation] = {input : InputImplementation() for input in inputs}
        if outputs is None:
            self._outputs_implementaion: Optional[Dict[OutputsEnum, OutputImplementation]] = None
        else:
            self._outputs_implementaion = {output : OutputImplementation() for output in outputs}

    def configure_state(self, state: StatesEnum):
        try:
            return self._states_implementaion[state]
        except KeyError:
            raise KeyError()

    def configure_input(self, input: InputsEnum):
        try:
            return self._inputs_implementaion[input]
        except KeyError:
            raise KeyError()

    def configure_output(self, output: OutputsEnum):
        if self._outputs_implementaion is None:
            raise KeyError()
        try:
            return self._outputs_implementaion[output]
        except KeyError:
            raise KeyError()


class StateImplementation:
    def __init__(self) -> None:
        self.on_enter = set()
        self.on_exit = set()

    def add_on_enter(self, func: Callable):
        self.on_enter.add(func)
        return self

    def remove_on_enter(self, func: Callable):
        self.on_enter.discard(func)
        return self

    def add_on_exit(self, func: Callable):
        self.on_exit.add(func)
        return self

    def remove_on_exit(self, func: Callable):
        self.on_exit.discard(func)
        return self


class InputImplementation:
    def __init__(self) -> None:
        self.on_input = set()

    def add_on_input(self, func: Callable):
        self.on_input.add(func)
        return self

    def remove_on_input(self, func: Callable):
        self.on_input.discard(func)
        return self


class OutputImplementation:
    def __init__(self) -> None:
        self.on_output = set()

    def add_on_output(self, func: Callable):
        self.on_output.add(func)
        return self

    def remove_on_output(self, func: Callable):
        self.on_output.discard(func)
        return self


class Fsm(Generic[StatesEnum, InputsEnum, OutputsEnum]):
    def __init__(self, fsm_definition: FsmDefinition[StatesEnum, InputsEnum, OutputsEnum], \
            fsm_implementation: FsmImplementation[StatesEnum, InputsEnum, OutputsEnum]) -> None:
        self._fsm_definition: FsmDefinition[StatesEnum, InputsEnum, OutputsEnum] = fsm_definition
        self._fsm_implementation: FsmImplementation[StatesEnum, InputsEnum, OutputsEnum] = fsm_implementation

    def configure_state(self, state: StatesEnum) -> StateImplementation:
        return self._fsm_implementation.configure_state(state)

    def configure_input(self, input: InputsEnum) -> InputImplementation:
        return self._fsm_implementation.configure_input(input)

    def configure_output(self, output: OutputsEnum) -> OutputImplementation:
        return self._fsm_implementation.configure_output(output)

    @property
    def state(self):
        return self._fsm_implementation.state

    def trigger(self, input: InputsEnum):
        self._fsm_definition.trigger(self._fsm_implementation, input)


class Expression(ABC):
    def __init__(self) -> None:
        pass


class ChainMixin:
    def Action(self, callable: Callable) -> Action:
        self.next = Action(callable)
        return self.next

    def State(self, value: object) -> State:
        self.next = State(value)
        return self.next

    def If(self, condition: Callable, next_true: Expression, next_false: Expression) -> If:
        self.next = If(condition, next_true, next_false)
        return self.next

    def Label(self, label: str) -> Label:
        self.next = Label(label)
        return self.next

    def Jump(self, target: str) -> Jump:
        self.next = Jump(target)
        return self.next

    def Output(self, output: object) -> Output:
        self.next = Output(output)
        return self.next


class Action(Expression, ChainMixin):
    # TODO: Add parameters to this (and to Fsm trigger)
    def __init__(self, callable: Callable) -> None:
        super().__init__()

        self.callable: Callable = callable
        self.next: Optional[Expression] = None


class Output(Expression, ChainMixin):
    def __init__(self, output: object) -> None:
        super().__init__()

        self.output: object = output
        self.next: Optional[Expression] = None


class State(Expression):   # TODO: Add typing to _state_
    def __init__(self, state: object) -> None:
        super().__init__()

        self.state: object = state


class If(Expression):
    def __init__(self, condition: Callable, next_true: Expression, next_false: Expression) -> None:
        super().__init__()

        self.condition: Callable = condition
        self.next_true: Expression = next_true
        self.next_false: Expression = next_false


class Label(Expression, ChainMixin):
    def __init__(self, label: str) -> None:
        super().__init__()

        self.label: str = label
        self.next: Optional[Expression] = None


class Jump(Expression):
    def __init__(self, target: str) -> None:
        super().__init__()

        self.target: str = target


class CallableExpression(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def __call__(self, fsm_definition: FsmDefinition, fsm_implementation: FsmImplementation) -> Optional[CallableExpression]:
        pass


class CallableAction(CallableExpression):
    def __init__(self, callable: Callable) -> None:
        super().__init__()

        self.callable: Callable = callable
        self.next: Optional[CallableExpression] = None

    def __call__(self, fsm_definition: FsmDefinition, fsm_implementation: FsmImplementation) -> Optional[CallableExpression]:
        self.callable()

        return self.next

    
class CallableOutput(CallableExpression):
    def __init__(self, output: object) -> None:
        super().__init__()

        self.output: object = output  # TODO: Typing here and in __call__
        self.next: Optional[CallableExpression] = None

    def __call__(self, fsm_definition: FsmDefinition, fsm_implementation: FsmImplementation) -> Optional[CallableExpression]:
        try:
            output_definition = fsm_definition.configure_output(self.output)
        except KeyError:
            pass
        else:
            for on_output in output_definition.on_output:
                on_output(fsm_definition, fsm_implementation, self.output)
        try:
            output_implementation = fsm_implementation.configure_output(self.output)  # type:ignore
        except KeyError:
            pass
        else:
            for on_output in output_implementation.on_output:
                on_output(fsm_definition, fsm_implementation, self.output)
        
        return self.next


class CallableState(CallableExpression):
    def __init__(self, state: StatesEnum) -> None:
        super().__init__()

        self.state: StatesEnum = state
 
    def __call__(self, fsm_definition: FsmDefinition, fsm_implementation: FsmImplementation) -> Optional[CallableExpression]:
        if self.state is None:
            return None

        source_state_definition = fsm_definition.configure_state(fsm_implementation.state)
        source_state_implementation = fsm_implementation.configure_state(fsm_implementation.state)
        destination_state_definition = fsm_definition.configure_state(fsm_implementation.state)
        destination_state_implementation = fsm_implementation.configure_state(self.state)

        for on_exit in source_state_definition.on_exit:
            on_exit(fsm_definition, fsm_implementation, fsm_implementation.state, self.state)
        for on_exit in source_state_implementation.on_exit:
            on_exit(fsm_definition, fsm_implementation, fsm_implementation.state, self.state)
        for on_enter in destination_state_definition.on_enter:
            on_enter(fsm_definition, fsm_implementation, self.state, fsm_implementation.state)
        for on_enter in destination_state_implementation.on_enter:
            on_enter(fsm_definition, fsm_implementation, self.state, fsm_implementation.state)

        fsm_implementation.state = self.state

        return None


class CallableIf(CallableExpression):
    def __init__(self, condition: Callable) -> None:
        super().__init__()

        self.condition: Callable = condition
        self.next_true: Optional[CallableExpression] = None
        self.next_false: Optional[CallableExpression] = None
    
    def __call__(self, fsm_definition: FsmDefinition, fsm_implementation: FsmImplementation) -> Optional[CallableExpression]:
        condition = self.condition()

        if condition is True:
            return self.next_true
        else:
            return self.next_false


# ------------------------------------------
# ------------------------------------------
# ------------------------------------------

class PandaState(Enum):
    Sleep = 1
    Awake = 2
    Happy = 3

class PandaInput(Enum):
    WakeUp = 1
    Feed = 2
    Rest = 3

class PandaOutput(Enum):
    Poop = 1

def on_enter_print(panda_def, panda_impl, dst, src):
    print(f"On enter from {src} to {dst}")

def on_exit_print(panda_def, panda_impl, src, dst):
    print(f"On exit from {src} to {dst}")

def on_output_print(panda_def, panda_impl, output):
    print(f"On output {output}")

def on_input_print(panda_def, panda_impl, input):
    print(f"On input {input}")

def action_print(str):
    print(f"Action print: {str}")

def random_condition():
    if random() > 0.5:
        return True
    else:
        return False


a = Action(partial(action_print, "WakeUp"))\
        .State(PandaState.Awake)
print(a)


panda_definition = FsmDefinition(PandaState, PandaInput, PandaOutput, PandaState.Awake, "Panda")
panda_definition.configure_state(PandaState.Sleep)\
    .add_input(PandaInput.WakeUp)\
    .add_on_enter(on_enter_print)\
    .add_on_exit(on_exit_print)

panda_definition.configure_state(PandaState.Awake)\
    .add_input(PandaInput.Feed)\
    .add_on_enter(on_enter_print)\
    .add_on_exit(on_exit_print)
panda_definition.configure_state(PandaState.Happy)\
    .add_input(PandaInput.Rest)\
    .add_on_enter(on_enter_print)\
    .add_on_exit(on_exit_print)

panda_definition.configure_input(PandaInput.WakeUp)\
    .set_expression(
        Action(partial(action_print, "WakeUp"))\
        .State(PandaState.Awake)
    )\
    .add_on_input(on_input_print)
panda_definition.configure_input(PandaInput.Feed)\
    .set_expression(
        Action(partial(action_print, "Feed"))
        .Action(partial(action_print, "Feed2"))
        .If(
            random_condition,
            Action(partial(action_print, "Random True"))\
            .State(None)
            ,
            Action(partial(action_print, "Random False"))\
            .State(PandaState.Happy)
        )
    )\
    .add_on_input(on_input_print)

#Effect1() >> Effect2() >> Effect3()

# .Decide(some_callable, 
# {
#     1: bla,
#     2: foo,
#     3: boo,
# }
# )

panda_definition.configure_input(PandaInput.Rest)\
    .set_expression(
        Action(partial(action_print, "Rest"))\
        .If(
            random_condition,
            Action(partial(action_print, "Random2 True"))\
            .Output(PandaOutput.Poop)\
            .Jump("something")
            ,
            Action(partial(action_print, "Random2 False"))\
            .Label("something")\
            .State(PandaState.Sleep)
        )
    )\
    .add_on_input(on_input_print)
panda_definition.configure_output(PandaOutput.Poop)\
    .add_on_output(on_output_print)\
    .remove_on_output(on_output_print)\
    .add_on_output(on_output_print)

panda_definition.build()

panda_fsm = panda_definition.implement()

print(f"Panda state is {panda_fsm.state}")
panda_fsm.trigger(PandaInput.Feed)
print(f"Panda state is {panda_fsm.state}")
