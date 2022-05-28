# SDL-FSM
Finite state machine logic roughly as envisioned in SDL

# Basic principles

Each FSM is defined as an instance of a class derived from `FSM_base`. This is quite natural as classes 
were originally designed for this purpose.
The FSM logic is designed to be largely immutable (to prevent accidental breakage), though mutable variants can be 
added to override the defaults provided.

## States
Set of FSM states is given by a string enum inherited from FSM_STATE base class:
```python
class states(FSM_STATE):
    A = "description for A"
    B = "description for B"
```

## Internal variables
Any internal variable relevant to FSM functionality (e.g. counters) can be held in class/instance variables.

## Initialization
`__init__` method is used to initialize the FSM, as one would expect:
```python
def __init__(self, name: str): # name is a custom parameter
    self.name = name
    self.state = self.states.A # must set the initial state
    FSM_base.__init__(self) # must call inherited constructor
```

## Transitions

Transitions are defined as `Transition` instances that bind the edge in FSM graph and relevant side-effects. Transitions 
can be created in immutable form with frozen=True, that prevent their modification in runtime. 
```python
eat = Transition(src=states.HUNGRY, dst=states.ANGRY, frozen=True).side_effect(create_garbage)
```
Once a transition is started, it will fire all side-effects in order, and if no exceptions are thrown FSM will enter 
the new state. Unlike most other FSM frameworks, there is no on_enter_state callback (as it makes no sense in SDL).
If transition can not complete due to exceptions, the FSM will be invalidated (as there is no concrete state for it
to be in). Invalid FSMs will throw exceptions when interacted with.

## Event handling
Methods are essentially handlers for external events. Calling a method on an FSM instance causes FSM to react in some 
form.

To install an event handler, use the following pattern:
```python
@event(states= [ ... ] )
def some_handler(self, caller=None, **kwargs) -> Optional[Transition]:
    if self.counter > 5:
        return self.transition_a
    else:
        return None 
```
Here states indicates in which states can this event be received. If an event is posted in an inappropriate state, 
the caller will be notified about that via exception. 

Handler's return value is either a Transition object, or None if no action should be taken in response to Event. Thus,
if you want you can entirely override the logic of event handling by not providing states argument and implementing 
your own filtering inside the handler (including possible exception throwing to indicate inappropriate state).

# Linking the FSMs
To define complex logic one needs a clear mechanism to link FSMs. In SDL this is achieved by message passing. Thus,
FSMs are not supposed to be nested into each other like typical objects in an OOP program, but rather connected together. 

Linking FSMs is achieved via publish-subscribe mechanism. Any Transition may define a special `send` side-effect as follows:

```python

# define a static message that will be same for all FSM instances
poke = Message(payload={"type":"poke"})

# define a magic method that will form message content. 
# This method will be replaced with Message instance during __init__
@message 
def ping_packet(self, *args, **kwargs)->object:
    return {"type":"ping", "seq":self.counter}

# attach messages to Transitions as follows:
do_send_ping = Transition(src=states.WAIT, dst=states.WAIT, frozen=True).send(ping_packet).send(poke)
```

This ensures messages are sent, but they do not have any particular destination. To link FSMs, another FSM must be 
set up to receive messages. This is done either from outside of both FSMs as follows:

```python
Tx.ICMP_PING_RQ.subscribe(RX.ICMP_PING_RQ) 
```

or, alternatively, this can also be done in the constructor of the FSM

```python
def __init__(self, transmitter):  # transmitter is an instance of another FSM passed in
    self.tx = transmitter
    transmitter.ICMP_PING_RQ.subscribe(self.on_ping_request)  # subscribe to ICMP_PING_RQ message (handled by on_ping_request)
    self.ICMP_PING_REPLY.subscribe(transmitter.on_ping_reply)  # you can also subscribe someone else!
    FSM_base.__init__(self)  # must call inherited constructor
```
or in any of the methods:

```python
def unsubscribe_tx(self):
    self.tx.ICMP_PING_RQ.unsubscribe(self.on_ping_request)        
```

Thus, the subscribtions can be added and removed dynamically at runtime to reflect changes in your program.
When an FSM instance receives a message, a event_handler with matching name is called to process it.
Most importantly, names of messages and handlers in different FSMs do not have to be identical, allowing you to define
cleanly proxy FSMs.

## Terminating the FSM instances
The FSM instances keep references to all objects with which they have subscribe relationship in both directions.
If you want to destroy an FSM, those references must be cleared such that GC can get to the FSM entity without
delay (as well as to prevent zombie FSMs and associated bugs). Do this with .invalidate() method on FSM instance.
Invalidated FSMs are automatically unsubscribed from everywhere (i.e. stop receiving events) and can not run Transitions,
thus are unable to generate new messages.

