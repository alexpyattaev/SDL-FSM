from SDL_FSM import FSM_base, FSM_STATES, Transition, event, message, Message, Event_Handler
import asyncio


class Timer(FSM_base):
    """A very simple timer FSM"""
    class states(FSM_STATES):
        ARMED = "Wait for DNS query"
        IDLE = "Wait for reply from remote"

    timeout: int = 0
    task: asyncio.Task

    def __init__(self, timeout):
        self.state = self.states.IDLE
        self.timeout = timeout

        FSM_base.__init__(self)

    def hookup(self, arm_message: Message = None, fire_event: Event_Handler = None):
        """hook up the timer to controlling FSM
        :param arm_message: a message that will be sent to Timer to arm it
        :param fire_event: an event the timer should trigger when expiring
        :returns pointer to self (for builder pattern)
        """
        if fire_event is not None:
            self.fire.subscribe(fire_event)
        if arm_message is not None:
            arm_message.subscribe(self.arm)
        return self

    @event(state=states.IDLE)
    def arm(self, timeout=None) -> Event_Handler.ReturnType:
        if timeout is not None:
            self.timeout = timeout
        print("Armed timer")
        return self._do_arm

    fire = Message("timer fired")

    def impl_timer_run(self):
        async def sleeper():
            await asyncio.sleep(self.timeout)
            self._do_fire()

        self.task = asyncio.create_task(sleeper())

    _do_arm = Transition(src=states.IDLE, dst=states.ARMED).exec(impl_timer_run)
    _do_fire = Transition(src=states.ARMED, dst=states.IDLE).send_later(fire)


class PingSender(FSM_base):
    class states(FSM_STATES):

        DNS_LOOKUP = "Wait for DNS query"
        WAIT_REPLY = "Wait for reply from remote"
        DONE = "Test complete, terminal state"

    sent: int = 0
    acks: int = 0

    def __init__(self, target, interval=1.0):
        self.target = target
        self.interval = interval
        self.state = self.states.DNS_LOOKUP

        # create an embedded FSM (a timer) and hook it up
        self.timeout_timer = Timer(interval).hookup(arm_message=self.arm_timer, fire_event=self.timer_fired)

        FSM_base.__init__(self)

    dns_query = Message(payload="placeholder")

    @message
    def ICMP_PING_RQ(self) -> object:
        self.sent += 1
        return {"type": "ping", "seq": self.sent}

    arm_timer = Message()

    perform_dns = Transition(src=states.DNS_LOOKUP, dst=states.WAIT_REPLY).send(dns_query).send(arm_timer)
    send_ping = Transition(src=states.WAIT_REPLY, dst=states.WAIT_REPLY).send(ICMP_PING_RQ).send(arm_timer)
    finalize = Transition(src=states.WAIT_REPLY, dst=states.DONE)

    @event(states=[states.DNS_LOOKUP, states.WAIT_REPLY])
    def timer_fired(self) -> Event_Handler.ReturnType:
        if self.state == self.states.DNS_LOOKUP:
            self.invalidate("Timeout on DNS")
            return None
        else:
            return self.send_ping

    @event(state=states.DNS_LOOKUP)
    def DNS_resolution(self, data) -> Event_Handler.ReturnType:
        print(data)
        if data is not None:
            return self.send_ping
        else:
            self.invalidate("invalid DNS reply!")

    @event(state=states.WAIT_REPLY)
    def ICMP_PING_REPLY(self, data) -> Event_Handler.ReturnType:
        print(data)
        if data is not None:
            self.acks += 1
        else:
            self.invalidate("Timeout!")
        return None

    def __repr__(self):
        return f"Sender to {self.target}"


ps = PingSender(target=1)
print(ps.ICMP_PING_RQ())
