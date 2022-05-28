from typing import Optional

from SDL_FSM import FSM_base, FSM_STATES, Transition, event, message, Message


class Timer(FSM_base):
    class states(FSM_STATES):
        ARMED = "Wait for DNS query"
        IDLE = "Wait for reply from remote"

    timeout = 0

    def __init__(self, timeout):
        self.state = self.states.IDLE
        self.timeout = timeout
        FSM_base.__init__(self)

    @event
    def arm(self, timeout=None):
        if timeout is not None:
            self.timeout = timeout
        print("Armed timer")
        return self.do_arm

    fire = Message("timer fired")

    do_arm = Transition(src=states.IDLE, dst=states.ARMED)
    do_fire = Transition(src=states.ARMED, dst=states.IDLE).send(fire)


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

        # create and hook up the timer
        self.timeout_timer = Timer(interval)
        self.timeout_timer.fire.subscribe(self.timer_fired)
        self.arm_timer.subscribe(self.timeout_timer.arm)

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
    def timer_fired(self) -> Optional[Transition]:
        if self.state == self.states.DNS_LOOKUP:
            self.invalidate("Timeout on DNS")
            return None
        else:
            return self.send_ping

    @event(state=states.DNS_LOOKUP)
    def DNS_resolution(self, data) -> Optional[Transition]:
        print(data)
        if data is not None:
            return self.send_ping
        else:
            self.invalidate("invalid DNS reply!")

    @event(state=states.WAIT_REPLY)
    def ICMP_PING_REPLY(self, data) -> Optional[Transition]:
        print(data)
        if data is not None:
            self.acks += 1
        else:
            self.invalidate("Timeout!")
        return None

    def __repr__(self):
        return f"Sender to {self.target}"
