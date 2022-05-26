import dataclasses
from enum import auto

import pytest

from SDL_FSM import FSM_base, FSM_STATES, Transition, event


@pytest.fixture
def panda_fsm():
    """Construct a Panda FSM that eats shoots and leaves"""
    def shout_side_effect(FSM: FSM_base, extra_words: str = "", **_):
        print(f"{FSM} says WAAAGH {extra_words}")

    class Panda(FSM_base):

        class states(FSM_STATES):
            HUNGRY = "hungry, looking for food"
            ANGRY = "angry, looking for trouble"
            HAPPY = "happy, doing stuff"

        hunger_level: int = 0
        anger_level: int = 0

        def __init__(self, name: str):
            self.name = name
            self.state = self.states.HUNGRY

            FSM_base.__init__(self)

        def get_fed(self, **_):
            self.hunger_level -= 1

        def walk(self, **_):
            self.hunger_level += 1

        def get_tired(self, **_):
            self.anger_level += 1

        eat = Transition(src=states.HUNGRY, dst=states.ANGRY).self_effect(get_fed)
        shoot = Transition(src=states.ANGRY, dst=states.HAPPY).side_effect(shout_side_effect)
        leave = Transition(src=states.HAPPY, dst=states.HUNGRY).self_effect(walk).self_effect(get_tired)

        @event(state=states.HUNGRY)
        def food_sighted(self, **_):
            print(f"{self.name} food_sighted")
            if self.hunger_level > 0:
                return self.eat
            else:
                return None

        def __repr__(self):
            return self.name

        @event(state=states.ANGRY)
        def target_sighted(self, **_):
            print(f"{self.name} target_sighted")
            if self.anger_level > 0:
                return self.shoot
            else:
                return None

        @event(state=states.HAPPY)
        def all_done(self, **_):
            print(f"{self.name} all_done")
            return self.leave

    return Panda