# noinspection PyUnresolvedReferences
from fixtures import panda_fsm as Panda


def test_creation(Panda):
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
