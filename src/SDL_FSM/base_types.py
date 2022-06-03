import enum


class FSM_STATES(str, enum.Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name
