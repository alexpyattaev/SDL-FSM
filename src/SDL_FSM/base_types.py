import enum


class FSM_STATES(str, enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name