from enum import IntEnum


class Action(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF_POT = 2
    RAISE_POT = 3
    ALL_IN = 4


ACTION_NAMES = {
    Action.FOLD: "fold",
    Action.CHECK_CALL: "check_call",
    Action.RAISE_HALF_POT: "raise_half_pot",
    Action.RAISE_POT: "raise_pot",
    Action.ALL_IN: "all_in",
}


def action_name(action):
    return ACTION_NAMES[Action(action)]
