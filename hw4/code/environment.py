import random


def init_state():
    """
    initialize state
    """
    dealer = random.randint(1, 10)
    player = random.randint(1, 10)
    return dealer * 100 + player


def drawcard(x):
    """
    draw a card randomly
    """
    if random.randint(1, 3) < 3:
        return x + random.randint(1, 10)
    else:
        return x - random.randint(1, 10)


def step(state, action):
    """
    get next state and reward.
    state = dealer * 100 + player
    action = {0, 1}, 0 for stick and 1 for hit
    """
    dealer = state // 100
    player = state % 100
    if action: # hit
        player = drawcard(player)
        if player < 1 or player > 21:
            return "terminal", -1.0
        else:
            return dealer * 100 + player, 0.0
    elif not action: # stick
        while dealer < 17:
            dealer = drawcard(dealer)
            if dealer < 1 or dealer > 21:
                return "terminal", 1.0
        if dealer > player:
            return "terminal", -1.0
        elif dealer < player:
            return "terminal", 1.0
        else:
            return "terminal", 0.0
