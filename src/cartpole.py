"""cartpole.py - the CartPole simulator, shared by the CPU and GPU programs.

ONE source, TWO uses:
    sim = build_sim(lambda f: f)               -> plain Python (CPU baseline)
    sim = build_sim(cuda.jit(device=True))     -> compiled to GPU code (GPU kernels)
So all three approaches do exactly the same math; only the scheduling differs.

From the paper: CartPole, state = 4 floats (cart position, cart velocity, pole
angle, pole angular velocity), a step is simple row-wise math.
NOT in the paper (our choices, see README): the equations (we use gym
CartPole-v1, which the paper's code was ported from), where actions come from,
and how episodes reset.

Why every constant is wrapped in np.float32: the GPU should do cheap 32-bit math
(a gaming GPU is ~64x slower at 64-bit math). A plain Python number such as 9.8
is 64-bit and would silently turn the whole step into 64-bit math.
"""
import math
from types import SimpleNamespace

import numpy as np

f32 = np.float32

# ---- gym CartPole-v1 constants ----------------------------------------------
GRAVITY = f32(9.8)
MASS_CART = f32(1.0)
MASS_POLE = f32(0.1)
TOTAL_MASS = MASS_CART + MASS_POLE
POLE_HALF_LEN = f32(0.5)
POLEMASS_LEN = MASS_POLE * POLE_HALF_LEN
FORCE_MAG = f32(10.0)
TAU = f32(0.02)                    # seconds per step
X_LIMIT = f32(2.4)                 # episode ends if |x| > 2.4 ...
THETA_LIMIT = f32(0.20943951)      # ... or |theta| > 12 degrees
FOUR_THIRDS = f32(4.0) / f32(3.0)

# helpers for the random numbers
ONE_OVER_2_24 = f32(1.0 / 16777216.0)
HALF = f32(0.5)
TENTH = f32(0.1)
FIVE_HUNDREDTH = f32(0.05)

INIT_STEP = 0xFFFFFFFF             # "time" used to draw the very first state of each game


def build_sim(jit):
    """Define the simulator functions, passing each one through `jit`."""

    # ---- random numbers -----------------------------------------------------
    # A random number is a pure function of (seed, game, step, stream), so no
    # generator state has to be stored. All three approaches therefore see the
    # same "random" actions and resets.
    @jit
    def hash32(x):
        x ^= x >> 16
        x = (x * 0x7FEB352D) & 0xFFFFFFFF
        x ^= x >> 15
        x = (x * 0x846CA68B) & 0xFFFFFFFF
        x ^= x >> 16
        return x

    @jit
    def rand01(seed, game, t, stream):
        h = hash32(hash32(hash32(seed) ^ t) ^ (game * 8 + stream))
        return f32(h >> 8) * ONE_OVER_2_24          # uniform in [0, 1)

    # gym's reset: every state value uniform in [-0.05, 0.05].
    @jit
    def initial_state(seed, game, t):
        x = rand01(seed, game, t, 0) * TENTH - FIVE_HUNDREDTH
        v = rand01(seed, game, t, 1) * TENTH - FIVE_HUNDREDTH
        th = rand01(seed, game, t, 2) * TENTH - FIVE_HUNDREDTH
        w = rand01(seed, game, t, 3) * TENTH - FIVE_HUNDREDTH
        return x, v, th, w

    # Stand-in for the neural network the paper uses: push left (0) or right (1).
    @jit
    def choose_action(seed, game, t):
        if rand01(seed, game, t, 4) < HALF:
            return 1
        return 0

    # ---- one CartPole step (gym CartPole-v1, explicit Euler) ----------------
    @jit
    def physics(x, v, th, w, action):
        if action == 1:
            force = FORCE_MAG
        else:
            force = -FORCE_MAG
        c = f32(math.cos(th))
        s = f32(math.sin(th))
        temp = (force + POLEMASS_LEN * w * w * s) / TOTAL_MASS
        th_acc = (GRAVITY * s - c * temp) / (
            POLE_HALF_LEN * (FOUR_THIRDS - MASS_POLE * c * c / TOTAL_MASS))
        x_acc = temp - POLEMASS_LEN * th_acc * c / TOTAL_MASS
        x = x + TAU * v            # positions use the OLD velocities
        v = v + TAU * x_acc
        th = th + TAU * w
        w = w + TAU * th_acc
        return x, v, th, w

    # Advance one game by one step at time t. If the episode ended (cart or
    # pole out of range) the game restarts from a fresh initial state, so every
    # game does exactly K steps. Reward is +1 per step (as in gym), so it is
    # implied (N*K samples) and not stored. Last value: 1 if an episode ended.
    @jit
    def step_game(x, v, th, w, game, t, seed):
        x, v, th, w = physics(x, v, th, w, choose_action(seed, game, t))
        if abs(x) > X_LIMIT or abs(th) > THETA_LIMIT:
            x, v, th, w = initial_state(seed, game, t)
            return x, v, th, w, 1
        return x, v, th, w, 0

    return SimpleNamespace(step_game=step_game, initial_state=initial_state,
                           choose_action=choose_action)
