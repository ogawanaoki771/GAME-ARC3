import turtle
import random
import math
import numpy as np
from collections import deque


# ============================================================
# TWO SPACES WORLD
# Predictive World Model + Dream Simulation
#
# Features
# ------------------------------------------------------------
# - Two large spaces
# - Thin tunnel
# - Gate
# - Gate opens by pushing
# - Reset blocks around walls
# - Reset changes POSITION ONLY
# - Goal
# - Circular AI monster
# - Monster predictive AI
# - Visual memory
# - Place memory
# - Transformation memory
# - Predictive World Model
# - Multi-step planning
# - Dream simulation
# - No reward
# ============================================================


# ============================================================
# SCREEN
# ============================================================

SCREEN_W = 1240
SCREEN_H = 820

WORLD_LEFT = -560
WORLD_RIGHT = 560
WORLD_BOTTOM = -300
WORLD_TOP = 290


# ============================================================
# SIMULATION
# ============================================================

NUM_AGENTS = 3

STEPS_PER_EPISODE = 700
MAX_EPISODES = 20


# ============================================================
# VISION
# ============================================================

OBS_W = 72
OBS_H = 44

CHANNELS = 4

PATCH = 6


# ============================================================
# LEARNING
# ============================================================

VISUAL_SIM_THRESHOLD = 0.74
PLACE_SIM_THRESHOLD = 0.80
TRANSFORM_SIM_THRESHOLD = 0.70

MEMORY_WINDOW = 140


# ============================================================
# PHYSICS
# ============================================================

GRAVITY = 0.2

GROUND_ACCEL = 0.85
AIR_ACCEL = 0.55

GROUND_FRICTION = 0.88
AIR_FRICTION = 0.985

JUMP_POWER = 11.0

MAX_SPEED = 10.0


# ============================================================
# ROOM GEOMETRY
# ============================================================

LEFT_ROOM_X1 = -535
LEFT_ROOM_X2 = -55

RIGHT_ROOM_X1 = 55
RIGHT_ROOM_X2 = 535


# ============================================================
# TUNNEL
# ============================================================

TUNNEL_X1 = -55
TUNNEL_X2 = 55

TUNNEL_Y1 = -80
TUNNEL_Y2 = 80


# ============================================================
# GATE
# ============================================================

GATE_X = 0
GATE_Y = 0

GATE_RADIUS = 24
GATE_PUSH_DISTANCE = 34


# ============================================================
# GOAL
# ============================================================

GOAL_X = 430
GOAL_Y = 220


# ============================================================
# RESET BLOCKS
# ============================================================

RESET_BLOCK_SIZE = 16

RESET_COOLDOWN = 25


# ============================================================
# MONSTER
# ============================================================

MONSTER_RADIUS = 24

MONSTER_SPEED = 4.2

MONSTER_PUSH = 1.75

MONSTER_BOUNCE = 1.25

MONSTER_MEMORY = 35

MONSTER_DETECTION_RANGE = 300


# ============================================================
# ACTIONS
# ============================================================

ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3
ACTION_BRAKE = 4
ACTION_WAIT = 5


ACTIONS = [
    ACTION_NONE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_JUMP,
    ACTION_BRAKE,
    ACTION_WAIT
]


ACTION_NAMES = {

    ACTION_NONE:
        "NONE",

    ACTION_LEFT:
        "LEFT",

    ACTION_RIGHT:
        "RIGHT",

    ACTION_JUMP:
        "JUMP",

    ACTION_BRAKE:
        "BRAKE",

    ACTION_WAIT:
        "WAIT"
}


ACTION_ACTIVITY_BIAS = {

    ACTION_NONE:
        -0.25,

    ACTION_LEFT:
        0.25,

    ACTION_RIGHT:
        0.25,

    ACTION_JUMP:
        0.30,

    ACTION_BRAKE:
        0.05,

    ACTION_WAIT:
        -0.10
}


# ============================================================
# UTILITY
# ============================================================

def clamp(
    x,
    lo,
    hi
):

    return max(
        lo,
        min(
            hi,
            x
        )
    )


def l1_distance(
    a,
    b
):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    return float(
        np.mean(
            np.abs(
                a - b
            )
        )
    )


def soft_distance(
    a,
    b,
    scale=4.0
):

    return math.exp(
        -scale
        *
        l1_distance(
            a,
            b
        )
    )


# ============================================================
# VISUAL CELL
# ============================================================

class VisualCell:

    def __init__(
        self,
        cell_id,
        feature,
        gx,
        gy
    ):

        self.id = cell_id

        self.feature = np.asarray(
            feature,
            dtype=np.float32
        )

        self.gx = gx
        self.gy = gy

        self.visits = 1

        self.activation = 1.0

        self.energy = 1.0


    def similarity(
        self,
        feature
    ):

        return soft_distance(
            self.feature,
            feature,
            4.3
        )


    def update(
        self,
        feature
    ):

        feature = np.asarray(
            feature,
            dtype=np.float32
        )

        self.feature = (

            0.91
            *
            self.feature

            +

            0.09
            *
            feature
        )

        self.visits += 1

        self.activation = min(
            2.0,
            self.activation + 0.04
        )

        self.energy = min(
            2.0,
            self.energy + 0.02
        )


    def decay(self):

        self.activation *= 0.994

        self.energy *= 0.999


# ============================================================
# SPATIAL REPRESENTATION
# ============================================================

class SpatialRepresentation:

    def __init__(self):

        self.cells = {}

        self.next_id = 0


    def patches(
        self,
        image
    ):

        image = np.asarray(
            image,
            dtype=np.float32
        ).reshape(
            OBS_H,
            OBS_W,
            CHANNELS
        )

        result = []

        for y in range(
            0,
            OBS_H,
            PATCH
        ):

            for x in range(
                0,
                OBS_W,
                PATCH
            ):

                patch = image[
                    y:y + PATCH,
                    x:x + PATCH
                ]

                feature = patch.mean(
                    axis=(0, 1)
                )

                result.append(
                    (
                        x // PATCH,
                        y // PATCH,
                        feature
                    )
                )

        return result


    def encode(
        self,
        image
    ):

        for (
            gx,
            gy,
            feature
        ) in self.patches(
            image
        ):

            key = (
                gx,
                gy
            )

            best = None

            best_score = 0.0

            for cell in self.cells.get(
                key,
                []
            ):

                score = cell.similarity(
                    feature
                )

                if score > best_score:

                    best_score = score

                    best = cell


            if (
                best is None
                or
                best_score
                < VISUAL_SIM_THRESHOLD
            ):

                cell = VisualCell(
                    self.next_id,
                    feature,
                    gx,
                    gy
                )

                self.next_id += 1

                self.cells.setdefault(
                    key,
                    []
                ).append(
                    cell
                )

            else:

                best.update(
                    feature
                )


    def feature_vector(
        self,
        image
    ):

        values = []

        for (
            _,
            _,
            feature
        ) in self.patches(
            image
        ):

            values.extend(
                feature.tolist()
            )

        return np.asarray(
            values,
            dtype=np.float32
        )


    def count(self):

        return sum(
            len(v)
            for v in self.cells.values()
        )


    def decay(self):

        for cells in self.cells.values():

            for cell in cells:

                cell.decay()


# ============================================================
# PLACE STATE
# ============================================================

class PlaceState:

    def __init__(
        self,
        place_id,
        feature,
        signature
    ):

        self.id = place_id

        self.center = np.asarray(
            feature,
            dtype=np.float32
        )

        self.signature = np.asarray(
            signature,
            dtype=np.float32
        )

        self.visits = 1

        self.energy = 1.0

        self.activation = 1.0

        self.transitions = {}

        self.state_history = deque(
            maxlen=MEMORY_WINDOW
        )


    def similarity(
        self,
        feature,
        signature
    ):

        a = soft_distance(
            self.center,
            feature,
            5.0
        )

        b = soft_distance(
            self.signature,
            signature,
            7.0
        )

        return (
            0.65 * a
            +
            0.35 * b
        )


    def absorb(
        self,
        feature,
        signature,
        state
    ):

        self.center = (

            0.92
            *
            self.center

            +

            0.08
            *
            feature
        )

        self.signature = (

            0.90
            *
            self.signature

            +

            0.10
            *
            signature
        )

        self.visits += 1

        self.energy = min(
            2.0,
            self.energy + 0.015
        )

        self.activation = min(
            2.0,
            self.activation + 0.05
        )

        self.state_history.append(
            np.asarray(
                state,
                dtype=np.float32
            )
        )


    def connect(
        self,
        target
    ):

        self.transitions[target] = (
            self.transitions.get(
                target,
                0
            )
            +
            1
        )


    def decay(self):

        self.activation *= 0.994

        self.energy *= 0.999


# ============================================================
# TRANSFORMATION CELL
# ============================================================

class TransformationCell:

    def __init__(
        self,
        transform_id,
        before,
        after,
        action,
        delta
    ):

        self.id = transform_id

        self.before_place = before

        self.after_place = after

        self.action_counts = {

            action:
                1
        }

        self.primary_action = action

        self.delta = np.asarray(
            delta,
            dtype=np.float32
        )

        self.visits = 1

        self.energy = 1.0

        self.stability = 0.15

        self.error = 1.0

        self.history = deque(
            maxlen=MEMORY_WINDOW
        )


    def similarity(
        self,
        delta,
        before
    ):

        if (
            self.before_place
            != before
        ):

            return 0.0

        return soft_distance(
            self.delta,
            delta,
            2.8
        )


    def update(
        self,
        delta,
        action
    ):

        error = l1_distance(
            self.delta,
            delta
        )

        self.delta = (

            0.88
            *
            self.delta

            +

            0.12
            *
            delta
        )

        self.error = (

            0.90
            *
            self.error

            +

            0.10
            *
            error
        )

        self.stability = clamp(

            0.97
            *
            self.stability

            +

            0.03
            *
            (
                1.0
                -
                error
            ),

            0.0,
            1.0
        )

        self.visits += 1

        self.energy = min(
            2.0,
            self.energy + 0.02
        )

        self.action_counts[action] = (

            self.action_counts.get(
                action,
                0
            )
            +
            1
        )

        self.primary_action = max(
            self.action_counts,
            key=self.action_counts.get
        )

        self.history.append(
            error
        )


    def curiosity(self):

        novelty = (

            1.0
            /
            math.sqrt(
                max(
                    1,
                    self.visits
                )
            )
        )

        isolation = (

            1.0

            if
            len(
                self.action_counts
            ) == 1

            else
            0.25
        )

        return (

            0.8
            *
            novelty

            +

            1.1
            *
            self.error

            +

            0.35
            *
            isolation

            +

            0.6
            *
            (
                1.0
                -
                self.stability
            )
        )


    def decay(self):

        self.energy *= 0.998


# ============================================================
# WORLD
# ============================================================

class World:

    def __init__(self):

        self.drawer = turtle.Turtle(
            visible=False
        )

        self.drawer.penup()

        self.drawer.speed(0)

        self.time = 0

        self.gate_open = False

        self.goal_reached = False

        self.reset_events = 0

        self.gate_pushes = 0

        self.monster = None

        self.reset_blocks = []

        self.create_reset_blocks()


    # ========================================================
    # RESET BLOCKS
    # ========================================================

    def create_reset_blocks(self):

        self.reset_blocks.clear()

        for y in range(
            WORLD_BOTTOM + 15,
            WORLD_TOP,
            RESET_BLOCK_SIZE
        ):

            self.reset_blocks.append(
                (
                    WORLD_LEFT + 5,
                    y
                )
            )

        for y in range(
            WORLD_BOTTOM + 15,
            WORLD_TOP,
            RESET_BLOCK_SIZE
        ):

            self.reset_blocks.append(
                (
                    WORLD_RIGHT - 5,
                    y
                )
            )

        for x in range(
            WORLD_LEFT + 10,
            WORLD_RIGHT,
            RESET_BLOCK_SIZE
        ):

            self.reset_blocks.append(
                (
                    x,
                    WORLD_TOP - 5
                )
            )

        for x in range(
            WORLD_LEFT + 10,
            WORLD_RIGHT,
            RESET_BLOCK_SIZE
        ):

            self.reset_blocks.append(
                (
                    x,
                    WORLD_BOTTOM + 5
                )
            )


    def reset(self):

        self.time = 0

        self.gate_open = False

        self.goal_reached = False

        self.reset_events = 0

        self.gate_pushes = 0

        self.create_reset_blocks()


    # ========================================================
    # GEOMETRY
    # ========================================================

    def inside_left_room(
        self,
        x,
        y
    ):

        return (

            LEFT_ROOM_X1 < x
            <
            LEFT_ROOM_X2

            and

            WORLD_BOTTOM < y
            <
            WORLD_TOP
        )


    def inside_right_room(
        self,
        x,
        y
    ):

        return (

            RIGHT_ROOM_X1 < x
            <
            RIGHT_ROOM_X2

            and

            WORLD_BOTTOM < y
            <
            WORLD_TOP
        )


    def inside_tunnel(
        self,
        x,
        y
    ):

        return (

            TUNNEL_X1 <= x
            <= TUNNEL_X2

            and

            TUNNEL_Y1 <= y
            <= TUNNEL_Y2
        )


    def valid_area(
        self,
        x,
        y
    ):

        return (

            self.inside_left_room(
                x,
                y
            )

            or

            self.inside_right_room(
                x,
                y
            )

            or

            self.inside_tunnel(
                x,
                y
            )
        )


    # ========================================================
    # RESET COLLISION
    # ========================================================

    def touches_reset_block(
        self,
        x,
        y
    ):

        margin = 13

        if (

            abs(
                x
                -
                WORLD_LEFT
            )
            <
            margin

            or

            abs(
                x
                -
                WORLD_RIGHT
            )
            <
            margin

            or

            abs(
                y
                -
                WORLD_BOTTOM
            )
            <
            margin

            or

            abs(
                y
                -
                WORLD_TOP
            )
            <
            margin

        ):

            return True

        return False


    # ========================================================
    # GATE
    # ========================================================

    def gate_distance(
        self,
        x,
        y
    ):

        return math.sqrt(

            (
                x
                -
                GATE_X
            ) ** 2

            +

            (
                y
                -
                GATE_Y
            ) ** 2
        )


    def push_gate(
        self,
        agent
    ):

        if self.gate_open:

            return

        distance = self.gate_distance(
            agent.x,
            agent.y
        )

        if (
            distance
            <
            GATE_PUSH_DISTANCE
        ):

            self.gate_open = True

            self.gate_pushes += 1


    # ========================================================
    # GOAL
    # ========================================================

    def check_goal(
        self,
        agent
    ):

        distance = math.sqrt(

            (
                agent.x
                -
                GOAL_X
            ) ** 2

            +

            (
                agent.y
                -
                GOAL_Y
            ) ** 2
        )

        if distance < 35:

            self.goal_reached = True

            agent.goal_reached = True


    # ========================================================
    # WORLD LIMIT
    # ========================================================

    def enforce_world(
        self,
        agent
    ):

        if (
            agent.x
            <
            WORLD_LEFT + 12
        ):

            agent.x = (
                WORLD_LEFT + 18
            )

            agent.vx = (
                abs(
                    agent.vx
                )
                *
                0.4
            )


        if (
            agent.x
            >
            WORLD_RIGHT - 12
        ):

            agent.x = (
                WORLD_RIGHT - 18
            )

            agent.vx = (
                -abs(
                    agent.vx
                )
                *
                0.4
            )


        if (
            agent.y
            <
            WORLD_BOTTOM + 12
        ):

            agent.y = (
                WORLD_BOTTOM + 18
            )

            agent.vy = (
                abs(
                    agent.vy
                )
                *
                0.4
            )


        if (
            agent.y
            >
            WORLD_TOP - 12
        ):

            agent.y = (
                WORLD_TOP - 18
            )

            agent.vy = (
                -abs(
                    agent.vy
                )
                *
                0.4
            )


    # ========================================================
    # RESET AGENT POSITION
    # ========================================================

    def reset_agent_position(
        self,
        agent
    ):

        agent.x = (

            LEFT_ROOM_X1
            +
            55
            +
            agent.id
            *
            30
        )

        agent.y = (
            WORLD_BOTTOM
            +
            70
        )

        agent.vx = 0.0

        agent.vy = 0.0

        agent.grounded = True

        agent.jumps = 0

        self.reset_events += 1


    # ========================================================
    # PHYSICS
    # ========================================================

    def step(
        self,
        agent,
        action
    ):

        self.time += 1

        # ----------------------------------------------------
        # ACTION
        # ----------------------------------------------------

        if action == ACTION_LEFT:

            agent.vx -= (

                GROUND_ACCEL

                if agent.grounded

                else AIR_ACCEL
            )

            agent.heading = -1


        elif action == ACTION_RIGHT:

            agent.vx += (

                GROUND_ACCEL

                if agent.grounded

                else AIR_ACCEL
            )

            agent.heading = 1


        elif action == ACTION_JUMP:

            if agent.grounded:

                agent.vy = JUMP_POWER

                agent.grounded = False

                agent.jumps = 1

            elif agent.jumps < 2:

                agent.vy = (
                    JUMP_POWER
                    *
                    0.82
                )

                agent.jumps += 1


        elif action == ACTION_BRAKE:

            agent.vx *= 0.20


        elif action == ACTION_WAIT:

            agent.vx *= 0.90


        # ----------------------------------------------------
        # HORIZONTAL
        # ----------------------------------------------------

        agent.vx = clamp(

            agent.vx,

            -MAX_SPEED,

            MAX_SPEED
        )

        agent.vx *= (

            GROUND_FRICTION

            if agent.grounded

            else AIR_FRICTION
        )

        agent.x += agent.vx


        # ----------------------------------------------------
        # VERTICAL
        # ----------------------------------------------------

        if not agent.grounded:

            agent.vy -= GRAVITY

            agent.y += agent.vy


        # ----------------------------------------------------
        # SIMPLE GROUND
        # ----------------------------------------------------

        if (
            agent.y
            <=
            WORLD_BOTTOM + 18
        ):

            agent.y = (
                WORLD_BOTTOM + 18
            )

            agent.vy = 0.0

            agent.grounded = True

            agent.jumps = 0


        # ----------------------------------------------------
        # GATE
        # ----------------------------------------------------

        self.push_gate(
            agent
        )


        # ----------------------------------------------------
        # CLOSED GATE
        # ----------------------------------------------------

        if not self.gate_open:

            if (

                abs(agent.x)
                < 18

                and

                abs(agent.y)
                < 95

            ):

                if agent.x < 0:

                    agent.x = -22

                else:

                    agent.x = 22

                agent.vx *= -0.25


        # ----------------------------------------------------
        # OPEN TUNNEL
        # ----------------------------------------------------

        else:

            if self.inside_tunnel(
                agent.x,
                agent.y
            ):

                if (
                    agent.y
                    <
                    TUNNEL_Y1 + 18
                ):

                    agent.y = (
                        TUNNEL_Y1 + 18
                    )

                    agent.vy = abs(
                        agent.vy
                    ) * 0.3

                    agent.grounded = True

                if (
                    agent.y
                    >
                    TUNNEL_Y2 - 18
                ):

                    agent.y = (
                        TUNNEL_Y2 - 18
                    )

                    agent.vy = -abs(
                        agent.vy
                    ) * 0.3


        # ----------------------------------------------------
        # WORLD
        # ----------------------------------------------------

        self.enforce_world(
            agent
        )


        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        if self.touches_reset_block(
            agent.x,
            agent.y
        ):

            if (

                self.time
                -
                agent.last_reset
                >
                RESET_COOLDOWN

            ):

                self.reset_agent_position(
                    agent
                )

                agent.last_reset = (
                    self.time
                )


        # ----------------------------------------------------
        # GOAL
        # ----------------------------------------------------

        self.check_goal(
            agent
        )


# ============================================================
# MONSTER AI
# ============================================================

class Monster:

    """
    Predictive circular enemy.

    - nearest target
    - velocity prediction
    - future position prediction
    - target memory
    - collision impulse
    """

    def __init__(
        self,
        world
    ):

        self.world = world

        self.x = 270

        self.y = 20

        self.vx = 0.0

        self.vy = 0.0

        self.target_id = None

        self.memory = deque(
            maxlen=MONSTER_MEMORY
        )

        self.attack_count = 0

        self.turtle = turtle.Turtle(
            shape="circle"
        )

        self.turtle.color(
            "#ff304f"
        )

        self.turtle.penup()

        self.turtle.speed(0)


    def reset(self):

        self.x = 270

        self.y = 20

        self.vx = 0.0

        self.vy = 0.0

        self.target_id = None

        self.memory.clear()

        self.attack_count = 0

        self.turtle.goto(
            self.x,
            self.y
        )


    # ========================================================
    # TARGET
    # ========================================================

    def choose_target(
        self,
        agents
    ):

        candidates = []

        for agent in agents:

            distance = math.sqrt(

                (
                    agent.x
                    -
                    self.x
                ) ** 2

                +

                (
                    agent.y
                    -
                    self.y
                ) ** 2
            )

            if (
                distance
                <
                MONSTER_DETECTION_RANGE
            ):

                candidates.append(
                    (
                        distance,
                        agent
                    )
                )


        if not candidates:

            self.target_id = None

            return None


        candidates.sort(
            key=lambda x: x[0]
        )

        target = candidates[0][1]

        self.target_id = target.id

        return target


    # ========================================================
    # PREDICT TARGET
    # ========================================================

    def predict_target(
        self,
        target
    ):

        distance = math.sqrt(

            (
                target.x
                -
                self.x
            ) ** 2

            +

            (
                target.y
                -
                self.y
            ) ** 2
        )

        prediction_time = clamp(

            distance / 25.0,

            2.0,

            12.0
        )

        predicted_x = (

            target.x

            +

            target.vx
            *
            prediction_time
        )

        predicted_y = (

            target.y

            +

            target.vy
            *
            prediction_time
        )

        return (
            predicted_x,
            predicted_y
        )


    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        agents
    ):

        target = self.choose_target(
            agents
        )

        if target is None:

            self.vx += (

                math.sin(
                    self.world.time
                    *
                    0.03
                )
                *
                0.15
            )

            self.vy += (

                math.cos(
                    self.world.time
                    *
                    0.025
                )
                *
                0.10
            )

        else:

            px, py = (
                self.predict_target(
                    target
                )
            )

            dx = px - self.x

            dy = py - self.y

            distance = math.sqrt(

                dx * dx
                +
                dy * dy
            )

            if distance > 1:

                self.vx += (

                    dx
                    /
                    distance
                ) * 0.42

                self.vy += (

                    dy
                    /
                    distance
                ) * 0.42


            self.memory.append(
                (
                    target.x,
                    target.y,
                    target.vx,
                    target.vy
                )
            )


        # ----------------------------------------------------
        # Gate
        # ----------------------------------------------------

        if not self.world.gate_open:

            if self.x < 70:

                self.vx += 0.35


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        speed = math.sqrt(

            self.vx ** 2
            +
            self.vy ** 2
        )

        if (
            speed
            >
            MONSTER_SPEED
        ):

            self.vx = (

                self.vx
                /
                speed
                *
                MONSTER_SPEED
            )

            self.vy = (

                self.vy
                /
                speed
                *
                MONSTER_SPEED
            )


        # ----------------------------------------------------
        # MOVE
        # ----------------------------------------------------

        self.x += self.vx

        self.y += self.vy

        self.vx *= 0.96

        self.vy *= 0.96


        # ----------------------------------------------------
        # WORLD
        # ----------------------------------------------------

        if (
            self.x
            <
            RIGHT_ROOM_X1 + 25
        ):

            self.x = (
                RIGHT_ROOM_X1 + 25
            )

            self.vx *= -0.7


        if (
            self.x
            >
            RIGHT_ROOM_X2 - 25
        ):

            self.x = (
                RIGHT_ROOM_X2 - 25
            )

            self.vx *= -0.7


        if (
            self.y
            <
            WORLD_BOTTOM + 25
        ):

            self.y = (
                WORLD_BOTTOM + 25
            )

            self.vy *= -0.7


        if (
            self.y
            >
            WORLD_TOP - 25
        ):

            self.y = (
                WORLD_TOP - 25
            )

            self.vy *= -0.7


        # ----------------------------------------------------
        # COLLISION
        # ----------------------------------------------------

        for agent in agents:

            self.collide(
                agent
            )


        self.turtle.goto(
            self.x,
            self.y
        )


    # ========================================================
    # COLLISION
    # ========================================================

    def collide(
        self,
        agent
    ):

        dx = (
            agent.x
            -
            self.x
        )

        dy = (
            agent.y
            -
            self.y
        )

        distance = math.sqrt(
            dx * dx
            +
            dy * dy
        )

        collision_distance = (

            MONSTER_RADIUS
            +
            13
        )


        if (

            distance
            <
            collision_distance

            and

            distance
            >
            0.01

        ):

            nx = dx / distance

            ny = dy / distance


            agent.vx += (
                nx
                *
                MONSTER_PUSH
            )

            agent.vy += (
                ny
                *
                MONSTER_PUSH
            )


            self.vx -= (
                nx
                *
                MONSTER_BOUNCE
            )

            self.vy -= (
                ny
                *
                MONSTER_BOUNCE
            )


            penetration = (

                collision_distance
                -
                distance
            )


            agent.x += (
                nx
                *
                penetration
            )

            agent.y += (
                ny
                *
                penetration
            )


            self.attack_count += 1


# ============================================================
# VISUAL FIELD
# ============================================================

class VisualField:

    def __init__(
        self,
        world,
        monster
    ):

        self.world = world

        self.monster = monster


    def to_grid(
        self,
        x,
        y
    ):

        gx = int(

            (
                x
                -
                WORLD_LEFT
            )

            /

            (
                WORLD_RIGHT
                -
                WORLD_LEFT
            )

            *

            (
                OBS_W - 1
            )
        )

        gy = int(

            (
                y
                -
                WORLD_BOTTOM
            )

            /

            (
                WORLD_TOP
                -
                WORLD_BOTTOM
            )

            *

            (
                OBS_H - 1
            )
        )

        return (

            clamp(
                gx,
                0,
                OBS_W - 1
            ),

            clamp(
                gy,
                0,
                OBS_H - 1
            )
        )


    def dot(
        self,
        channel,
        x,
        y,
        value
    ):

        gx, gy = self.to_grid(
            x,
            y
        )

        for dx in (
            -1,
            0,
            1
        ):

            for dy in (
                -1,
                0,
                1
            ):

                xx = gx + dx

                yy = gy + dy

                if (

                    0 <= xx < OBS_W

                    and

                    0 <= yy < OBS_H

                ):

                    channel[
                        yy,
                        xx
                    ] = max(

                        channel[
                            yy,
                            xx
                        ],

                        value
                    )


    def line(
        self,
        channel,
        x1,
        y1,
        x2,
        y2,
        value
    ):

        gx1, gy1 = self.to_grid(
            x1,
            y1
        )

        gx2, gy2 = self.to_grid(
            x2,
            y2
        )

        n = max(

            abs(
                gx2 - gx1
            ),

            abs(
                gy2 - gy1
            ),

            1
        )

        for i in range(
            n + 1
        ):

            t = i / n

            gx = int(

                gx1
                +
                (
                    gx2
                    -
                    gx1
                )
                *
                t
            )

            gy = int(

                gy1
                +
                (
                    gy2
                    -
                    gy1
                )
                *
                t
            )

            if (

                0 <= gx < OBS_W

                and

                0 <= gy < OBS_H

            ):

                channel[
                    gy,
                    gx
                ] = max(

                    channel[
                        gy,
                        gx
                    ],

                    value
                )


    def capture(
        self,
        agents,
        viewer_id
    ):

        img = np.zeros(

            (
                OBS_H,
                OBS_W,
                CHANNELS
            ),

            dtype=np.float32
        )


        # ----------------------------------------------------
        # WORLD WALLS
        # ----------------------------------------------------

        self.line(

            img[:, :, 0],

            WORLD_LEFT,
            WORLD_BOTTOM,

            WORLD_LEFT,
            WORLD_TOP,

            1.0
        )


        self.line(

            img[:, :, 0],

            WORLD_RIGHT,
            WORLD_BOTTOM,

            WORLD_RIGHT,
            WORLD_TOP,

            1.0
        )


        self.line(

            img[:, :, 0],

            WORLD_LEFT,
            WORLD_TOP,

            WORLD_RIGHT,
            WORLD_TOP,

            1.0
        )


        self.line(

            img[:, :, 0],

            WORLD_LEFT,
            WORLD_BOTTOM,

            WORLD_RIGHT,
            WORLD_BOTTOM,

            1.0
        )


        # ----------------------------------------------------
        # TUNNEL
        # ----------------------------------------------------

        self.line(

            img[:, :, 0],

            TUNNEL_X1,
            TUNNEL_Y1,

            TUNNEL_X2,
            TUNNEL_Y1,

            0.9
        )


        self.line(

            img[:, :, 0],

            TUNNEL_X1,
            TUNNEL_Y2,

            TUNNEL_X2,
            TUNNEL_Y2,

            0.9
        )


        # ----------------------------------------------------
        # GATE
        # ----------------------------------------------------

        if not self.world.gate_open:

            self.dot(

                img[:, :, 1],

                GATE_X,
                GATE_Y,

                1.0
            )


        # ----------------------------------------------------
        # GOAL
        # ----------------------------------------------------

        self.dot(

            img[:, :, 3],

            GOAL_X,
            GOAL_Y,

            1.0
        )


        # ----------------------------------------------------
        # MONSTER
        # ----------------------------------------------------

        self.dot(

            img[:, :, 1],

            self.monster.x,
            self.monster.y,

            1.0
        )


        # ----------------------------------------------------
        # AGENTS
        # ----------------------------------------------------

        for agent in agents:

            self.dot(

                img[:, :, 2],

                agent.x,
                agent.y,

                (
                    1.0
                    if
                    agent.id
                    ==
                    viewer_id

                    else
                    0.65
                )
            )


        return img.reshape(
            -1
        )


# ============================================================
# WORLD MODEL
# ============================================================

class WorldModel:

    """
    Predictive World Model.

    The model does NOT use reward.

    It learns:

        observation
             ↓
        place
             ↓
        transformation
             ↓
        prediction
             ↓
        danger
             ↓
        planning
             ↓
        action

    It also contains a lightweight
    internal Dream Simulator.
    """

    def __init__(self):

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        self.visual = SpatialRepresentation()

        self.places = []

        self.transforms = []

        self.traces = deque(
            maxlen=1200
        )

        self.prediction_memory = deque(
            maxlen=800
        )

        self.action_memory = {

            action:
                deque(
                    maxlen=250
                )

            for action in ACTIONS
        }

        self.error_history = deque(
            maxlen=800
        )

        # ----------------------------------------------------
        # IDS
        # ----------------------------------------------------

        self.next_place = 0

        self.next_transform = 0

        # ----------------------------------------------------
        # PLANNING
        # ----------------------------------------------------

        self.planning_depth = 4

        self.dream_count = 20

        self.exploration = 0.30

        self.prediction_weight = 1.25

        self.novelty_weight = 0.90

        self.danger_weight = 1.55

        self.stability_weight = 0.75

        # ----------------------------------------------------
        # INTERNAL STATE
        # ----------------------------------------------------

        self.model_step = 0


    # ========================================================
    # PLACE SIGNATURE
    # ========================================================

    def place_signature(
        self,
        image,
        body,
        context
    ):

        visual = (
            self.visual.feature_vector(
                image
            )
        )

        signature = np.concatenate(
            [

                visual[::4],

                body,

                context
            ]
        )

        return (
            visual,
            signature
        )


    # ========================================================
    # ENCODE PLACE
    # ========================================================

    def encode_place(
        self,
        image,
        body,
        context,
        state
    ):

        feature, signature = (
            self.place_signature(
                image,
                body,
                context
            )
        )

        best = None

        best_score = 0.0

        for place in self.places:

            score = place.similarity(

                feature,

                signature
            )

            if score > best_score:

                best_score = score

                best = place


        if (

            best is None

            or

            best_score
            <
            PLACE_SIM_THRESHOLD

        ):

            place = PlaceState(

                self.next_place,

                feature,

                signature
            )

            self.next_place += 1

            place.state_history.append(
                state
            )

            self.places.append(
                place
            )

            created = True

        else:

            best.absorb(

                feature,

                signature,

                state
            )

            place = best

            created = False


        return (

            place,

            feature,

            signature,

            created
        )


    # ========================================================
    # FIND TRANSFORMATION
    # ========================================================

    def find_transform(
        self,
        before,
        delta
    ):

        best = None

        best_score = 0.0

        for transform in self.transforms:

            score = transform.similarity(

                delta,

                before.id
            )

            if score > best_score:

                best_score = score

                best = transform


        return (
            best,
            best_score
        )


    # ========================================================
    # FORM TRANSFORMATION
    # ========================================================

    def form_transform(
        self,
        before,
        after,
        action,
        delta
    ):

        transform, score = (
            self.find_transform(

                before,

                delta
            )
        )


        if (

            transform is None

            or

            score
            <
            TRANSFORM_SIM_THRESHOLD

        ):

            transform = TransformationCell(

                self.next_transform,

                before.id,

                after.id,

                action,

                delta
            )

            self.next_transform += 1

            self.transforms.append(
                transform
            )

            error = 1.0

        else:

            transform.update(

                delta,

                action
            )

            error = transform.error

            transform.after_place = (
                after.id
            )


        before.connect(
            after.id
        )

        self.error_history.append(
            error
        )

        return (
            transform,
            error
        )


    # ========================================================
    # LEARN
    # ========================================================

    def learn(
        self,
        image_before,
        body_before,
        action,
        image_after,
        body_after,
        context_before,
        context_after
    ):

        self.model_step += 1

        # ----------------------------------------------------
        # Visual encoding
        # ----------------------------------------------------

        self.visual.encode(
            image_before
        )

        self.visual.encode(
            image_after
        )


        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        visual_before = (
            self.visual.feature_vector(
                image_before
            )
        )

        visual_after = (
            self.visual.feature_vector(
                image_after
            )
        )


        state_before = np.concatenate(

            [

                visual_before[::4],

                body_before,

                context_before

            ]
        )


        state_after = np.concatenate(

            [

                visual_after[::4],

                body_after,

                context_after

            ]
        )


        # ----------------------------------------------------
        # PLACE
        # ----------------------------------------------------

        before, _, _, created_before = (
            self.encode_place(

                image_before,

                body_before,

                context_before,

                state_before
            )
        )


        after, _, _, created_after = (
            self.encode_place(

                image_after,

                body_after,

                context_after,

                state_after
            )
        )


        # ----------------------------------------------------
        # DELTA
        # ----------------------------------------------------

        delta = (

            state_after

            -

            state_before
        )


        transform, error = (
            self.form_transform(

                before,

                after,

                action,

                delta
            )
        )


        # ----------------------------------------------------
        # ACTION MEMORY
        # ----------------------------------------------------

        self.action_memory[
            action
        ].append(

            {

                "before":
                    before.id,

                "after":
                    after.id,

                "delta":
                    delta.copy(),

                "error":
                    error
            }
        )


        # ----------------------------------------------------
        # PREDICTION MEMORY
        # ----------------------------------------------------

        self.prediction_memory.append(

            {

                "before":
                    before.id,

                "after":
                    after.id,

                "action":
                    action,

                "error":
                    error
            }
        )


        # ----------------------------------------------------
        # TRACE
        # ----------------------------------------------------

        self.traces.append(

            {

                "place_before":
                    before.id,

                "place_after":
                    after.id,

                "action":
                    action,

                "error":
                    error,

                "step":
                    self.model_step
            }
        )


        return {

            "before":
                before,

            "after":
                after,

            "transform":
                transform,

            "error":
                error,

            "created_before":
                created_before,

            "created_after":
                created_after
        }


    # ========================================================
    # CURRENT PLACE
    # ========================================================

    def current_place(
        self,
        image,
        body,
        context
    ):

        feature, signature = (
            self.place_signature(

                image,

                body,

                context
            )
        )

        best = None

        best_score = 0.0

        for place in self.places:

            score = place.similarity(

                feature,

                signature
            )

            if score > best_score:

                best_score = score

                best = place


        return (
            best,
            best_score
        )


    # ========================================================
    # PREDICT ACTION
    # ========================================================

    def predict_action(
        self,
        place,
        action
    ):

        if place is None:

            return None


        candidates = [

            t

            for t in self.transforms

            if (

                t.before_place
                ==
                place.id

                and

                action
                in
                t.action_counts
            )
        ]


        if not candidates:

            return None


        candidates.sort(

            key=lambda t:

                (

                    t.stability
                    *
                    0.7

                    +

                    math.log1p(
                        t.visits
                    )
                    *
                    0.1

                    -

                    t.error
                    *
                    0.5
                ),

            reverse=True
        )


        return candidates[0]


    # ========================================================
    # PREDICT PLACE
    # ========================================================

    def predict_place(
        self,
        place,
        action
    ):

        transform = (
            self.predict_action(

                place,

                action
            )
        )


        if transform is None:

            return None


        for candidate in self.places:

            if (

                candidate.id

                ==

                transform.after_place
            ):

                return candidate


        return None


    # ========================================================
    # NOVELTY
    # ========================================================

    def action_novelty(
        self,
        place,
        action
    ):

        if place is None:

            return 1.0


        count = 0


        for transform in self.transforms:

            if (

                transform.before_place
                ==
                place.id

                and

                action
                in
                transform.action_counts

            ):

                count += (

                    transform.action_counts[
                        action
                    ]
                )


        return (

            1.0

            /

            math.sqrt(
                1.0
                +
                count
            )
        )


    # ========================================================
    # PREDICTION UNCERTAINTY
    # ========================================================

    def prediction_uncertainty(
        self,
        place,
        action
    ):

        transform = (
            self.predict_action(

                place,

                action
            )
        )


        if transform is None:

            return 1.0


        return clamp(

            0.55
            *
            transform.error

            +

            0.45
            *
            (
                1.0
                -
                transform.stability
            ),

            0.0,

            1.0
        )


    # ========================================================
    # DANGER
    # ========================================================

    def danger_score(
        self,
        body,
        context,
        action
    ):

        danger = 0.0


        # ----------------------------------------------------
        # MONSTER
        # body[9]
        # ----------------------------------------------------

        monster_distance = (
            body[9]
            *
            600.0
        )


        if (
            monster_distance
            <
            80
        ):

            danger += 1.0

        elif (
            monster_distance
            <
            150
        ):

            danger += 0.55

        elif (
            monster_distance
            <
            250
        ):

            danger += 0.20


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        speed = math.sqrt(

            body[2] ** 2

            +

            body[3] ** 2
        )


        if speed > 0.85:

            danger += 0.15


        # ----------------------------------------------------
        # WALL
        # ----------------------------------------------------

        x = (
            body[0]
            *
            560.0
        )

        y = (
            body[1]
            *
            300.0
        )


        wall_distance = min(

            abs(
                x
                -
                WORLD_LEFT
            ),

            abs(
                x
                -
                WORLD_RIGHT
            ),

            abs(
                y
                -
                WORLD_BOTTOM
            ),

            abs(
                y
                -
                WORLD_TOP
            )
        )


        if (
            wall_distance
            <
            25
        ):

            danger += 0.9

        elif (
            wall_distance
            <
            50
        ):

            danger += 0.35


        # ----------------------------------------------------
        # JUMP
        # ----------------------------------------------------

        if action == ACTION_JUMP:

            danger *= 0.65


        # ----------------------------------------------------
        # BRAKE
        # ----------------------------------------------------

        if action == ACTION_BRAKE:

            danger *= 0.80


        return clamp(

            danger,

            0.0,

            1.5
        )


    # ========================================================
    # GOAL DIRECTION
    # ========================================================

    def goal_direction_score(
        self,
        body,
        action
    ):

        x = (
            body[0]
            *
            560.0
        )

        y = (
            body[1]
            *
            300.0
        )

        vx = (
            body[2]
            *
            MAX_SPEED
        )

        vy = (
            body[3]
            *
            13.0
        )


        before_distance = math.sqrt(

            (
                GOAL_X
                -
                x
            ) ** 2

            +

            (
                GOAL_Y
                -
                y
            ) ** 2
        )


        predicted_vx = vx


        if action == ACTION_LEFT:

            predicted_vx -= (
                GROUND_ACCEL
            )

        elif action == ACTION_RIGHT:

            predicted_vx += (
                GROUND_ACCEL
            )

        elif action == ACTION_BRAKE:

            predicted_vx *= 0.2

        elif action == ACTION_WAIT:

            predicted_vx *= 0.9


        predicted_x = (
            x
            +
            predicted_vx
            *
            3.0
        )


        predicted_y = (
            y
            +
            vy
            *
            3.0
        )


        after_distance = math.sqrt(

            (
                GOAL_X
                -
                predicted_x
            ) ** 2

            +

            (
                GOAL_Y
                -
                predicted_y
            ) ** 2
        )


        improvement = (

            before_distance
            -
            after_distance
        )


        return clamp(

            improvement
            /
            100.0,

            -1.0,

            1.0
        )


    # ========================================================
    # GATE SCORE
    # ========================================================

    def gate_score(
        self,
        body,
        action
    ):

        gate_open = (
            body[10]
            >
            0.5
        )


        if gate_open:

            return 0.0


        x = (
            body[0]
            *
            560.0
        )

        y = (
            body[1]
            *
            300.0
        )


        distance = math.sqrt(

            x * x
            +
            y * y
        )


        if (
            distance
            >
            150
        ):

            return 0.0


        score = 0.0


        if action == ACTION_RIGHT:

            score += 0.8

        elif action == ACTION_LEFT:

            score += 0.3

        elif action == ACTION_JUMP:

            score += 0.1


        return score


    # ========================================================
    # ONE STEP EVALUATION
    # ========================================================

    def evaluate_action(
        self,
        image,
        body,
        context,
        action
    ):

        place, place_score = (
            self.current_place(

                image,

                body,

                context
            )
        )


        if place is None:

            novelty = 1.0

            prediction = 0.5

            uncertainty = 1.0

        else:

            novelty = (
                self.action_novelty(

                    place,

                    action
                )
            )


            transform = (
                self.predict_action(

                    place,

                    action
                )
            )


            if transform is None:

                prediction = 0.5

            else:

                prediction = (

                    transform.stability

                    -

                    transform.error
                )


            uncertainty = (
                self.prediction_uncertainty(

                    place,

                    action
                )
            )


        danger = (
            self.danger_score(

                body,

                context,

                action
            )
        )


        goal = (
            self.goal_direction_score(

                body,

                action
            )
        )


        gate = (
            self.gate_score(

                body,

                action
            )
        )


        score = (

            self.prediction_weight
            *
            prediction

            +

            self.novelty_weight
            *
            novelty

            +

            self.stability_weight
            *
            uncertainty

            +

            goal
            *
            0.45

            +

            gate

            -

            self.danger_weight
            *
            danger
        )


        return {

            "action":
                action,

            "score":
                score,

            "prediction":
                prediction,

            "novelty":
                novelty,

            "uncertainty":
                uncertainty,

            "danger":
                danger,

            "goal":
                goal,

            "gate":
                gate,

            "place_score":
                place_score
        }


    # ========================================================
    # DREAM SIMULATION
    # ========================================================

    def dream_simulation(
        self,
        image,
        body,
        context,
        first_action
    ):

        """
        Internal simulation.

        実際のWorldは動かさない。

        記憶されたTransformationを使って
        「もしこの行動をしたら」
        を仮想的に展開する。
        """

        root_place, _ = (
            self.current_place(

                image,

                body,

                context
            )
        )


        if root_place is None:

            return {

                "score":
                    0.0,

                "depth":
                    0,

                "places":
                    []
            }


        current_place = root_place

        total_score = 0.0

        discount = 1.0

        visited = []

        action = first_action


        for depth in range(
            self.planning_depth
        ):

            transform = (
                self.predict_action(

                    current_place,

                    action
                )
            )


            if transform is None:

                # 未知の未来
                total_score += (

                    0.25
                    *
                    discount
                )

                break


            prediction = (

                transform.stability

                -

                transform.error
            )


            novelty = (

                1.0
                /
                math.sqrt(
                    1.0
                    +
                    transform.visits
                )
            )


            uncertainty = (
                self.prediction_uncertainty(

                    current_place,

                    action
                )
            )


            local_score = (

                self.prediction_weight
                *
                prediction

                +

                self.novelty_weight
                *
                novelty

                +

                0.50
                *
                uncertainty
            )


            total_score += (

                local_score
                *
                discount
            )


            visited.append(
                transform.after_place
            )


            next_place = None


            for place in self.places:

                if (

                    place.id

                    ==

                    transform.after_place
                ):

                    next_place = place

                    break


            if next_place is None:

                break


            current_place = next_place


            # 次の行動は、
            # 未来で最も安定した行動を選ぶ

            possible = [

                t

                for t in self.transforms

                if (

                    t.before_place
                    ==
                    current_place.id
                )
            ]


            if possible:

                possible.sort(

                    key=lambda t:

                        (

                            t.stability
                            -
                            t.error
                        ),

                    reverse=True
                )

                action = (
                    possible[0]
                    .primary_action
                )

            else:

                action = random.choice(
                    ACTIONS
                )


            discount *= 0.72


        return {

            "score":
                total_score,

            "depth":
                len(visited),

            "places":
                visited
        }


    # ========================================================
    # MULTI STEP PLAN
    # ========================================================

    def plan_action(
        self,
        image,
        body,
        context
    ):

        best_action = None

        best_score = -999999.0


        for action in ACTIONS:

            dream = (
                self.dream_simulation(

                    image,

                    body,

                    context,

                    action
                )
            )


            immediate = (
                self.evaluate_action(

                    image,

                    body,

                    context,

                    action
                )
            )


            total = (

                dream["score"]

                +

                immediate["score"]
                *
                0.8
            )


            if total > best_score:

                best_score = total

                best_action = action


        return best_action


    # ========================================================
    # SELECT ACTION
    # ========================================================

    def select_action(
        self,
        image,
        body,
        context
    ):

        planned = self.plan_action(

            image,

            body,

            context
        )


        evaluations = [

            self.evaluate_action(

                image,

                body,

                context,

                action
            )

            for action in ACTIONS
        ]


        for evaluation in evaluations:

            if (

                evaluation["action"]

                ==

                planned
            ):

                evaluation["score"] += 0.75


        evaluations.sort(

            key=lambda x:
                x["score"],

            reverse=True
        )


        # ----------------------------------------------------
        # EXPLORATION
        # ----------------------------------------------------

        if random.random() < self.exploration:

            top = evaluations[

                :

                min(
                    3,
                    len(
                        evaluations
                    )
                )
            ]

            selected = random.choice(
                top
            )

        else:

            selected = evaluations[0]


        return selected["action"]


    # ========================================================
    # REPLAY
    # ========================================================

    def replay(
        self,
        count=600
    ):

        if not self.transforms:

            return


        for _ in range(count):

            transform = random.choice(
                self.transforms
            )


            transform.energy = min(

                2.0,

                transform.energy
                +
                0.015
            )


            if transform.error < 0.25:

                transform.stability = clamp(

                    transform.stability
                    +
                    0.0015,

                    0.0,

                    1.0
                )

            else:

                transform.stability = clamp(

                    transform.stability
                    -
                    0.0002,

                    0.0,

                    1.0
                )


    # ========================================================
    # CONSOLIDATE
    # ========================================================

    def consolidate(self):

        if len(
            self.transforms
        ) < 2:

            return


        groups = {}


        for transform in self.transforms:

            key = (

                transform.before_place,

                transform.primary_action
            )


            groups.setdefault(
                key,
                []
            ).append(
                transform
            )


        for group in groups.values():

            if len(group) < 2:

                continue


            base = group[0]


            for other in group[1:]:

                if (

                    base.after_place

                    ==

                    other.after_place
                ):

                    base.delta = (

                        0.9
                        *
                        base.delta

                        +

                        0.1
                        *
                        other.delta
                    )


                    base.visits += (
                        other.visits
                    )


                    base.energy = min(

                        2.0,

                        base.energy
                        +
                        other.energy
                        *
                        0.1
                    )


    # ========================================================
    # DECAY
    # ========================================================

    def decay(self):

        self.visual.decay()


        for place in self.places:

            place.decay()


        for transform in self.transforms:

            transform.decay()


        self.exploration = max(

            0.12,

            self.exploration
            *
            0.9995
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    def statistics(self):

        error = (

            float(
                np.mean(
                    self.error_history
                )
            )

            if self.error_history

            else
            0.0
        )


        stability = (

            float(
                np.mean(
                    [

                        t.stability

                        for t
                        in
                        self.transforms

                    ]
                )
            )

            if self.transforms

            else
            0.0
        )


        uncertainty = 0.0


        if self.transforms:

            values = []


            for transform in self.transforms:

                if (

                    transform.before_place
                    <
                    len(
                        self.places
                    )
                ):

                    place = self.places[
                        transform.before_place
                    ]

                    values.append(

                        self.prediction_uncertainty(

                            place,

                            transform.primary_action
                        )
                    )


            if values:

                uncertainty = float(
                    np.mean(
                        values
                    )
                )


        return {

            "visual":
                self.visual.count(),

            "places":
                len(
                    self.places
                ),

            "transforms":
                len(
                    self.transforms
                ),

            "error":
                error,

            "stability":
                stability,

            "uncertainty":
                uncertainty,

            "exploration":
                self.exploration
        }


# ============================================================
# AGENT
# ============================================================

class Agent:

    def __init__(
        self,
        agent_id,
        color,
        world,
        model,
        vision
    ):

        self.id = agent_id

        self.color = color

        self.world = world

        self.model = model

        self.vision = vision

        self.turtle = turtle.Turtle(
            shape="turtle"
        )

        self.turtle.color(
            color
        )

        self.turtle.penup()

        self.turtle.speed(0)

        self.goal_reached = False

        self.reset()


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.x = (

            LEFT_ROOM_X1
            +
            55
            +
            self.id
            *
            30
        )

        self.y = (
            WORLD_BOTTOM
            +
            70
        )

        self.vx = 0.0

        self.vy = 0.0

        self.grounded = True

        self.jumps = 0

        self.heading = 1

        self.last_action = (
            ACTION_NONE
        )

        self.last_error = 0.0

        self.last_prediction = 0.0

        self.last_danger = 0.0

        self.last_reset = -999

        self.steps = 0

        self.goal_reached = False

        self.turtle.goto(
            self.x,
            self.y
        )


    # ========================================================
    # BODY STATE
    # ========================================================

    def body_state(self):

        gate_distance = (
            self.world.gate_distance(

                self.x,

                self.y
            )
        )


        goal_distance = math.sqrt(

            (
                self.x
                -
                GOAL_X
            ) ** 2

            +

            (
                self.y
                -
                GOAL_Y
            ) ** 2
        )


        monster_distance = math.sqrt(

            (
                self.x
                -
                self.vision.monster.x
            ) ** 2

            +

            (
                self.y
                -
                self.vision.monster.y
            ) ** 2
        )


        wall_left = (
            self.x
            -
            WORLD_LEFT
        )

        wall_right = (
            WORLD_RIGHT
            -
            self.x
        )

        wall_bottom = (
            self.y
            -
            WORLD_BOTTOM
        )

        wall_top = (
            WORLD_TOP
            -
            self.y
        )


        tunnel = float(

            self.world.inside_tunnel(

                self.x,

                self.y
            )
        )


        return np.asarray(

            [

                self.x / 560.0,

                self.y / 300.0,

                self.vx / MAX_SPEED,

                self.vy / 13.0,

                float(
                    self.grounded
                ),

                self.jumps / 2.0,

                self.heading,

                gate_distance / 100.0,

                goal_distance / 800.0,

                monster_distance / 600.0,

                float(
                    self.world.gate_open
                ),

                float(
                    self.goal_reached
                ),

                wall_left / 560.0,

                wall_right / 560.0,

                wall_bottom / 300.0,

                wall_top / 300.0,

                tunnel

            ],

            dtype=np.float32
        )


    # ========================================================
    # CONTEXT
    # ========================================================

    def local_context(self):

        monster_distance = math.sqrt(

            (
                self.x
                -
                self.vision.monster.x
            ) ** 2

            +

            (
                self.y
                -
                self.vision.monster.y
            ) ** 2
        )


        return np.asarray(

            [

                float(
                    self.world.gate_open
                ),

                float(
                    self.world.goal_reached
                ),

                monster_distance
                /
                600.0,

                float(
                    self.world.time
                    %
                    100
                )
                /
                100.0

            ],

            dtype=np.float32
        )


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        agents
    ):

        image_before = (
            self.vision.capture(

                agents,

                self.id
            )
        )


        body_before = (
            self.body_state()
        )


        context_before = (
            self.local_context()
        )


        # ----------------------------------------------------
        # RANDOM EXPLORATION
        # ----------------------------------------------------

        if random.random() < 0.12:

            action = random.choice(
                ACTIONS
            )

        else:

            action = (
                self.model.select_action(

                    image_before,

                    body_before,

                    context_before
                )
            )


        # ----------------------------------------------------
        # PHYSICS
        # ----------------------------------------------------

        self.world.step(

            self,

            action
        )


        # ----------------------------------------------------
        # AFTER
        # ----------------------------------------------------

        image_after = (
            self.vision.capture(

                agents,

                self.id
            )
        )


        body_after = (
            self.body_state()
        )


        context_after = (
            self.local_context()
        )


        result = self.model.learn(

            image_before,

            body_before,

            action,

            image_after,

            body_after,

            context_before,

            context_after
        )


        self.last_action = action

        self.last_error = (
            result["error"]
        )

        self.last_prediction = (

            1.0
            -
            result["error"]
        )


        self.last_danger = (
            self.model.danger_score(

                body_after,

                context_after,

                action
            )
        )


        self.steps += 1


        self.turtle.goto(

            self.x,

            self.y
        )


# ============================================================
# UI
# ============================================================

ui = turtle.Turtle(
    visible=False
)

ui.penup()

ui.speed(0)


graph = turtle.Turtle(
    visible=False
)

graph.penup()

graph.speed(0)


def write(
    x,
    y,
    text,
    size=10,
    color="#dddddd"
):

    ui.goto(
        x,
        y
    )

    ui.color(
        color
    )

    ui.write(

        text,

        font=(

            "Arial",

            size,

            "normal"
        )
    )


def line(
    drawer,
    x1,
    y1,
    x2,
    y2,
    color,
    width
):

    drawer.color(
        color
    )

    drawer.pensize(
        width
    )

    drawer.penup()

    drawer.goto(
        x1,
        y1
    )

    drawer.pendown()

    drawer.goto(
        x2,
        y2
    )

    drawer.penup()


# ============================================================
# WORLD DRAW
# ============================================================

def draw_world(
    world,
    monster
):

    d = world.drawer

    d.clear()


    # --------------------------------------------------------
    # OUTER WALLS
    # --------------------------------------------------------

    line(

        d,

        WORLD_LEFT,
        WORLD_BOTTOM,

        WORLD_LEFT,
        WORLD_TOP,

        "#44495a",

        6
    )


    line(

        d,

        WORLD_RIGHT,
        WORLD_BOTTOM,

        WORLD_RIGHT,
        WORLD_TOP,

        "#44495a",

        6
    )


    line(

        d,

        WORLD_LEFT,
        WORLD_TOP,

        WORLD_RIGHT,
        WORLD_TOP,

        "#44495a",

        6
    )


    line(

        d,

        WORLD_LEFT,
        WORLD_BOTTOM,

        WORLD_RIGHT,
        WORLD_BOTTOM,

        "#44495a",

        6
    )


    # --------------------------------------------------------
    # ROOM SEPARATION
    # --------------------------------------------------------

    line(

        d,

        -55,
        TUNNEL_Y2,

        -55,
        WORLD_TOP,

        "#777b8c",

        8
    )


    line(

        d,

        -55,
        WORLD_BOTTOM,

        -55,
        TUNNEL_Y1,

        "#777b8c",

        8
    )


    line(

        d,

        55,
        TUNNEL_Y2,

        55,
        WORLD_TOP,

        "#777b8c",

        8
    )


    line(

        d,

        55,
        WORLD_BOTTOM,

        55,
        TUNNEL_Y1,

        "#777b8c",

        8
    )


    # --------------------------------------------------------
    # TUNNEL
    # --------------------------------------------------------

    line(

        d,

        TUNNEL_X1,
        TUNNEL_Y1,

        TUNNEL_X2,
        TUNNEL_Y1,

        "#55a6aa",

        5
    )


    line(

        d,

        TUNNEL_X1,
        TUNNEL_Y2,

        TUNNEL_X2,
        TUNNEL_Y2,

        "#55a6aa",

        5
    )


    # --------------------------------------------------------
    # GATE
    # --------------------------------------------------------

    if not world.gate_open:

        d.goto(
            GATE_X,
            GATE_Y
        )

        d.dot(
            48,
            "#ffd447"
        )

        d.goto(
            GATE_X,
            GATE_Y
        )

        d.dot(
            28,
            "#ff7a33"
        )

        d.goto(

            GATE_X - 12,

            GATE_Y - 34
        )

        d.color(
            "#ffd447"
        )

        d.write(

            "PUSH",

            font=(

                "Arial",

                10,

                "bold"
            )
        )

    else:

        d.goto(
            GATE_X,
            GATE_Y
        )

        d.dot(
            30,
            "#55ff99"
        )

        d.goto(
            -80,
            105
        )

        d.color(
            "#55ff99"
        )

        d.write(

            "GATE OPEN",

            font=(

                "Arial",

                11,

                "bold"
            )
        )


    # --------------------------------------------------------
    # GOAL
    # --------------------------------------------------------

    d.goto(

        GOAL_X,

        GOAL_Y
    )

    d.dot(

        60,

        "#50ff8a"
    )


    d.goto(

        GOAL_X,

        GOAL_Y
    )

    d.dot(

        32,

        "#dffff0"
    )


    d.goto(

        GOAL_X - 25,

        GOAL_Y + 38
    )

    d.color(
        "#50ff8a"
    )

    d.write(

        "GOAL",

        font=(

            "Arial",

            13,

            "bold"
        )
    )


    # --------------------------------------------------------
    # RESET BLOCKS
    # --------------------------------------------------------

    for x, y in world.reset_blocks:

        d.goto(
            x,
            y
        )

        d.dot(
            9,
            "#b342ff"
        )


    # --------------------------------------------------------
    # MONSTER
    # --------------------------------------------------------

    d.goto(

        monster.x,

        monster.y
    )

    d.dot(

        MONSTER_RADIUS * 2,

        "#ff304f"
    )


    d.goto(

        monster.x,

        monster.y
    )

    d.dot(

        12,

        "#28050c"
    )


    if monster.target_id is not None:

        d.color(
            "#ff8c9a"
        )

        d.goto(

            monster.x - 22,

            monster.y + 28
        )

        d.write(

            f"TARGET A{monster.target_id}",

            font=(

                "Arial",

                7,

                "normal"
            )
        )


# ============================================================
# MODEL UI
# ============================================================

def draw_model(
    model,
    agents,
    episode,
    step,
    world,
    monster
):

    ui.clear()

    s = model.statistics()


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    write(

        -540,

        340,

        "PREDICTIVE TWO SPACES WORLD",

        16,

        "#ffffff"
    )


    write(

        -540,

        318,

        f"EPISODE "
        f"{episode + 1}/{MAX_EPISODES}"
        f"   STEP "
        f"{step}/{STEPS_PER_EPISODE}",

        10,

        "#55ffbb"
    )


    # --------------------------------------------------------
    # WORLD
    # --------------------------------------------------------

    write(

        -540,

        292,

        "LEFT SPACE",

        12,

        "#66cfff"
    )


    write(

        -540,

        273,

        "THIN TUNNEL",

        9,

        "#aaaaaa"
    )


    gate_color = (

        "#55ff99"

        if world.gate_open

        else

        "#ffd447"
    )


    write(

        -540,

        250,

        (

            "GATE : OPEN"

            if world.gate_open

            else

            "GATE : CLOSED"
        ),

        12,

        gate_color
    )


    write(

        -540,

        230,

        f"Gate pushes : "
        f"{world.gate_pushes}",

        9,

        "#dddddd"
    )


    write(

        -540,

        210,

        f"Reset events : "
        f"{world.reset_events}",

        9,

        "#bb66ff"
    )


    write(

        -540,

        190,

        (

            "GOAL REACHED"

            if world.goal_reached

            else

            "GOAL NOT REACHED"
        ),

        11,

        (

            "#50ff8a"

            if world.goal_reached

            else

            "#ff7777"
        )
    )


    # --------------------------------------------------------
    # MONSTER
    # --------------------------------------------------------

    write(

        180,

        320,

        "AI MONSTER",

        12,

        "#ff4055"
    )


    write(

        180,

        300,

        f"target : "
        f"{monster.target_id}",

        9,

        "#ff9aaa"
    )


    write(

        180,

        282,

        f"attacks : "
        f"{monster.attack_count}",

        9,

        "#ff9aaa"
    )


    write(

        180,

        264,

        f"vx={monster.vx:+.1f} "
        f"vy={monster.vy:+.1f}",

        9,

        "#ff9aaa"
    )


    # --------------------------------------------------------
    # WORLD MODEL
    # --------------------------------------------------------

    write(

        500,

        300,

        "WORLD MODEL",

        12,

        "#ffffff"
    )


    write(

        500,

        280,

        f"VisualCells : "
        f"{s['visual']}",

        9
    )


    write(

        500,

        262,

        f"Places      : "
        f"{s['places']}",

        9
    )


    write(

        500,

        244,

        f"Transforms  : "
        f"{s['transforms']}",

        9
    )


    write(

        500,

        226,

        f"Error       : "
        f"{s['error']:.3f}",

        9,

        "#ffbb55"
    )


    write(

        500,

        208,

        f"Stability   : "
        f"{s['stability']:.3f}",

        9,

        "#77ccff"
    )


    write(

        500,

        190,

        f"Uncertainty : "
        f"{s['uncertainty']:.3f}",

        9,

        "#ff88aa"
    )


    write(

        500,

        172,

        f"Explore     : "
        f"{s['exploration']:.3f}",

        9,

        "#55ffbb"
    )


    # --------------------------------------------------------
    # AGENTS
    # --------------------------------------------------------

    y = 145


    for agent in agents:

        write(

            -540,

            y,

            f"A{agent.id} "
            f"{ACTION_NAMES[agent.last_action]:<5} "
            f"x={agent.x:+.0f} "
            f"y={agent.y:+.0f} "
            f"vx={agent.vx:+.1f} "
            f"vy={agent.vy:+.1f}",

            9,

            agent.color
        )


        write(

            -540,

            y - 14,

            f"prediction={agent.last_prediction:.2f} "
            f"danger={agent.last_danger:.2f}",

            7,

            "#999999"
        )


        y -= 35


    # --------------------------------------------------------
    # PLACE GRAPH
    # --------------------------------------------------------

    write(

        500,

        135,

        "PLACE MEMORY",

        9,

        "#ffffff"
    )


    graph.clear()


    coords = {}


    max_places = min(

        35,

        len(
            model.places
        )
    )


    for i in range(
        max_places
    ):

        place = model.places[i]


        x = (

            500
            +
            (
                i % 7
            )
            *
            42
        )


        y = (

            105
            -
            (
                i // 7
            )
            *
            32
        )


        coords[i] = (

            x,

            y
        )


        radius = min(

            14,

            5
            +
            int(
                math.log1p(
                    place.visits
                )
            )
        )


        graph.goto(

            x,

            y
        )


        graph.dot(

            radius,

            "#5ea8ff"
        )


    for i in range(
        max_places
    ):

        place = model.places[i]


        if i not in coords:

            continue


        x1, y1 = coords[i]


        for target in list(

            place.transitions
        )[-5:]:


            if target not in coords:

                continue


            x2, y2 = coords[target]


            line(

                graph,

                x1,
                y1,

                x2,
                y2,

                "#334f77",

                1
            )


# ============================================================
# MAIN
# ============================================================

screen = turtle.Screen()


screen.setup(

    SCREEN_W,

    SCREEN_H
)


screen.bgcolor(

    "#080910"
)


screen.title(

    "Predictive Two Spaces / AI Monster"
)


screen.tracer(
    False
)


world = World()


monster = Monster(
    world
)


world.monster = monster


vision = VisualField(

    world,

    monster
)


model = WorldModel()


colors = [

    "#00eaff",

    "#57ff9b",

    "#ff9f43"
]


agents = [

    Agent(

        i,

        colors[i],

        world,

        model,

        vision

    )

    for i in range(
        NUM_AGENTS
    )
]


episode = 0

step = 0

finished = False


# ============================================================
# EPISODE RESET
# ============================================================

def reset_episode():

    global step

    step = 0

    world.reset()

    monster.reset()

    for agent in agents:

        agent.reset()


# ============================================================
# END EPISODE
# ============================================================

def end_episode():

    global episode

    global finished


    # --------------------------------------------------------
    # MEMORY REPLAY
    # --------------------------------------------------------

    model.replay(
        500
    )


    # --------------------------------------------------------
    # MEMORY CONSOLIDATION
    # --------------------------------------------------------

    model.consolidate()


    # --------------------------------------------------------
    # DECAY
    # --------------------------------------------------------

    model.decay()


    draw_model(

        model,

        agents,

        episode,

        step,

        world,

        monster
    )


    screen.update()


    episode += 1


    if (
        episode
        >=
        MAX_EPISODES
    ):

        finished = True


        write(

            -170,

            -335,

            "SIMULATION FINISHED",

            17,

            "#ffffff"
        )


        screen.update()

        return


    screen.ontimer(

        start_episode,

        900
    )


# ============================================================
# RUN STEP
# ============================================================

def run_step():

    global step


    if finished:

        return


    # --------------------------------------------------------
    # MONSTER
    # --------------------------------------------------------

    monster.update(
        agents
    )


    # --------------------------------------------------------
    # AGENTS
    # --------------------------------------------------------

    for agent in agents:

        if not agent.goal_reached:

            agent.step(
                agents
            )


    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    draw_world(

        world,

        monster
    )


    draw_model(

        model,

        agents,

        episode,

        step,

        world,

        monster
    )


    screen.update()


    step += 1


    # --------------------------------------------------------
    # CONTINUE
    # --------------------------------------------------------

    if (

        step
        <
        STEPS_PER_EPISODE

    ):

        screen.ontimer(

            run_step,

            18
        )

    else:

        screen.ontimer(

            end_episode,

            250
        )


# ============================================================
# START EPISODE
# ============================================================

def start_episode():

    if finished:

        return


    reset_episode()

    run_step()


# ============================================================
# START
# ============================================================

reset_episode()


draw_world(

    world,

    monster
)


draw_model(

    model,

    agents,

    episode,

    step,

    world,

    monster
)


screen.update()


screen.ontimer(

    start_episode,

    700
)


turtle.done()
