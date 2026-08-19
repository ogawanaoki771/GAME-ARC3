import turtle
import random
import math
import numpy as np

from collections import deque


# ============================================================
# NARROW TWO-SPACES WORLD
#
# ・狭い2空間
# ・複雑なトンネル
# ・ゲート
# ・リセットブロック
# ・障害物
# ・AIモンスター
# ・複数AIエージェント
# ・WorldModel
# ・報酬なし
# ============================================================


# ============================================================
# SCREEN
# ============================================================

SCREEN_W = 1240
SCREEN_H = 820


# ============================================================
# WORLD
# ============================================================

WORLD_LEFT = -520
WORLD_RIGHT = 520

WORLD_BOTTOM = -285
WORLD_TOP = 275


# ============================================================
# AGENTS
# ============================================================

NUM_AGENTS = 3

STEPS_PER_EPISODE = 900
MAX_EPISODES = 30


# ============================================================
# VISUAL
# ============================================================

OBS_W = 64
OBS_H = 40

CHANNELS = 5

PATCH = 4


# ============================================================
# MODEL
# ============================================================

VISUAL_SIM_THRESHOLD = 0.78
PLACE_SIM_THRESHOLD = 0.82
TRANSFORM_SIM_THRESHOLD = 0.73

MEMORY_WINDOW = 180


# ============================================================
# PHYSICS
# ============================================================

GRAVITY = 0.68

GROUND_ACCEL = 0.72
AIR_ACCEL = 0.48

GROUND_FRICTION = 0.86
AIR_FRICTION = 0.985

JUMP_POWER = 10.4

MAX_SPEED = 8.0


# ============================================================
# ROOM GEOMETRY
# ============================================================

# 左部屋
LEFT_ROOM_X1 = -495
LEFT_ROOM_X2 = -80

# 右部屋
RIGHT_ROOM_X1 = 80
RIGHT_ROOM_X2 = 495


# ============================================================
# CENTRAL TUNNEL
# ============================================================

TUNNEL_X1 = -80
TUNNEL_X2 = 80

TUNNEL_Y1 = -55
TUNNEL_Y2 = 55


# ============================================================
# GATE
# ============================================================

GATE_X = 0
GATE_Y = 0

GATE_RADIUS = 23
GATE_PUSH_DISTANCE = 38


# ============================================================
# GOAL
# ============================================================

GOAL_X = 420
GOAL_Y = 215

GOAL_RADIUS = 30


# ============================================================
# RESET BLOCK
# ============================================================

RESET_BLOCK_SIZE = 15

RESET_COOLDOWN = 35


# ============================================================
# MONSTER
# ============================================================

MONSTER_RADIUS = 22

MONSTER_SPEED = 4.7

MONSTER_ACCEL = 0.38

MONSTER_PUSH = 2.2

MONSTER_BOUNCE = 1.6

MONSTER_DETECTION_RANGE = 330

MONSTER_MEMORY = 80


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

    ACTION_NONE: "NONE",

    ACTION_LEFT: "LEFT",

    ACTION_RIGHT: "RIGHT",

    ACTION_JUMP: "JUMP",

    ACTION_BRAKE: "BRAKE",

    ACTION_WAIT: "WAIT"
}


ACTION_ACTIVITY_BIAS = {

    ACTION_NONE: -0.25,

    ACTION_LEFT: 0.20,

    ACTION_RIGHT: 0.20,

    ACTION_JUMP: 0.30,

    ACTION_BRAKE: 0.10,

    ACTION_WAIT: -0.05
}


# ============================================================
# UTILITY
# ============================================================

def clamp(x, lo, hi):

    return max(
        lo,
        min(hi, x)
    )


def distance(
    x1,
    y1,
    x2,
    y2
):

    return math.sqrt(
        (x2 - x1) ** 2
        +
        (y2 - y1) ** 2
    )


def l1_distance(a, b):

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
        * l1_distance(
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
            4.2
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

            0.92
            * self.feature

            +

            0.08
            * feature
        )

        self.visits += 1

        self.activation = min(
            2.0,
            self.activation + 0.035
        )

        self.energy = min(
            2.0,
            self.energy + 0.018
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

        for gx, gy, feature in self.patches(
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
                <
                VISUAL_SIM_THRESHOLD
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


        for _, _, feature in self.patches(
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
            len(cells)
            for cells
            in self.cells.values()
        )


    def decay(self):

        for cells in self.cells.values():

            for cell in cells:

                cell.decay()


# ============================================================
# PLACE
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

        visual_score = soft_distance(
            self.center,
            feature,
            5.0
        )

        state_score = soft_distance(
            self.signature,
            signature,
            6.0
        )

        return (
            0.60
            * visual_score

            +

            0.40
            * state_score
        )


    def absorb(
        self,
        feature,
        signature,
        state
    ):

        self.center = (

            0.93
            * self.center

            +

            0.07
            * feature
        )

        self.signature = (

            0.91
            * self.signature

            +

            0.09
            * signature
        )

        self.visits += 1

        self.energy = min(
            2.0,
            self.energy + 0.012
        )

        self.activation = min(
            2.0,
            self.activation + 0.04
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
# TRANSFORMATION
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
            action: 1
        }

        self.delta = np.asarray(
            delta,
            dtype=np.float32
        )

        self.visits = 1

        self.energy = 1.0

        self.stability = 0.2

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
            2.7
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
            * self.delta

            +

            0.12
            * delta
        )

        self.error = (

            0.90
            * self.error

            +

            0.10
            * error
        )

        self.stability = clamp(

            0.97
            * self.stability

            +

            0.03
            * (
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
            self.energy + 0.018
        )

        self.action_counts[action] = (

            self.action_counts.get(
                action,
                0
            )

            +

            1
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

        uncertainty = self.error

        instability = (

            1.0
            -
            self.stability
        )

        return (

            0.75
            * novelty

            +

            1.0
            * uncertainty

            +

            0.65
            * instability
        )


    def decay(self):

        self.energy *= 0.998


# ============================================================
# OBSTACLE
# ============================================================

class Obstacle:

    def __init__(
        self,
        x1,
        y1,
        x2,
        y2
    ):

        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2


    def contains(
        self,
        x,
        y,
        radius=0
    ):

        return (

            x
            >
            self.x1 - radius

            and

            x
            <
            self.x2 + radius

            and

            y
            >
            self.y1 - radius

            and

            y
            <
            self.y2 + radius
        )


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

        self.reset_blocks = []

        self.obstacles = []

        self.create_geometry()


    # ========================================================
    # GEOMETRY
    # ========================================================

    def create_geometry(self):

        self.create_reset_blocks()

        self.create_obstacles()


    # ========================================================
    # RESET BLOCKS
    # ========================================================

    def create_reset_blocks(self):

        self.reset_blocks.clear()


        # 左壁

        for y in range(
            WORLD_BOTTOM + 12,
            WORLD_TOP,
            RESET_BLOCK_SIZE
        ):

            self.reset_blocks.append(
                (
                    WORLD_LEFT + 5,
                    y
                )
            )


        # 右壁

        for y in range(
            WORLD_BOTTOM + 12,
            WORLD_TOP,
            RESET_BLOCK_SIZE
        ):

            self.reset_blocks.append(
                (
                    WORLD_RIGHT - 5,
                    y
                )
            )


        # 上壁

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


        # 下壁

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


    # ========================================================
    # OBSTACLES
    # ========================================================

    def create_obstacles(self):

        self.obstacles.clear()


        # ----------------------------------------------------
        # 左部屋
        # ----------------------------------------------------

        self.obstacles.append(
            Obstacle(
                -400,
                -205,
                -275,
                -155
            )
        )

        self.obstacles.append(
            Obstacle(
                -235,
                70,
                -110,
                115
            )
        )

        self.obstacles.append(
            Obstacle(
                -470,
                -30,
                -370,
                15
            )
        )


        # ----------------------------------------------------
        # 中央付近
        # ----------------------------------------------------

        self.obstacles.append(
            Obstacle(
                -105,
                -210,
                -72,
                -75
            )
        )

        self.obstacles.append(
            Obstacle(
                72,
                75,
                105,
                210
            )
        )


        # ----------------------------------------------------
        # 右部屋
        # ----------------------------------------------------

        self.obstacles.append(
            Obstacle(
                170,
                -190,
                285,
                -135
            )
        )

        self.obstacles.append(
            Obstacle(
                310,
                35,
                440,
                85
            )
        )

        self.obstacles.append(
            Obstacle(
                150,
                125,
                245,
                165
            )
        )


        # ----------------------------------------------------
        # ゴール前
        # ----------------------------------------------------

        self.obstacles.append(
            Obstacle(
                335,
                135,
                385,
                185
            )
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.time = 0

        self.gate_open = False

        self.goal_reached = False

        self.reset_events = 0

        self.gate_pushes = 0


    # ========================================================
    # ROOM
    # ========================================================

    def inside_left_room(
        self,
        x,
        y
    ):

        return (

            LEFT_ROOM_X1
            <
            x
            <
            LEFT_ROOM_X2

            and

            WORLD_BOTTOM
            <
            y
            <
            WORLD_TOP
        )


    def inside_right_room(
        self,
        x,
        y
    ):

        return (

            RIGHT_ROOM_X1
            <
            x
            <
            RIGHT_ROOM_X2

            and

            WORLD_BOTTOM
            <
            y
            <
            WORLD_TOP
        )


    def inside_tunnel(
        self,
        x,
        y
    ):

        return (

            TUNNEL_X1
            <=
            x
            <=
            TUNNEL_X2

            and

            TUNNEL_Y1
            <=
            y
            <=
            TUNNEL_Y2
        )


    # ========================================================
    # OBSTACLE COLLISION
    # ========================================================

    def collides_obstacle(
        self,
        x,
        y,
        radius=10
    ):

        for obstacle in self.obstacles:

            if obstacle.contains(
                x,
                y,
                radius
            ):

                return obstacle


        return None


    # ========================================================
    # RESET BLOCK
    # ========================================================

    def touches_reset_block(
        self,
        x,
        y
    ):

        margin = 11

        return (

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
        )


    # ========================================================
    # GATE
    # ========================================================

    def gate_distance(
        self,
        x,
        y
    ):

        return distance(
            x,
            y,
            GATE_X,
            GATE_Y
        )


    def push_gate(
        self,
        agent
    ):

        if self.gate_open:

            return


        d = self.gate_distance(
            agent.x,
            agent.y
        )


        if d < GATE_PUSH_DISTANCE:

            self.gate_open = True

            self.gate_pushes += 1


    # ========================================================
    # GOAL
    # ========================================================

    def check_goal(
        self,
        agent
    ):

        d = distance(
            agent.x,
            agent.y,
            GOAL_X,
            GOAL_Y
        )


        if d < GOAL_RADIUS:

            self.goal_reached = True

            agent.goal_reached = True


    # ========================================================
    # RESET AGENT
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
            * 35
        )

        agent.y = (

            WORLD_BOTTOM
            +
            65
        )

        agent.vx = 0

        agent.vy = 0

        agent.grounded = True

        agent.jumps = 0

        self.reset_events += 1


    # ========================================================
    # WORLD BOUNDARY
    # ========================================================

    def enforce_boundary(
        self,
        agent
    ):

        if (
            agent.x
            <
            WORLD_LEFT + 13
        ):

            agent.x = (
                WORLD_LEFT + 20
            )

            agent.vx = abs(
                agent.vx
            ) * 0.3


        if (
            agent.x
            >
            WORLD_RIGHT - 13
        ):

            agent.x = (
                WORLD_RIGHT - 20
            )

            agent.vx = -abs(
                agent.vx
            ) * 0.3


        if (
            agent.y
            <
            WORLD_BOTTOM + 13
        ):

            agent.y = (
                WORLD_BOTTOM + 20
            )

            agent.vy = abs(
                agent.vy
            ) * 0.3


        if (
            agent.y
            >
            WORLD_TOP - 13
        ):

            agent.y = (
                WORLD_TOP - 20
            )

            agent.vy = -abs(
                agent.vy
            ) * 0.3


    # ========================================================
    # OBSTACLE RESPONSE
    # ========================================================

    def resolve_obstacle(
        self,
        agent
    ):

        obstacle = self.collides_obstacle(
            agent.x,
            agent.y,
            12
        )


        if obstacle is None:

            return


        left_pen = abs(
            agent.x
            -
            obstacle.x1
        )

        right_pen = abs(
            agent.x
            -
            obstacle.x2
        )

        bottom_pen = abs(
            agent.y
            -
            obstacle.y1
        )

        top_pen = abs(
            agent.y
            -
            obstacle.y2
        )


        minimum = min(
            left_pen,
            right_pen,
            bottom_pen,
            top_pen
        )


        if minimum == left_pen:

            agent.x = (
                obstacle.x1
                -
                14
            )

            agent.vx *= -0.35


        elif minimum == right_pen:

            agent.x = (
                obstacle.x2
                +
                14
            )

            agent.vx *= -0.35


        elif minimum == bottom_pen:

            agent.y = (
                obstacle.y1
                -
                14
            )

            agent.vy = 0

            agent.grounded = True


        else:

            agent.y = (
                obstacle.y2
                +
                14
            )

            agent.vy = 0


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

                else

                AIR_ACCEL
            )

            agent.heading = -1


        elif action == ACTION_RIGHT:

            agent.vx += (

                GROUND_ACCEL
                if agent.grounded

                else

                AIR_ACCEL
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
                    * 0.78
                )

                agent.jumps += 1


        elif action == ACTION_BRAKE:

            agent.vx *= 0.18


        elif action == ACTION_WAIT:

            agent.vx *= 0.88


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        agent.vx = clamp(
            agent.vx,
            -MAX_SPEED,
            MAX_SPEED
        )


        agent.vx *= (

            GROUND_FRICTION
            if agent.grounded

            else

            AIR_FRICTION
        )


        agent.x += agent.vx


        # ----------------------------------------------------
        # GRAVITY
        # ----------------------------------------------------

        if not agent.grounded:

            agent.vy -= GRAVITY

            agent.y += agent.vy


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
                <
                22

                and

                abs(agent.y)
                <
                80
            ):

                if agent.x < 0:

                    agent.x = -25

                else:

                    agent.x = 25

                agent.vx *= -0.30


        # ----------------------------------------------------
        # TUNNEL
        # ----------------------------------------------------

        if self.inside_tunnel(
            agent.x,
            agent.y
        ):

            if (
                agent.y
                <
                TUNNEL_Y1
                +
                14
            ):

                agent.y = (
                    TUNNEL_Y1
                    +
                    15
                )

                agent.vy = abs(
                    agent.vy
                ) * 0.2


            if (
                agent.y
                >
                TUNNEL_Y2
                -
                14
            ):

                agent.y = (
                    TUNNEL_Y2
                    -
                    15
                )

                agent.vy = -abs(
                    agent.vy
                ) * 0.2


        # ----------------------------------------------------
        # OBSTACLE
        # ----------------------------------------------------

        self.resolve_obstacle(
            agent
        )


        # ----------------------------------------------------
        # BOUNDARY
        # ----------------------------------------------------

        self.enforce_boundary(
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

    def __init__(
        self,
        world
    ):

        self.world = world

        self.x = 300
        self.y = 20

        self.vx = 0
        self.vy = 0

        self.target_id = None

        self.memory = deque(
            maxlen=MONSTER_MEMORY
        )

        self.attack_count = 0

        self.last_target_distance = 9999

        self.turtle = turtle.Turtle(
            shape="circle"
        )

        self.turtle.color(
            "#ff3155"
        )

        self.turtle.penup()

        self.turtle.speed(0)


    def reset(self):

        self.x = 300
        self.y = 20

        self.vx = 0
        self.vy = 0

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

            d = distance(
                self.x,
                self.y,
                agent.x,
                agent.y
            )


            if d < MONSTER_DETECTION_RANGE:

                # 近いほど優先
                #
                # 速度が大きいAgentは
                # 少し狙いやすくする

                movement = math.sqrt(
                    agent.vx ** 2
                    +
                    agent.vy ** 2
                )

                score = (
                    d
                    -
                    movement
                    * 10
                )

                candidates.append(
                    (
                        score,
                        agent
                    )
                )


        if not candidates:

            self.target_id = None

            return None


        candidates.sort(
            key=lambda v: v[0]
        )


        target = candidates[0][1]

        self.target_id = target.id

        return target


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict_target(
        self,
        target
    ):

        d = distance(
            self.x,
            self.y,
            target.x,
            target.y
        )


        prediction_time = clamp(

            d
            /
            28.0,

            2.0,

            14.0
        )


        # 最近の行動も考慮

        vx = target.vx
        vy = target.vy


        if len(
            self.memory
        ) >= 3:

            recent = list(
                self.memory
            )[-3:]


            avg_vx = np.mean(
                [
                    v[2]
                    for v in recent
                ]
            )

            avg_vy = np.mean(
                [
                    v[3]
                    for v in recent
                ]
            )


            vx = (
                0.65
                * vx
                +
                0.35
                * avg_vx
            )

            vy = (
                0.65
                * vy
                +
                0.35
                * avg_vy
            )


        px = (
            target.x
            +
            vx
            * prediction_time
        )

        py = (
            target.y
            +
            vy
            * prediction_time
        )


        return (
            px,
            py
        )


    # ========================================================
    # WALL AVOIDANCE
    # ========================================================

    def avoid_obstacles(self):

        for obstacle in (
            self.world.obstacles
        ):

            closest_x = clamp(
                self.x,
                obstacle.x1,
                obstacle.x2
            )

            closest_y = clamp(
                self.y,
                obstacle.y1,
                obstacle.y2
            )


            d = distance(
                self.x,
                self.y,
                closest_x,
                closest_y
            )


            if d < 40:

                if d < 0.1:

                    dx = 1
                    dy = 0

                else:

                    dx = (
                        self.x
                        -
                        closest_x
                    ) / d

                    dy = (
                        self.y
                        -
                        closest_y
                    ) / d


                self.vx += (
                    dx
                    * 0.55
                )

                self.vy += (
                    dy
                    * 0.55
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

            # 探索
            self.vx += (
                math.sin(
                    self.world.time
                    * 0.025
                )
                * 0.12
            )

            self.vy += (
                math.cos(
                    self.world.time
                    * 0.035
                )
                * 0.12
            )


        else:

            px, py = (
                self.predict_target(
                    target
                )
            )


            dx = px - self.x

            dy = py - self.y


            d = math.sqrt(
                dx * dx
                +
                dy * dy
            )


            if d > 0.01:

                self.vx += (
                    dx
                    /
                    d
                    *
                    MONSTER_ACCEL
                )

                self.vy += (
                    dy
                    /
                    d
                    *
                    MONSTER_ACCEL
                )


            self.memory.append(
                (
                    target.x,
                    target.y,
                    target.vx,
                    target.vy
                )
            )


        # ----------------------------------------------------
        # Tunnel preference
        # ----------------------------------------------------

        if self.world.gate_open:

            if (
                self.x > 100
                and
                self.x < 180
            ):

                # トンネルへ入りやすくする
                self.vx -= 0.08


        # ----------------------------------------------------
        # Avoid obstacles
        # ----------------------------------------------------

        self.avoid_obstacles()


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        speed = math.sqrt(
            self.vx ** 2
            +
            self.vy ** 2
        )


        if speed > MONSTER_SPEED:

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


        self.vx *= 0.955

        self.vy *= 0.955


        # ----------------------------------------------------
        # BOUNDARY
        # ----------------------------------------------------

        if (
            self.x
            <
            RIGHT_ROOM_X1
            +
            22
        ):

            self.x = (
                RIGHT_ROOM_X1
                +
                22
            )

            self.vx *= -0.7


        if (
            self.x
            >
            RIGHT_ROOM_X2
            -
            22
        ):

            self.x = (
                RIGHT_ROOM_X2
                -
                22
            )

            self.vx *= -0.7


        if (
            self.y
            <
            WORLD_BOTTOM
            +
            22
        ):

            self.y = (
                WORLD_BOTTOM
                +
                22
            )

            self.vy *= -0.7


        if (
            self.y
            >
            WORLD_TOP
            -
            22
        ):

            self.y = (
                WORLD_TOP
                -
                22
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


        d = math.sqrt(
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

            d
            <
            collision_distance

            and

            d
            >
            0.01
        ):

            nx = dx / d

            ny = dy / d


            # 強いノックバック

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
                d
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


    # ========================================================
    # GRID
    # ========================================================

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
                OBS_W
                -
                1
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
                OBS_H
                -
                1
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


    # ========================================================
    # DOT
    # ========================================================

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

                    0
                    <=
                    xx
                    <
                    OBS_W

                    and

                    0
                    <=
                    yy
                    <
                    OBS_H
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


    # ========================================================
    # LINE
    # ========================================================

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
                gx2
                -
                gx1
            ),

            abs(
                gy2
                -
                gy1
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
                * t
            )


            gy = int(

                gy1

                +

                (
                    gy2
                    -
                    gy1
                )
                * t
            )


            if (

                0
                <=
                gx
                <
                OBS_W

                and

                0
                <=
                gy
                <
                OBS_H
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


    # ========================================================
    # CAPTURE
    # ========================================================

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
        # Walls
        # ----------------------------------------------------

        wall = img[:, :, 0]


        self.line(
            wall,
            WORLD_LEFT,
            WORLD_BOTTOM,
            WORLD_LEFT,
            WORLD_TOP,
            1.0
        )

        self.line(
            wall,
            WORLD_RIGHT,
            WORLD_BOTTOM,
            WORLD_RIGHT,
            WORLD_TOP,
            1.0
        )

        self.line(
            wall,
            WORLD_LEFT,
            WORLD_TOP,
            WORLD_RIGHT,
            WORLD_TOP,
            1.0
        )

        self.line(
            wall,
            WORLD_LEFT,
            WORLD_BOTTOM,
            WORLD_RIGHT,
            WORLD_BOTTOM,
            1.0
        )


        # ----------------------------------------------------
        # Tunnel
        # ----------------------------------------------------

        self.line(
            wall,
            TUNNEL_X1,
            TUNNEL_Y1,
            TUNNEL_X2,
            TUNNEL_Y1,
            0.95
        )

        self.line(
            wall,
            TUNNEL_X1,
            TUNNEL_Y2,
            TUNNEL_X2,
            TUNNEL_Y2,
            0.95
        )


        # ----------------------------------------------------
        # Obstacles
        # ----------------------------------------------------

        for obstacle in (
            self.world.obstacles
        ):

            self.line(
                wall,
                obstacle.x1,
                obstacle.y1,
                obstacle.x2,
                obstacle.y1,
                0.8
            )

            self.line(
                wall,
                obstacle.x2,
                obstacle.y1,
                obstacle.x2,
                obstacle.y2,
                0.8
            )

            self.line(
                wall,
                obstacle.x2,
                obstacle.y2,
                obstacle.x1,
                obstacle.y2,
                0.8
            )

            self.line(
                wall,
                obstacle.x1,
                obstacle.y2,
                obstacle.x1,
                obstacle.y1,
                0.8
            )


        # ----------------------------------------------------
        # Gate
        # ----------------------------------------------------

        if not self.world.gate_open:

            self.dot(
                img[:, :, 1],
                GATE_X,
                GATE_Y,
                1.0
            )


        # ----------------------------------------------------
        # Monster
        # ----------------------------------------------------

        self.dot(
            img[:, :, 1],
            self.monster.x,
            self.monster.y,
            1.0
        )


        # ----------------------------------------------------
        # Agents
        # ----------------------------------------------------

        for agent in agents:

            value = (

                1.0

                if
                agent.id
                ==
                viewer_id

                else

                0.65
            )


            self.dot(
                img[:, :, 2],
                agent.x,
                agent.y,
                value
            )


        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        self.dot(
            img[:, :, 3],
            GOAL_X,
            GOAL_Y,
            1.0
        )


        # ----------------------------------------------------
        # Tunnel / gate state
        # ----------------------------------------------------

        if self.world.gate_open:

            self.dot(
                img[:, :, 4],
                0,
                0,
                1.0
            )


        return img.reshape(-1)


# ============================================================
# WORLD MODEL
# ============================================================

class WorldModel:

    def __init__(self):

        self.visual = SpatialRepresentation()

        self.places = []

        self.transforms = []

        self.traces = deque(
            maxlen=900
        )

        self.error_history = deque(
            maxlen=600
        )

        self.next_place = 0

        self.next_transform = 0


    # ========================================================
    # PLACE SIGNATURE
    # ========================================================

    def place_signature(
        self,
        image,
        body,
        context
    ):

        visual = self.visual.feature_vector(
            image
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

        best_score = 0


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

            place = best

            place.absorb(
                feature,
                signature,
                state
            )

            created = False


        return (

            place,

            feature,

            signature,

            created
        )


    # ========================================================
    # TRANSFORM
    # ========================================================

    def find_transform(
        self,
        before,
        delta
    ):

        best = None

        best_score = 0


        for transform in (
            self.transforms
        ):

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

            transform = (
                TransformationCell(

                    self.next_transform,

                    before.id,

                    after.id,

                    action,

                    delta
                )
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

        self.visual.encode(
            image_before
        )

        self.visual.encode(
            image_after
        )


        state_before = np.concatenate(

            [

                self.visual.feature_vector(
                    image_before
                )[::4],

                body_before,

                context_before
            ]
        )


        state_after = np.concatenate(

            [

                self.visual.feature_vector(
                    image_after
                )[::4],

                body_after,

                context_after
            ]
        )


        before, _, _, _ = (
            self.encode_place(

                image_before,

                body_before,

                context_before,

                state_before
            )
        )


        after, _, _, _ = (
            self.encode_place(

                image_after,

                body_after,

                context_after,

                state_after
            )
        )


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


        self.traces.append(

            (
                before.id,

                action,

                after.id,

                error
            )
        )


        return {

            "before":
                before,

            "after":
                after,

            "transform":
                transform,

            "error":
                error
        }


    # ========================================================
    # SCORE
    # ========================================================

    def score_action(
        self,
        image,
        body,
        context,
        action
    ):

        feature, signature = (
            self.place_signature(

                image,

                body,

                context
            )
        )


        best = None

        best_score = 0


        for place in self.places:

            score = place.similarity(
                feature,
                signature
            )


            if score > best_score:

                best_score = score

                best = place


        score = (
            ACTION_ACTIVITY_BIAS[
                action
            ]
        )


        if best is None:

            return (
                score
                +
                random.random()
            )


        candidates = [

            t

            for t
            in self.transforms

            if (

                t.before_place
                ==
                best.id

                and

                action
                in
                t.action_counts
            )
        ]


        if not candidates:

            # 未知行動を優先

            score += 2.2

            score += (
                random.random()
                *
                0.5
            )


        else:

            curiosity = max(

                t.curiosity()

                for t
                in candidates
            )


            score += curiosity


            # 遷移が多い場所は
            # 探索価値が高い

            score += min(

                1.2,

                len(
                    best.transitions
                )
                *
                0.10
            )


        return score


    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def select_action(
        self,
        image,
        body,
        context
    ):

        scores = [

            self.score_action(

                image,

                body,

                context,

                action
            )

            for action
            in ACTIONS
        ]


        order = np.argsort(
            scores
        )[::-1]


        # 探索率

        if random.random() < 0.32:

            return random.choice(

                [
                    ACTIONS[
                        int(
                            order[0]
                        )
                    ],

                    ACTIONS[
                        int(
                            order[1]
                        )
                    ],

                    ACTIONS[
                        int(
                            order[2]
                        )
                    ]
                ]
            )


        return ACTIONS[
            int(
                order[0]
            )
        ]


    # ========================================================
    # REPLAY
    # ========================================================

    def replay(
        self,
        count=600
    ):

        if not self.transforms:

            return


        for _ in range(
            count
        ):

            transform = random.choice(
                self.transforms
            )


            transform.energy = min(

                2.0,

                transform.energy
                +
                0.015
            )


            if (
                transform.error
                >
                0.35
            ):

                transform.stability *= (
                    0.999
                )

            else:

                transform.stability = clamp(

                    transform.stability
                    +
                    0.001,

                    0,

                    1
                )


    # ========================================================
    # DECAY
    # ========================================================

    def decay(self):

        self.visual.decay()


        for place in self.places:

            place.decay()


        for transform in (
            self.transforms
        ):

            transform.decay()


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

            0
        )


        stability = (

            float(
                np.mean(
                    [
                        t.stability

                        for t
                        in self.transforms
                    ]
                )
            )

            if self.transforms

            else

            0
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
                stability
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
            35
        )

        self.y = (
            WORLD_BOTTOM
            +
            65
        )

        self.vx = 0

        self.vy = 0

        self.grounded = True

        self.jumps = 0

        self.heading = 1

        self.last_action = ACTION_NONE

        self.last_error = 0

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


        goal_distance = distance(

            self.x,
            self.y,

            GOAL_X,
            GOAL_Y
        )


        monster_distance = distance(

            self.x,
            self.y,

            self.vision.monster.x,
            self.vision.monster.y
        )


        obstacle_distance = 999


        for obstacle in (
            self.world.obstacles
        ):

            closest_x = clamp(

                self.x,

                obstacle.x1,

                obstacle.x2
            )

            closest_y = clamp(

                self.y,

                obstacle.y1,

                obstacle.y2
            )


            d = distance(

                self.x,
                self.y,

                closest_x,
                closest_y
            )


            obstacle_distance = min(

                obstacle_distance,

                d
            )


        return np.asarray(

            [

                self.x / 520,

                self.y / 285,

                self.vx / MAX_SPEED,

                self.vy / 13,

                float(
                    self.grounded
                ),

                self.jumps / 2,

                self.heading,

                gate_distance / 100,

                goal_distance / 800,

                monster_distance / 600,

                obstacle_distance / 300,

                float(
                    self.world.gate_open
                ),

                float(
                    self.goal_reached
                )

            ],

            dtype=np.float32
        )


    # ========================================================
    # CONTEXT
    # ========================================================

    def local_context(self):

        monster_distance = distance(

            self.x,
            self.y,

            self.vision.monster.x,
            self.vision.monster.y
        )


        obstacle_count = 0


        for obstacle in (
            self.world.obstacles
        ):

            if distance(

                self.x,
                self.y,

                clamp(
                    self.x,
                    obstacle.x1,
                    obstacle.x2
                ),

                clamp(
                    self.y,
                    obstacle.y1,
                    obstacle.y2
                )

            ) < 90:

                obstacle_count += 1


        return np.asarray(

            [

                float(
                    self.world.gate_open
                ),

                float(
                    self.world.goal_reached
                ),

                monster_distance / 600,

                obstacle_count / 5,

                float(
                    self.world.time % 120
                )
                /
                120

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
        # Random exploration
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
        # Physics
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

        self.steps += 1


        self.turtle.goto(
            self.x,
            self.y
        )


# ============================================================
# DRAWING
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
# DRAW WORLD
# ============================================================

def draw_world(
    world,
    monster
):

    d = world.drawer

    d.clear()


    # --------------------------------------------------------
    # Outer walls
    # --------------------------------------------------------

    line(

        d,

        WORLD_LEFT,
        WORLD_BOTTOM,

        WORLD_LEFT,
        WORLD_TOP,

        "#3d4252",

        6
    )


    line(

        d,

        WORLD_RIGHT,
        WORLD_BOTTOM,

        WORLD_RIGHT,
        WORLD_TOP,

        "#3d4252",

        6
    )


    line(

        d,

        WORLD_LEFT,
        WORLD_TOP,

        WORLD_RIGHT,
        WORLD_TOP,

        "#3d4252",

        6
    )


    line(

        d,

        WORLD_LEFT,
        WORLD_BOTTOM,

        WORLD_RIGHT,
        WORLD_BOTTOM,

        "#3d4252",

        6
    )


    # --------------------------------------------------------
    # Central walls
    # --------------------------------------------------------

    line(

        d,

        -80,
        TUNNEL_Y2,

        -80,
        WORLD_TOP,

        "#6b7184",

        7
    )


    line(

        d,

        -80,
        WORLD_BOTTOM,

        -80,
        TUNNEL_Y1,

        "#6b7184",

        7
    )


    line(

        d,

        80,
        TUNNEL_Y2,

        80,
        WORLD_TOP,

        "#6b7184",

        7
    )


    line(

        d,

        80,
        WORLD_BOTTOM,

        80,
        TUNNEL_Y1,

        "#6b7184",

        7
    )


    # --------------------------------------------------------
    # Tunnel
    # --------------------------------------------------------

    line(

        d,

        TUNNEL_X1,
        TUNNEL_Y1,

        TUNNEL_X2,
        TUNNEL_Y1,

        "#3d9ba5",

        5
    )


    line(

        d,

        TUNNEL_X1,
        TUNNEL_Y2,

        TUNNEL_X2,
        TUNNEL_Y2,

        "#3d9ba5",

        5
    )


    # --------------------------------------------------------
    # Obstacles
    # --------------------------------------------------------

    for obstacle in (
        world.obstacles
    ):

        line(

            d,

            obstacle.x1,
            obstacle.y1,

            obstacle.x2,
            obstacle.y1,

            "#72798b",

            5
        )


        line(

            d,

            obstacle.x2,
            obstacle.y1,

            obstacle.x2,
            obstacle.y2,

            "#72798b",

            5
        )


        line(

            d,

            obstacle.x2,
            obstacle.y2,

            obstacle.x1,
            obstacle.y2,

            "#72798b",

            5
        )


        line(

            d,

            obstacle.x1,
            obstacle.y2,

            obstacle.x1,
            obstacle.y1,

            "#72798b",

            5
        )


    # --------------------------------------------------------
    # Gate
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
            27,
            "#ff7035"
        )

        d.goto(
            -24,
            -38
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
            "#48ff9b"
        )

        d.goto(
            -65,
            82
        )

        d.color(
            "#48ff9b"
        )

        d.write(

            "OPEN",

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
        55,
        "#42ff8a"
    )

    d.goto(
        GOAL_X,
        GOAL_Y
    )

    d.dot(
        27,
        "#dffff0"
    )

    d.goto(
        GOAL_X - 24,
        GOAL_Y + 35
    )

    d.color(
        "#42ff8a"
    )

    d.write(

        "GOAL",

        font=(

            "Arial",

            12,

            "bold"
        )
    )


    # --------------------------------------------------------
    # RESET BLOCKS
    # --------------------------------------------------------

    for x, y in (
        world.reset_blocks
    ):

        d.goto(
            x,
            y
        )

        d.dot(
            8,
            "#a83cff"
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

        "#ff3155"
    )


    d.goto(
        monster.x,
        monster.y
    )

    d.dot(
        11,
        "#22060a"
    )


    if monster.target_id is not None:

        d.goto(

            monster.x - 25,

            monster.y + 27
        )

        d.color(
            "#ff8b9b"
        )

        d.write(

            f"A{monster.target_id}",

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


    stats = model.statistics()


    write(

        -500,
        335,

        "NARROW TWO-SPACES / WORLD MODEL AI",

        16,

        "#ffffff"
    )


    write(

        -500,
        313,

        f"EPISODE "
        f"{episode + 1}/{MAX_EPISODES}"
        f"   STEP "
        f"{step}/{STEPS_PER_EPISODE}",

        10,

        "#52ffba"
    )


    # --------------------------------------------------------
    # WORLD
    # --------------------------------------------------------

    write(

        -500,
        285,

        "SPACE : NARROW",

        11,

        "#5fd7ff"
    )


    write(

        -500,
        265,

        "TUNNEL : VERY THIN",

        10,

        "#72bfc8"
    )


    write(

        -500,
        242,

        (

            "GATE : OPEN"

            if world.gate_open

            else

            "GATE : CLOSED"
        ),

        11,

        (

            "#55ff99"

            if world.gate_open

            else

            "#ffd447"
        )
    )


    write(

        -500,
        220,

        f"Gate pushes : "
        f"{world.gate_pushes}",

        9
    )


    write(

        -500,
        202,

        f"Reset events : "
        f"{world.reset_events}",

        9,

        "#b76cff"
    )


    write(

        -500,
        182,

        (

            "GOAL REACHED"

            if world.goal_reached

            else

            "GOAL NOT REACHED"
        ),

        10,

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

        170,
        310,

        "PREDICTIVE MONSTER AI",

        11,

        "#ff4058"
    )


    write(

        170,
        290,

        f"Target : "
        f"A{monster.target_id}",

        9,

        "#ff9aaa"
    )


    write(

        170,
        272,

        f"Attacks : "
        f"{monster.attack_count}",

        9,

        "#ff9aaa"
    )


    write(

        170,
        254,

        f"vx={monster.vx:+.2f} "
        f"vy={monster.vy:+.2f}",

        9,

        "#ff9aaa"
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    write(

        500,
        290,

        f"VisualCells : "
        f"{stats['visual']}",

        9
    )


    write(

        500,
        272,

        f"Places : "
        f"{stats['places']}",

        9
    )


    write(

        500,
        254,

        f"Transforms : "
        f"{stats['transforms']}",

        9
    )


    write(

        500,
        236,

        f"Error : "
        f"{stats['error']:.3f}",

        9,

        "#ffbb55"
    )


    write(

        500,
        218,

        f"Stability : "
        f"{stats['stability']:.3f}",

        9,

        "#69cfff"
    )


    # --------------------------------------------------------
    # AGENTS
    # --------------------------------------------------------

    y = 145


    for agent in agents:

        write(

            -500,

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


        y -= 19


    # --------------------------------------------------------
    # MEMORY GRAPH
    # --------------------------------------------------------

    write(

        500,
        165,

        "PLACE MEMORY",

        9,

        "#ffffff"
    )


    graph.clear()


    coords = {}


    max_places = min(

        42,

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
                i
                %
                7
            )
            *
            42
        )


        y = (

            135
            -
            (
                i
                //
                7
            )
            *
            31
        )


        coords[i] = (
            x,
            y
        )


        radius = min(

            13,

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

            "#55aaff"
        )


    # --------------------------------------------------------
    # TRANSITIONS
    # --------------------------------------------------------

    for i in range(
        max_places
    ):

        place = model.places[i]


        if i not in coords:

            continue


        x1, y1 = coords[i]


        for target in list(

            place.transitions

        )[-6:]:


            if target not in coords:

                continue


            x2, y2 = coords[target]


            line(

                graph,

                x1,
                y1,

                x2,
                y2,

                "#304e75",

                1
            )


# ============================================================
# SCREEN
# ============================================================

screen = turtle.Screen()

screen.setup(

    SCREEN_W,

    SCREEN_H
)

screen.bgcolor(
    "#07090f"
)

screen.title(

    "Narrow Two Spaces / World Model / AI Monster"
)

screen.tracer(
    False
)


# ============================================================
# WORLD
# ============================================================

world = World()


# ============================================================
# MONSTER
# ============================================================

monster = Monster(
    world
)


# ============================================================
# VISION
# ============================================================

vision = VisualField(

    world,

    monster
)


# ============================================================
# MODEL
# ============================================================

model = WorldModel()


# ============================================================
# AGENTS
# ============================================================

colors = [

    "#00eaff",

    "#55ff9b",

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

    for i
    in range(
        NUM_AGENTS
    )
]


# ============================================================
# STATE
# ============================================================

episode = 0

step = 0

finished = False


# ============================================================
# RESET EPISODE
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


    # 報酬による学習ではない。
    #
    # 記憶の再活性化だけを行う。

    model.replay(
        700
    )

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


    if episode >= MAX_EPISODES:

        finished = True


        write(

            -170,

            -330,

            "SIMULATION FINISHED",

            17,

            "#ffffff"
        )


        screen.update()

        return


    screen.ontimer(

        start_episode,

        800
    )


# ============================================================
# MAIN STEP
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
    # NEXT
    # --------------------------------------------------------

    if step < STEPS_PER_EPISODE:

        screen.ontimer(

            run_step,

            16
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
