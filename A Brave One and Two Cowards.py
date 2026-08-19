import turtle
import random
import math
import numpy as np

from collections import deque


# ============================================================
# REWARD-FREE COOPERATIVE WORLD MODEL
#
# ・報酬関数なし
# ・予測誤差による自己教師学習
# ・Intrinsic Curiosity
# ・Multi-step World Model
# ・3 Agent shared memory
# ・協調行動
# ・Predictive / Intercept Monster
# ・狭い2空間
# ・トンネル
# ・ゲート
# ・リセットブロック
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
CHANNELS = 6
PATCH = 4


# ============================================================
# WORLD MODEL
# ============================================================

VISUAL_SIM_THRESHOLD = 0.78
PLACE_SIM_THRESHOLD = 0.80
TRANSFORM_SIM_THRESHOLD = 0.72

MEMORY_WINDOW = 180
MODEL_HISTORY = 1200

PREDICTION_HORIZONS = [1, 3, 6]


# ============================================================
# PHYSICS
# ============================================================

GRAVITY = 0.0

GROUND_ACCEL = 0.72
AIR_ACCEL = 0.48

GROUND_FRICTION = 0.86
AIR_FRICTION = 0.985

JUMP_POWER = 10.4

MAX_SPEED = 8.0


# ============================================================
# ROOM GEOMETRY
# ============================================================

LEFT_ROOM_X1 = -495
LEFT_ROOM_X2 = -80

RIGHT_ROOM_X1 = 80
RIGHT_ROOM_X2 = 495


TUNNEL_X1 = -80
TUNNEL_X2 = 80

TUNNEL_Y1 = -55
TUNNEL_Y2 = 55


# ============================================================
# GATE
# ============================================================

GATE_X = 0
GATE_Y = 0

GATE_PUSH_DISTANCE = 38


# ============================================================
# GOAL
# ============================================================

GOAL_X = 420
GOAL_Y = 215

GOAL_RADIUS = 30


# ============================================================
# RESET
# ============================================================

RESET_BLOCK_SIZE = 15
RESET_COOLDOWN = 35


# ============================================================
# MONSTER
# ============================================================

MONSTER_RADIUS = 22

MONSTER_SPEED = 5.1
MONSTER_ACCEL = 0.46

MONSTER_PUSH = 2.5
MONSTER_BOUNCE = 1.8

MONSTER_DETECTION_RANGE = 390
MONSTER_MEMORY = 120


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
    ACTION_NONE: -0.20,
    ACTION_LEFT: 0.15,
    ACTION_RIGHT: 0.15,
    ACTION_JUMP: 0.22,
    ACTION_BRAKE: 0.08,
    ACTION_WAIT: -0.02
}


# ============================================================
# UTILITY
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def l1_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.shape != b.shape:
        n = min(a.size, b.size)
        a = a.reshape(-1)[:n]
        b = b.reshape(-1)[:n]

    return float(np.mean(np.abs(a - b)))


def l2_distance(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    n = min(len(a), len(b))

    if n == 0:
        return 0.0

    return float(
        np.mean(
            (a[:n] - b[:n]) ** 2
        )
    )


def soft_distance(a, b, scale=4.0):
    return math.exp(
        -scale *
        l1_distance(a, b)
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

    def similarity(self, feature):

        return soft_distance(
            self.feature,
            feature,
            4.2
        )

    def update(self, feature):

        feature = np.asarray(
            feature,
            dtype=np.float32
        )

        self.feature = (
            0.92 * self.feature +
            0.08 * feature
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

    def patches(self, image):

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

    def encode(self, image):

        for gx, gy, feature in self.patches(image):

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
                best_score < VISUAL_SIM_THRESHOLD
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
                ).append(cell)

            else:

                best.update(feature)

    def feature_vector(self, image):

        values = []

        for _, _, feature in self.patches(image):

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
            for cells in self.cells.values()
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

        self.action_counts = {}

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
            0.58 * visual_score +
            0.42 * state_score
        )

    def absorb(
        self,
        feature,
        signature,
        state,
        action=None
    ):

        self.center = (
            0.93 * self.center +
            0.07 * feature
        )

        self.signature = (
            0.91 * self.signature +
            0.09 * signature
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

        if action is not None:

            self.action_counts[action] = (
                self.action_counts.get(
                    action,
                    0
                ) + 1
            )

        self.state_history.append(
            np.asarray(
                state,
                dtype=np.float32
            )
        )

    def connect(self, target):

        self.transitions[target] = (
            self.transitions.get(
                target,
                0
            ) + 1
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

        self.stability = 0.20
        self.error = 1.0

        self.history = deque(
            maxlen=MEMORY_WINDOW
        )

    def similarity(
        self,
        delta,
        before
    ):

        if self.before_place != before:
            return 0.0

        return soft_distance(
            self.delta,
            delta,
            2.7
        )

    def predict(self):

        return self.delta.copy()

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
            0.88 * self.delta +
            0.12 * delta
        )

        self.error = (
            0.90 * self.error +
            0.10 * error
        )

        self.stability = clamp(
            0.97 * self.stability +
            0.03 * (
                1.0 - error
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
            ) + 1
        )

        self.history.append(
            error
        )

    def curiosity(self):

        novelty = (
            1.0 /
            math.sqrt(
                max(
                    1,
                    self.visits
                )
            )
        )

        uncertainty = clamp(
            self.error,
            0.0,
            2.0
        )

        instability = (
            1.0 -
            self.stability
        )

        return (
            0.70 * novelty +
            0.95 * uncertainty +
            0.65 * instability
        )

    def decay(self):

        self.energy *= 0.998


# ============================================================
# EXPERIENCE
# ============================================================

class Experience:

    def __init__(
        self,
        state_before,
        action,
        state_after,
        place_before,
        place_after,
        error
    ):

        self.before = np.asarray(
            state_before,
            dtype=np.float32
        )

        self.action = action

        self.after = np.asarray(
            state_after,
            dtype=np.float32
        )

        self.place_before = place_before
        self.place_after = place_after

        self.error = error

        self.age = 0

        self.replays = 0


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
            x > self.x1 - radius and
            x < self.x2 + radius and
            y > self.y1 - radius and
            y < self.y2 + radius
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

    def create_geometry(self):

        self.create_reset_blocks()
        self.create_obstacles()

    def create_reset_blocks(self):

        self.reset_blocks.clear()

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

            self.reset_blocks.append(
                (
                    x,
                    WORLD_BOTTOM + 5
                )
            )

    def create_obstacles(self):

        self.obstacles.clear()

        self.obstacles.extend(
            [

                Obstacle(
                    -400,
                    -205,
                    -275,
                    -155
                ),

                Obstacle(
                    -235,
                    70,
                    -110,
                    115
                ),

                Obstacle(
                    -470,
                    -30,
                    -370,
                    15
                ),

                Obstacle(
                    -105,
                    -210,
                    -72,
                    -75
                ),

                Obstacle(
                    72,
                    75,
                    105,
                    210
                ),

                Obstacle(
                    170,
                    -190,
                    285,
                    -135
                ),

                Obstacle(
                    310,
                    35,
                    440,
                    85
                ),

                Obstacle(
                    150,
                    125,
                    245,
                    165
                ),

                Obstacle(
                    335,
                    135,
                    385,
                    185
                )
            ]
        )

    def reset(self):

        self.time = 0

        self.gate_open = False
        self.goal_reached = False

        self.reset_events = 0
        self.gate_pushes = 0

    def inside_tunnel(self, x, y):

        return (
            TUNNEL_X1 <= x <= TUNNEL_X2 and
            TUNNEL_Y1 <= y <= TUNNEL_Y2
        )

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

    def touches_reset_block(
        self,
        x,
        y
    ):

        margin = 11

        return (
            abs(x - WORLD_LEFT) < margin or
            abs(x - WORLD_RIGHT) < margin or
            abs(y - WORLD_BOTTOM) < margin or
            abs(y - WORLD_TOP) < margin
        )

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

    def push_gate(self, agent):

        if self.gate_open:
            return

        if self.gate_distance(
            agent.x,
            agent.y
        ) < GATE_PUSH_DISTANCE:

            self.gate_open = True
            self.gate_pushes += 1

    def check_goal(self, agent):

        if distance(
            agent.x,
            agent.y,
            GOAL_X,
            GOAL_Y
        ) < GOAL_RADIUS:

            self.goal_reached = True
            agent.goal_reached = True

    def reset_agent_position(
        self,
        agent
    ):

        agent.x = (
            LEFT_ROOM_X1 +
            55 +
            agent.id * 35
        )

        agent.y = (
            WORLD_BOTTOM +
            65
        )

        agent.vx = 0
        agent.vy = 0

        agent.grounded = True
        agent.jumps = 0

        self.reset_events += 1

    def enforce_boundary(self, agent):

        if agent.x < WORLD_LEFT + 13:

            agent.x = WORLD_LEFT + 20
            agent.vx = abs(agent.vx) * 0.3

        if agent.x > WORLD_RIGHT - 13:

            agent.x = WORLD_RIGHT - 20
            agent.vx = -abs(agent.vx) * 0.3

        if agent.y < WORLD_BOTTOM + 13:

            agent.y = WORLD_BOTTOM + 20
            agent.vy = abs(agent.vy) * 0.3

        if agent.y > WORLD_TOP - 13:

            agent.y = WORLD_TOP - 20
            agent.vy = -abs(agent.vy) * 0.3

    def resolve_obstacle(self, agent):

        obstacle = self.collides_obstacle(
            agent.x,
            agent.y,
            12
        )

        if obstacle is None:
            return

        distances = [

            (
                abs(agent.x - obstacle.x1),
                "left"
            ),

            (
                abs(agent.x - obstacle.x2),
                "right"
            ),

            (
                abs(agent.y - obstacle.y1),
                "bottom"
            ),

            (
                abs(agent.y - obstacle.y2),
                "top"
            )
        ]

        side = min(
            distances,
            key=lambda v: v[0]
        )[1]

        if side == "left":

            agent.x = obstacle.x1 - 14
            agent.vx *= -0.35

        elif side == "right":

            agent.x = obstacle.x2 + 14
            agent.vx *= -0.35

        elif side == "bottom":

            agent.y = obstacle.y1 - 14
            agent.vy = 0
            agent.grounded = True

        else:

            agent.y = obstacle.y2 + 14
            agent.vy = 0

    # ========================================================
    # ONE PHYSICS STEP
    # ========================================================

    def apply_action(
        self,
        agent,
        action
    ):

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

                agent.vy = JUMP_POWER * 0.78
                agent.jumps += 1

        elif action == ACTION_BRAKE:

            agent.vx *= 0.18

        elif action == ACTION_WAIT:

            agent.vx *= 0.88

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

        if not agent.grounded:

            agent.vy -= GRAVITY
            agent.y += agent.vy

        self.push_gate(agent)

        if not self.gate_open:

            if (
                abs(agent.x) < 22 and
                abs(agent.y) < 80
            ):

                if agent.x < 0:
                    agent.x = -25
                else:
                    agent.x = 25

                agent.vx *= -0.30

        if self.inside_tunnel(
            agent.x,
            agent.y
        ):

            if agent.y < TUNNEL_Y1 + 14:

                agent.y = TUNNEL_Y1 + 15
                agent.vy = abs(agent.vy) * 0.2

            if agent.y > TUNNEL_Y2 - 14:

                agent.y = TUNNEL_Y2 - 15
                agent.vy = -abs(agent.vy) * 0.2

        self.resolve_obstacle(agent)

        self.enforce_boundary(agent)

        if self.touches_reset_block(
            agent.x,
            agent.y
        ):

            if (
                self.time -
                agent.last_reset
                >
                RESET_COOLDOWN
            ):

                self.reset_agent_position(
                    agent
                )

                agent.last_reset = self.time

        self.check_goal(agent)

    def step_world(
        self,
        agents,
        actions
    ):

        # 全Agentを同じsimulation tickで更新する

        self.time += 1

        for agent, action in zip(
            agents,
            actions
        ):

            if not agent.goal_reached:

                self.apply_action(
                    agent,
                    action
                )


# ============================================================
# SHARED BLACKBOARD
# ============================================================

class SharedBlackboard:

    def __init__(self):

        self.data = {}

        self.step = 0

    def update(self, agent):

        self.data[agent.id] = {

            "x": agent.x,
            "y": agent.y,

            "vx": agent.vx,
            "vy": agent.vy,

            "heading": agent.heading,

            "grounded":
                agent.grounded,

            "goal":
                agent.goal_reached,

            "action":
                agent.last_action,

            "risk":
                agent.risk
        }

        self.step += 1

    def get(self, agent_id):

        return self.data.get(
            agent_id
        )

    def snapshot(self):

        return dict(
            self.data
        )

    def nearest_agent(
        self,
        x,
        y,
        exclude=None
    ):

        best = None
        best_d = 999999

        for aid, data in self.data.items():

            if aid == exclude:
                continue

            d = distance(
                x,
                y,
                data["x"],
                data["y"]
            )

            if d < best_d:

                best_d = d
                best = data

        return best


# ============================================================
# PREDICTIVE MONSTER
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

        self.memory = {
            i: deque(
                maxlen=MONSTER_MEMORY
            )
            for i in range(
                NUM_AGENTS
            )
        }

        self.attack_count = 0

        self.target_switches = 0

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

        for memory in self.memory.values():
            memory.clear()

        self.attack_count = 0
        self.target_switches = 0

        self.turtle.goto(
            self.x,
            self.y
        )

    def estimate_agent(
        self,
        agent
    ):

        memory = self.memory[
            agent.id
        ]

        vx = agent.vx
        vy = agent.vy

        ax = 0
        ay = 0

        if len(memory) >= 3:

            recent = list(memory)[-6:]

            vx_values = [
                item[2]
                for item in recent
            ]

            vy_values = [
                item[3]
                for item in recent
            ]

            vx_est = float(
                np.mean(vx_values)
            )

            vy_est = float(
                np.mean(vy_values)
            )

            vx = (
                0.65 * vx +
                0.35 * vx_est
            )

            vy = (
                0.65 * vy +
                0.35 * vy_est
            )

            if len(recent) >= 2:

                ax = (
                    recent[-1][2] -
                    recent[-2][2]
                )

                ay = (
                    recent[-1][3] -
                    recent[-2][3]
                )

        d = distance(
            self.x,
            self.y,
            agent.x,
            agent.y
        )

        prediction_time = clamp(
            d / 24.0,
            3.0,
            20.0
        )

        px = (
            agent.x +
            vx * prediction_time +
            0.5 * ax *
            prediction_time *
            prediction_time
        )

        py = (
            agent.y +
            vy * prediction_time +
            0.5 * ay *
            prediction_time *
            prediction_time
        )

        px = clamp(
            px,
            RIGHT_ROOM_X1 + 22,
            RIGHT_ROOM_X2 - 22
        )

        py = clamp(
            py,
            WORLD_BOTTOM + 22,
            WORLD_TOP - 22
        )

        return (
            px,
            py,
            prediction_time,
            vx,
            vy
        )

    def obstacle_penalty(
        self,
        x,
        y
    ):

        penalty = 0

        for obstacle in self.world.obstacles:

            closest_x = clamp(
                x,
                obstacle.x1,
                obstacle.x2
            )

            closest_y = clamp(
                y,
                obstacle.y1,
                obstacle.y2
            )

            d = distance(
                x,
                y,
                closest_x,
                closest_y
            )

            if d < 45:

                penalty += (
                    45 - d
                ) / 45

        return penalty

    def interception_score(
        self,
        agent
    ):

        px, py, horizon, vx, vy = (
            self.estimate_agent(
                agent
            )
        )

        predicted_distance = distance(
            self.x,
            self.y,
            px,
            py
        )

        current_distance = distance(
            self.x,
            self.y,
            agent.x,
            agent.y
        )

        movement = math.sqrt(
            vx * vx +
            vy * vy
        )

        # 高速移動中のAgentは
        # 将来位置が大きく変わるので優先

        score = (
            -predicted_distance
            - 0.22 * current_distance
            + 7.0 * movement
            + 4.0 * horizon
        )

        score -= (
            35 *
            self.obstacle_penalty(
                px,
                py
            )
        )

        return score

    def choose_target(
        self,
        agents
    ):

        candidates = []

        for agent in agents:

            if agent.goal_reached:
                continue

            d = distance(
                self.x,
                self.y,
                agent.x,
                agent.y
            )

            if d > MONSTER_DETECTION_RANGE:
                continue

            score = self.interception_score(
                agent
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
            key=lambda x: x[0],
            reverse=True
        )

        target = candidates[0][1]

        if (
            self.target_id is not None and
            self.target_id != target.id
        ):

            self.target_switches += 1

        self.target_id = target.id

        return target

    def avoid_obstacles(self):

        for obstacle in self.world.obstacles:

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

            if d < 55:

                if d < 0.1:

                    dx = 1
                    dy = 0

                else:

                    dx = (
                        self.x -
                        closest_x
                    ) / d

                    dy = (
                        self.y -
                        closest_y
                    ) / d

                strength = (
                    0.85 *
                    (
                        1 -
                        d / 55
                    )
                )

                self.vx += (
                    dx * strength
                )

                self.vy += (
                    dy * strength
                )

    def update(self, agents):

        target = self.choose_target(
            agents
        )

        if target is None:

            # 探索行動

            self.vx += (
                math.sin(
                    self.world.time *
                    0.025
                ) * 0.16
            )

            self.vy += (
                math.cos(
                    self.world.time *
                    0.035
                ) * 0.16
            )

        else:

            px, py, horizon, _, _ = (
                self.estimate_agent(
                    target
                )
            )

            dx = px - self.x
            dy = py - self.y

            d = math.sqrt(
                dx * dx +
                dy * dy
            )

            if d > 0.01:

                # 遠いほど予測点への
                # 直接迎撃

                strength = (
                    MONSTER_ACCEL
                    *
                    clamp(
                        d / 180,
                        0.55,
                        1.4
                    )
                )

                self.vx += (
                    dx / d *
                    strength
                )

                self.vy += (
                    dy / d *
                    strength
                )

            self.memory[
                target.id
            ].append(
                (
                    target.x,
                    target.y,
                    target.vx,
                    target.vy
                )
            )

        # tunnel-aware

        if self.world.gate_open:

            if (
                self.x > 80 and
                self.x < 180
            ):

                self.vx -= 0.10

        self.avoid_obstacles()

        speed = math.sqrt(
            self.vx * self.vx +
            self.vy * self.vy
        )

        if speed > MONSTER_SPEED:

            self.vx = (
                self.vx /
                speed *
                MONSTER_SPEED
            )

            self.vy = (
                self.vy /
                speed *
                MONSTER_SPEED
            )

        self.x += self.vx
        self.y += self.vy

        self.vx *= 0.955
        self.vy *= 0.955

        if (
            self.x <
            RIGHT_ROOM_X1 + 22
        ):

            self.x = (
                RIGHT_ROOM_X1 + 22
            )

            self.vx *= -0.7

        if (
            self.x >
            RIGHT_ROOM_X2 - 22
        ):

            self.x = (
                RIGHT_ROOM_X2 - 22
            )

            self.vx *= -0.7

        if (
            self.y <
            WORLD_BOTTOM + 22
        ):

            self.y = (
                WORLD_BOTTOM + 22
            )

            self.vy *= -0.7

        if (
            self.y >
            WORLD_TOP - 22
        ):

            self.y = (
                WORLD_TOP - 22
            )

            self.vy *= -0.7

        for agent in agents:

            self.collide(agent)

        self.turtle.goto(
            self.x,
            self.y
        )

    def collide(self, agent):

        dx = agent.x - self.x
        dy = agent.y - self.y

        d = math.sqrt(
            dx * dx +
            dy * dy
        )

        collision_distance = (
            MONSTER_RADIUS + 13
        )

        if (
            d <
            collision_distance
            and
            d > 0.01
        ):

            nx = dx / d
            ny = dy / d

            agent.vx += (
                nx * MONSTER_PUSH
            )

            agent.vy += (
                ny * MONSTER_PUSH
            )

            self.vx -= (
                nx * MONSTER_BOUNCE
            )

            self.vy -= (
                ny * MONSTER_BOUNCE
            )

            penetration = (
                collision_distance - d
            )

            agent.x += (
                nx * penetration
            )

            agent.y += (
                ny * penetration
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

    def to_grid(self, x, y):

        gx = int(
            (
                x - WORLD_LEFT
            )
            /
            (
                WORLD_RIGHT -
                WORLD_LEFT
            )
            *
            (
                OBS_W - 1
            )
        )

        gy = int(
            (
                y - WORLD_BOTTOM
            )
            /
            (
                WORLD_TOP -
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
                    0 <= xx < OBS_W and
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
            abs(gx2 - gx1),
            abs(gy2 - gy1),
            1
        )

        for i in range(
            n + 1
        ):

            t = i / n

            gx = int(
                gx1 +
                (
                    gx2 - gx1
                ) * t
            )

            gy = int(
                gy1 +
                (
                    gy2 - gy1
                ) * t
            )

            if (
                0 <= gx < OBS_W and
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
        viewer_id,
        blackboard
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
        # walls
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

        for obstacle in self.world.obstacles:

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
        # gate
        # ----------------------------------------------------

        if not self.world.gate_open:

            self.dot(
                img[:, :, 1],
                GATE_X,
                GATE_Y,
                1.0
            )

        # ----------------------------------------------------
        # monster
        # ----------------------------------------------------

        self.dot(
            img[:, :, 1],
            self.monster.x,
            self.monster.y,
            1.0
        )

        # ----------------------------------------------------
        # agents
        # ----------------------------------------------------

        for agent in agents:

            value = (
                1.0
                if agent.id == viewer_id
                else 0.65
            )

            self.dot(
                img[:, :, 2],
                agent.x,
                agent.y,
                value
            )

        # ----------------------------------------------------
        # goal
        # ----------------------------------------------------

        self.dot(
            img[:, :, 3],
            GOAL_X,
            GOAL_Y,
            1.0
        )

        # ----------------------------------------------------
        # gate state
        # ----------------------------------------------------

        if self.world.gate_open:

            self.dot(
                img[:, :, 4],
                0,
                0,
                1.0
            )

        # ----------------------------------------------------
        # monster predicted target
        # ----------------------------------------------------

        if self.monster.target_id is not None:

            target = next(
                (
                    a for a in agents
                    if a.id ==
                    self.monster.target_id
                ),
                None
            )

            if target is not None:

                px, py, _, _, _ = (
                    self.monster.estimate_agent(
                        target
                    )
                )

                self.dot(
                    img[:, :, 5],
                    px,
                    py,
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

        self.experiences = deque(
            maxlen=MODEL_HISTORY
        )

        self.error_history = deque(
            maxlen=600
        )

        self.information_history = deque(
            maxlen=600
        )

        self.next_place = 0
        self.next_transform = 0

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
            best_score <
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

    def find_transform(
        self,
        before,
        delta
    ):

        best = None
        best_score = 0

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
            score <
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
            state_after -
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

        experience = Experience(
            state_before,
            action,
            state_after,
            before.id,
            after.id,
            error
        )

        self.experiences.append(
            experience
        )

        return {
            "before": before,
            "after": after,
            "transform": transform,
            "error": error,
            "state_before":
                state_before,
            "state_after":
                state_after
        }

    # ========================================================
    # FIND CURRENT PLACE
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
        best_score = 0

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
            best_score,
            feature,
            signature
        )

    # ========================================================
    # PREDICT ACTION
    # ========================================================

    def action_prediction(
        self,
        place,
        action
    ):

        if place is None:

            return None, 1.0

        candidates = [

            t

            for t in self.transforms

            if (
                t.before_place ==
                place.id
                and
                action in
                t.action_counts
            )
        ]

        if not candidates:

            return None, 1.0

        candidates.sort(
            key=lambda t:
            (
                t.stability,
                t.energy,
                t.visits
            ),
            reverse=True
        )

        transform = candidates[0]

        return (
            transform,
            transform.error
        )

    # ========================================================
    # CURIOSITY
    # ========================================================

    def curiosity(
        self,
        place,
        action
    ):

        if place is None:

            return 2.5

        candidates = [

            t

            for t in self.transforms

            if (
                t.before_place ==
                place.id
                and
                action in
                t.action_counts
            )
        ]

        if not candidates:

            return 2.5

        best = max(
            candidates,
            key=lambda t:
            t.curiosity()
        )

        return best.curiosity()

    # ========================================================
    # MULTI STEP ROLLOUT
    # ========================================================

    def rollout_value(
        self,
        place,
        action
    ):

        if place is None:

            return 2.0

        transform, error = (
            self.action_prediction(
                place,
                action
            )
        )

        if transform is None:

            return 2.5

        next_place = next(
            (
                p for p in self.places
                if p.id ==
                transform.after_place
            ),
            None
        )

        if next_place is None:

            return 1.0

        branching = len(
            next_place.transitions
        )

        novelty = (
            1.0 /
            math.sqrt(
                max(
                    1,
                    next_place.visits
                )
            )
        )

        uncertainty = clamp(
            error,
            0,
            2
        )

        return (
            0.85 * novelty +
            0.75 * uncertainty +
            0.08 * branching
        )

    # ========================================================
    # ACTION SCORE
    # ========================================================

    def score_action(
        self,
        image,
        body,
        context,
        action,
        role_bias=0.0
    ):

        place, place_score, _, _ = (
            self.current_place(
                image,
                body,
                context
            )
        )

        score = ACTION_ACTIVITY_BIAS[
            action
        ]

        # 未知行動を積極的に試す

        if place is None:

            return (
                score +
                2.4 +
                random.random() * 0.5
            )

        candidates = [

            t

            for t in self.transforms

            if (
                t.before_place ==
                place.id
                and
                action in
                t.action_counts
            )
        ]

        if not candidates:

            score += 2.2

        else:

            score += max(
                t.curiosity()
                for t in candidates
            )

        # multi-step prediction

        score += (
            0.85 *
            self.rollout_value(
                place,
                action
            )
        )

        # 観測そのものの不確実性

        score += (
            0.55 *
            (
                1.0 -
                place_score
            )
        )

        score += role_bias

        # 完全ランダムではなく
        # deterministic + noise

        score += (
            random.random() *
            0.12
        )

        return score

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def select_action(
        self,
        image,
        body,
        context,
        role_biases=None
    ):

        if role_biases is None:

            role_biases = {
                action: 0
                for action in ACTIONS
            }

        scores = {}

        for action in ACTIONS:

            scores[action] = (
                self.score_action(
                    image,
                    body,
                    context,
                    action,
                    role_biases.get(
                        action,
                        0
                    )
                )
            )

        ranked = sorted(
            ACTIONS,
            key=lambda a:
            scores[a],
            reverse=True
        )

        # 以前のコードの
        # 上位3つランダム方式を改良

        temperature = 0.28

        if random.random() < 0.22:

            top = ranked[:4]

            weights = []

            for action in top:

                weights.append(
                    math.exp(
                        scores[action] /
                        temperature
                    )
                )

            total = sum(weights)

            r = random.random() * total

            acc = 0

            for action, weight in zip(
                top,
                weights
            ):

                acc += weight

                if r <= acc:

                    return (
                        action,
                        scores
                    )

        return (
            ranked[0],
            scores
        )

    # ========================================================
    # REPLAY
    # ========================================================

    def replay(
        self,
        count=700
    ):

        if not self.experiences:
            return

        for _ in range(count):

            experience = random.choice(
                list(
                    self.experiences
                )
            )

            transform = next(
                (
                    t
                    for t in self.transforms
                    if (
                        t.before_place ==
                        experience.place_before
                        and
                        t.after_place ==
                        experience.place_after
                    )
                ),
                None
            )

            if transform is None:
                continue

            # Rewardではない。
            #
            # 予測誤差を再学習して
            # dynamics modelの安定性を上げる。

            if experience.error > 0.35:

                transform.stability *= 0.9995

            else:

                transform.stability = clamp(
                    transform.stability +
                    0.0012,
                    0,
                    1
                )

            transform.energy = min(
                2.0,
                transform.energy +
                0.012
            )

            experience.replays += 1

    def decay(self):

        self.visual.decay()

        for place in self.places:
            place.decay()

        for transform in self.transforms:
            transform.decay()

    def statistics(self):

        error = (
            float(
                np.mean(
                    self.error_history
                )
            )
            if self.error_history
            else 0
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
            else 0
        )

        curiosity = (
            float(
                np.mean(
                    [
                        t.curiosity()
                        for t
                        in self.transforms
                    ]
                )
            )
            if self.transforms
            else 0
        )

        return {
            "visual":
                self.visual.count(),

            "places":
                len(self.places),

            "transforms":
                len(self.transforms),

            "experiences":
                len(self.experiences),

            "error":
                error,

            "stability":
                stability,

            "curiosity":
                curiosity
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
        vision,
        blackboard
    ):

        self.id = agent_id
        self.color = color

        self.world = world
        self.model = model
        self.vision = vision
        self.blackboard = blackboard

        self.turtle = turtle.Turtle(
            shape="turtle"
        )

        self.turtle.color(
            color
        )

        self.turtle.penup()
        self.turtle.speed(0)

        self.reset()

    def reset(self):

        self.x = (
            LEFT_ROOM_X1 +
            55 +
            self.id * 35
        )

        self.y = (
            WORLD_BOTTOM +
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

        self.role = "explorer"

        self.risk = 0.0

        self.predicted_monster_distance = 999

        self.turtle.goto(
            self.x,
            self.y
        )

    # ========================================================
    # COORDINATION
    # ========================================================

    def compute_risk(self):

        monster_d = distance(
            self.x,
            self.y,
            self.vision.monster.x,
            self.vision.monster.y
        )

        self.predicted_monster_distance = monster_d

        risk = clamp(
            1.0 -
            monster_d /
            MONSTER_DETECTION_RANGE,
            0,
            1
        )

        if abs(self.vx) > 5.5:
            risk += 0.10

        if (
            self.world.gate_open
            and
            self.x > 80
            and
            self.x < 180
        ):
            risk += 0.15

        self.risk = clamp(
            risk,
            0,
            1
        )

    def assign_role(
        self,
        agents
    ):

        self.compute_risk()

        distances_to_gate = [
            (
                distance(
                    a.x,
                    a.y,
                    GATE_X,
                    GATE_Y
                ),
                a.id
            )
            for a in agents
        ]

        distances_to_goal = [
            (
                distance(
                    a.x,
                    a.y,
                    GOAL_X,
                    GOAL_Y
                ),
                a.id
            )
            for a in agents
        ]

        gate_owner = min(
            distances_to_gate
        )[1]

        goal_owner = min(
            distances_to_goal
        )[1]

        if not self.world.gate_open:

            if self.id == gate_owner:

                self.role = "gate"

            else:

                self.role = "scout"

        elif self.risk > 0.62:

            self.role = "evasive"

        elif self.id == goal_owner:

            self.role = "goal"

        else:

            self.role = "scout"

    def role_biases(self):

        bias = {
            action: 0.0
            for action in ACTIONS
        }

        if self.role == "gate":

            # gate担当は中央方向を
            # 強く探索

            if self.x < 0:

                bias[
                    ACTION_RIGHT
                ] += 1.25

            if abs(self.y) > 65:

                bias[
                    ACTION_LEFT
                ] += 0.15

        elif self.role == "goal":

            # 右方向への進行を優先するが、
            # 報酬ではなく役割prior

            bias[
                ACTION_RIGHT
            ] += 0.55

        elif self.role == "evasive":

            # Monsterから離れるための
            # 左右方向切り替え

            monster_x = (
                self.vision.monster.x
            )

            if self.x < monster_x:

                bias[
                    ACTION_LEFT
                ] += 0.75

            else:

                bias[
                    ACTION_RIGHT
                ] += 0.75

            if not self.grounded:

                bias[
                    ACTION_JUMP
                ] += 0.25

            else:

                bias[
                    ACTION_JUMP
                ] += 0.45

        elif self.role == "scout":

            bias[
                ACTION_WAIT
            ] += 0.05

            bias[
                ACTION_JUMP
            ] += 0.12

        return bias

    # ========================================================
    # BODY
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

        for obstacle in self.world.obstacles:

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
                ),

                self.risk,

                self.id /
                max(
                    1,
                    NUM_AGENTS - 1
                )

            ],
            dtype=np.float32
        )

    # ========================================================
    # CONTEXT
    # ========================================================

    def local_context(self):

        values = [
            float(
                self.world.gate_open
            ),

            float(
                self.world.goal_reached
            ),

            self.vision.monster.x / 520,

            self.vision.monster.y / 285,

            self.vision.monster.vx / 6,

            self.vision.monster.vy / 6,

            self.risk,

            self.id / max(
                1,
                NUM_AGENTS - 1
            )
        ]

        # 他Agentの情報

        for other in range(
            NUM_AGENTS
        ):

            if other == self.id:
                continue

            data = self.blackboard.get(
                other
            )

            if data is None:

                values.extend(
                    [0, 0, 0, 0, 0]
                )

            else:

                values.extend(
                    [

                        (
                            data["x"] -
                            self.x
                        ) / 520,

                        (
                            data["y"] -
                            self.y
                        ) / 285,

                        data["vx"] / MAX_SPEED,

                        data["vy"] / 13,

                        data["risk"]

                    ]
                )

        return np.asarray(
            values,
            dtype=np.float32
        )

    # ========================================================
    # STEP
    # ========================================================

    def observe_and_select(
        self,
        agents
    ):

        self.assign_role(
            agents
        )

        image = self.vision.capture(
            agents,
            self.id,
            self.blackboard
        )

        body = self.body_state()
        context = self.local_context()

        action, scores = (
            self.model.select_action(
                image,
                body,
                context,
                self.role_biases()
            )
        )

        return (
            image,
            body,
            context,
            action,
            scores
        )

    def learn_transition(
        self,
        agents,
        image_before,
        body_before,
        context_before,
        action
    ):

        image_after = self.vision.capture(
            agents,
            self.id,
            self.blackboard
        )

        body_after = self.body_state()
        context_after = self.local_context()

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
        self.last_error = result["error"]

        self.steps += 1

        self.turtle.goto(
            self.x,
            self.y
        )

        return result


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

    # outer walls

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

    # central walls

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

    # tunnel

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

    # obstacles

    for obstacle in world.obstacles:

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

    # gate

    if not world.gate_open:

        d.goto(
            GATE_X,
            GATE_Y
        )

        d.dot(
            48,
            "#ffd447"
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

    # goal

    d.goto(
        GOAL_X,
        GOAL_Y
    )

    d.dot(
        55,
        "#42ff8a"
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

    # reset blocks

    for x, y in world.reset_blocks:

        d.goto(
            x,
            y
        )

        d.dot(
            8,
            "#a83cff"
        )

    # monster

    d.goto(
        monster.x,
        monster.y
    )

    d.dot(
        MONSTER_RADIUS * 2,
        "#ff3155"
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
        "REWARD-FREE COOPERATIVE WORLD MODEL",
        16,
        "#ffffff"
    )

    write(
        -500,
        313,
        f"EPISODE {episode + 1}/{MAX_EPISODES}"
        f"   STEP {step}/{STEPS_PER_EPISODE}",
        10,
        "#52ffba"
    )

    write(
        -500,
        285,
        "NO EXTERNAL REWARD",
        11,
        "#ffcc55"
    )

    write(
        -500,
        265,
        "LEARNING : PREDICTION ERROR + CURIOSITY",
        9,
        "#69cfff"
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
        222,
        f"Gate pushes : {world.gate_pushes}",
        9
    )

    write(
        -500,
        204,
        f"Reset events : {world.reset_events}",
        9,
        "#b76cff"
    )

    write(
        -500,
        186,
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

    # monster

    write(
        170,
        310,
        "PREDICTIVE MONSTER",
        11,
        "#ff4058"
    )

    write(
        170,
        290,
        f"Target : A{monster.target_id}",
        9,
        "#ff9aaa"
    )

    write(
        170,
        272,
        f"Attacks : {monster.attack_count}",
        9,
        "#ff9aaa"
    )

    write(
        170,
        254,
        f"Switches : {monster.target_switches}",
        9,
        "#ff9aaa"
    )

    write(
        170,
        236,
        f"vx={monster.vx:+.2f} "
        f"vy={monster.vy:+.2f}",
        9,
        "#ff9aaa"
    )

    # model

    write(
        500,
        290,
        f"VisualCells : {stats['visual']}",
        9
    )

    write(
        500,
        272,
        f"Places : {stats['places']}",
        9
    )

    write(
        500,
        254,
        f"Transforms : {stats['transforms']}",
        9
    )

    write(
        500,
        236,
        f"Experiences : {stats['experiences']}",
        9
    )

    write(
        500,
        218,
        f"Prediction error : "
        f"{stats['error']:.3f}",
        9,
        "#ffbb55"
    )

    write(
        500,
        200,
        f"Stability : "
        f"{stats['stability']:.3f}",
        9,
        "#69cfff"
    )

    write(
        500,
        182,
        f"Curiosity : "
        f"{stats['curiosity']:.3f}",
        9,
        "#d68cff"
    )

    # agents

    y = 145

    for agent in agents:

        write(
            -500,
            y,
            f"A{agent.id} "
            f"{agent.role:<8} "
            f"{ACTION_NAMES[agent.last_action]:<5} "
            f"x={agent.x:+.0f} "
            f"y={agent.y:+.0f} "
            f"vx={agent.vx:+.1f} "
            f"risk={agent.risk:.2f}",
            9,
            agent.color
        )

        y -= 19

    # place graph

    write(
        500,
        165,
        "SHARED PLACE MEMORY",
        9,
        "#ffffff"
    )

    graph.clear()

    coords = {}

    max_places = min(
        42,
        len(model.places)
    )

    for i in range(
        max_places
    ):

        place = model.places[i]

        x = (
            500 +
            (
                i % 7
            ) * 42
        )

        y = (
            135 -
            (
                i // 7
            ) * 31
        )

        coords[i] = (
            x,
            y
        )

        radius = min(
            13,
            5 +
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
    "Reward-Free Cooperative World Model"
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
# SHARED BLACKBOARD
# ============================================================

blackboard = SharedBlackboard()


# ============================================================
# VISION
# ============================================================

vision = VisualField(
    world,
    monster
)


# ============================================================
# SHARED WORLD MODEL
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
        vision,
        blackboard
    )

    for i in range(
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
# EPISODE RESET
# ============================================================

def reset_episode():

    global step

    step = 0

    world.reset()
    monster.reset()

    blackboard.data.clear()

    for agent in agents:

        agent.reset()


# ============================================================
# END EPISODE
# ============================================================

def end_episode():

    global episode
    global finished

    # --------------------------------------------------------
    # rewardではなく経験再活性化
    # --------------------------------------------------------

    model.replay(
        1000
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
    # 1. Blackboardを現在状態で更新
    # --------------------------------------------------------

    for agent in agents:

        agent.compute_risk()

        blackboard.update(
            agent
        )

    # --------------------------------------------------------
    # 2. Monster prediction
    # --------------------------------------------------------

    monster.update(
        agents
    )

    # --------------------------------------------------------
    # 3. 全Agentが同時にaction決定
    # --------------------------------------------------------

    observations = []

    actions = []

    for agent in agents:

        if agent.goal_reached:

            observations.append(
                None
            )

            actions.append(
                ACTION_WAIT
            )

            continue

        observation = (
            agent.observe_and_select(
                agents
            )
        )

        observations.append(
            observation
        )

        actions.append(
            observation[3]
        )

    # --------------------------------------------------------
    # 4. Physicsを同期更新
    # --------------------------------------------------------

    world.step_world(
        agents,
        actions
    )

    # --------------------------------------------------------
    # 5. 新しい状態をblackboardへ
    # --------------------------------------------------------

    for agent in agents:

        agent.compute_risk()

        blackboard.update(
            agent
        )

    # --------------------------------------------------------
    # 6. World Modelへ全遷移を学習
    # --------------------------------------------------------

    for agent, observation in zip(
        agents,
        observations
    ):

        if observation is None:
            continue

        image_before = observation[0]
        body_before = observation[1]
        context_before = observation[2]
        action = observation[3]

        agent.learn_transition(
            agents,
            image_before,
            body_before,
            context_before,
            action
        )

    # --------------------------------------------------------
    # 7. render
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
    # next
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
