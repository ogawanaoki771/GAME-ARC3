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
# RC-LIKE LIVING WORLD
# ============================================================

RC_FIELD_NODES = 18

RC_BASE_ENERGY = 0.55
RC_MAX_ENERGY = 1.80

RC_PULSE_SPEED = 0.035
RC_FLOW_SPEED = 0.018

LIVING_WALL_MIN_SCALE = 0.72
LIVING_WALL_MAX_SCALE = 1.28

WORLD_BREATH_AMOUNT_X = 13
WORLD_BREATH_AMOUNT_Y = 10

TUNNEL_BREATH_AMOUNT = 18

NUM_CHIMIMORYO = 8

CHIMIMORYO_MIN_RADIUS = 8
CHIMIMORYO_MAX_RADIUS = 19

CHIMIMORYO_MAX_SPEED = 3.8
CHIMIMORYO_ACCEL = 0.20

CHIMIMORYO_PUSH = 1.9
CHIMIMORYO_ENERGY_DRAIN = 0.12


# ============================================================
# RC ENERGY NODE
# ============================================================

class RCNode:

    def __init__(
        self,
        node_id,
        x,
        y,
        energy=None
    ):

        self.id = node_id

        self.x = float(x)
        self.y = float(y)

        self.base_x = float(x)
        self.base_y = float(y)

        self.vx = random.uniform(
            -0.35,
            0.35
        )

        self.vy = random.uniform(
            -0.35,
            0.35
        )

        self.energy = (
            random.uniform(
                0.35,
                1.25
            )
            if energy is None
            else energy
        )

        self.phase = random.uniform(
            0,
            math.tau
        )

        self.frequency = random.uniform(
            0.65,
            1.45
        )

        self.radius = random.uniform(
            65,
            150
        )

    def update(
        self,
        world_time
    ):

        pulse = math.sin(
            self.phase +
            world_time *
            RC_PULSE_SPEED *
            self.frequency
        )

        self.energy += (
            pulse * 0.009
        )

        self.energy += random.uniform(
            -0.012,
            0.012
        )

        self.energy = clamp(
            self.energy,
            0.08,
            RC_MAX_ENERGY
        )

        # エネルギー節自体も漂う

        target_x = (
            self.base_x +
            math.sin(
                self.phase +
                world_time * 0.011
            ) * 45
        )

        target_y = (
            self.base_y +
            math.cos(
                self.phase * 1.7 +
                world_time * 0.014
            ) * 38
        )

        self.vx += (
            target_x - self.x
        ) * 0.0009

        self.vy += (
            target_y - self.y
        ) * 0.0009

        self.vx += random.uniform(
            -0.025,
            0.025
        )

        self.vy += random.uniform(
            -0.025,
            0.025
        )

        self.vx *= 0.985
        self.vy *= 0.985

        self.x += self.vx
        self.y += self.vy

        self.x = clamp(
            self.x,
            WORLD_LEFT + 35,
            WORLD_RIGHT - 35
        )

        self.y = clamp(
            self.y,
            WORLD_BOTTOM + 35,
            WORLD_TOP - 35
        )


# ============================================================
# RC FIELD
# ============================================================

class RCField:

    def __init__(self):

        self.nodes = []

        self.global_energy = RC_BASE_ENERGY

        self.phase = random.uniform(
            0,
            math.tau
        )

        self.flow_x = 0.0
        self.flow_y = 0.0

        self.reset()

    def reset(self):

        self.nodes.clear()

        for i in range(
            RC_FIELD_NODES
        ):

            if i < RC_FIELD_NODES // 3:

                x = random.uniform(
                    LEFT_ROOM_X1 + 40,
                    LEFT_ROOM_X2 - 20
                )

            elif i < (
                RC_FIELD_NODES * 2 // 3
            ):

                x = random.uniform(
                    RIGHT_ROOM_X1 + 20,
                    RIGHT_ROOM_X2 - 40
                )

            else:

                x = random.uniform(
                    TUNNEL_X1 - 80,
                    TUNNEL_X2 + 80
                )

            y = random.uniform(
                WORLD_BOTTOM + 35,
                WORLD_TOP - 35
            )

            self.nodes.append(
                RCNode(
                    i,
                    x,
                    y
                )
            )

    def update(
        self,
        world_time
    ):

        for node in self.nodes:

            node.update(
                world_time
            )

        if self.nodes:

            self.global_energy = float(
                np.mean(
                    [
                        node.energy
                        for node in self.nodes
                    ]
                )
            )

        self.global_energy = clamp(
            self.global_energy,
            0.05,
            RC_MAX_ENERGY
        )

        self.flow_x = math.sin(
            self.phase +
            world_time *
            RC_FLOW_SPEED
        )

        self.flow_y = math.cos(
            self.phase * 1.3 +
            world_time *
            RC_FLOW_SPEED * 0.83
        )

    def energy_at(
        self,
        x,
        y
    ):

        total = 0.0
        weight_sum = 0.0

        for node in self.nodes:

            d = distance(
                x,
                y,
                node.x,
                node.y
            )

            weight = math.exp(
                -(
                    d * d
                ) /
                (
                    2 *
                    node.radius *
                    node.radius
                )
            )

            total += (
                node.energy *
                weight
            )

            weight_sum += weight

        if weight_sum <= 0.0001:

            return self.global_energy

        value = (
            total /
            weight_sum
        )

        return clamp(
            value,
            0.0,
            RC_MAX_ENERGY
        )

    def gradient_at(
        self,
        x,
        y
    ):

        sample = 16

        left = self.energy_at(
            x - sample,
            y
        )

        right = self.energy_at(
            x + sample,
            y
        )

        bottom = self.energy_at(
            x,
            y - sample
        )

        top = self.energy_at(
            x,
            y + sample
        )

        gx = (
            right - left
        ) / (
            sample * 2
        )

        gy = (
            top - bottom
        ) / (
            sample * 2
        )

        return gx, gy

    def disturb(
        self,
        x,
        y,
        amount
    ):

        if not self.nodes:
            return

        nearest = min(
            self.nodes,
            key=lambda node:
            distance(
                x,
                y,
                node.x,
                node.y
            )
        )

        nearest.energy = clamp(
            nearest.energy + amount,
            0.05,
            RC_MAX_ENERGY
        )


# ============================================================
# LIVING OBSTACLE
# ============================================================

class Obstacle:

    def __init__(
        self,
        x1,
        y1,
        x2,
        y2,
        movement=0.0,
        pulse=0.0,
        phase=None
    ):

        self.base_x1 = float(x1)
        self.base_y1 = float(y1)

        self.base_x2 = float(x2)
        self.base_y2 = float(y2)

        self.x1 = float(x1)
        self.y1 = float(y1)

        self.x2 = float(x2)
        self.y2 = float(y2)

        self.movement = movement
        self.pulse = pulse

        self.phase = (
            random.uniform(
                0,
                math.tau
            )
            if phase is None
            else phase
        )

        self.activity = random.uniform(
            0.65,
            1.35
        )

        self.energy = 0.5

        self.center_x = (
            x1 + x2
        ) * 0.5

        self.center_y = (
            y1 + y2
        ) * 0.5

    def update(
        self,
        world_time,
        rc_field
    ):

        base_center_x = (
            self.base_x1 +
            self.base_x2
        ) * 0.5

        base_center_y = (
            self.base_y1 +
            self.base_y2
        ) * 0.5

        base_width = abs(
            self.base_x2 -
            self.base_x1
        )

        base_height = abs(
            self.base_y2 -
            self.base_y1
        )

        self.energy = rc_field.energy_at(
            base_center_x,
            base_center_y
        )

        pulse_wave = math.sin(
            self.phase +
            world_time *
            0.028 *
            self.activity
        )

        second_wave = math.cos(
            self.phase * 1.8 +
            world_time *
            0.019
        )

        energy_factor = clamp(
            self.energy /
            RC_BASE_ENERGY,
            0.45,
            2.2
        )

        scale_x = (
            1.0 +
            pulse_wave *
            self.pulse *
            energy_factor
        )

        scale_y = (
            1.0 +
            second_wave *
            self.pulse *
            0.72 *
            energy_factor
        )

        scale_x = clamp(
            scale_x,
            LIVING_WALL_MIN_SCALE,
            LIVING_WALL_MAX_SCALE
        )

        scale_y = clamp(
            scale_y,
            LIVING_WALL_MIN_SCALE,
            LIVING_WALL_MAX_SCALE
        )

        drift_x = (
            math.sin(
                self.phase +
                world_time * 0.012
            ) *
            self.movement *
            energy_factor
        )

        drift_y = (
            math.cos(
                self.phase * 1.4 +
                world_time * 0.010
            ) *
            self.movement *
            0.72 *
            energy_factor
        )

        self.center_x = (
            base_center_x +
            drift_x
        )

        self.center_y = (
            base_center_y +
            drift_y
        )

        half_width = (
            base_width *
            scale_x *
            0.5
        )

        half_height = (
            base_height *
            scale_y *
            0.5
        )

        self.x1 = (
            self.center_x -
            half_width
        )

        self.x2 = (
            self.center_x +
            half_width
        )

        self.y1 = (
            self.center_y -
            half_height
        )

        self.y2 = (
            self.center_y +
            half_height
        )

        self.x1 = clamp(
            self.x1,
            WORLD_LEFT + 24,
            WORLD_RIGHT - 25
        )

        self.x2 = clamp(
            self.x2,
            WORLD_LEFT + 25,
            WORLD_RIGHT - 24
        )

        self.y1 = clamp(
            self.y1,
            WORLD_BOTTOM + 24,
            WORLD_TOP - 25
        )

        self.y2 = clamp(
            self.y2,
            WORLD_BOTTOM + 25,
            WORLD_TOP - 24
        )

        if self.x1 > self.x2:

            self.x1, self.x2 = (
                self.x2,
                self.x1
            )

        if self.y1 > self.y2:

            self.y1, self.y2 = (
                self.y2,
                self.y1
            )

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
# CHIMIMORYO
# ============================================================

class Chimimoryo:

    def __init__(
        self,
        entity_id,
        world
    ):

        self.id = entity_id
        self.world = world

        self.phase = random.uniform(
            0,
            math.tau
        )

        self.x = random.uniform(
            WORLD_LEFT + 70,
            WORLD_RIGHT - 70
        )

        self.y = random.uniform(
            WORLD_BOTTOM + 70,
            WORLD_TOP - 70
        )

        self.vx = random.uniform(
            -1.2,
            1.2
        )

        self.vy = random.uniform(
            -1.2,
            1.2
        )

        self.energy = random.uniform(
            0.3,
            1.1
        )

        self.radius = random.uniform(
            CHIMIMORYO_MIN_RADIUS,
            CHIMIMORYO_MAX_RADIUS
        )

        self.mode = random.choice(
            [
                "drifter",
                "hunter",
                "swarm",
                "flee"
            ]
        )

        self.target_id = None

        self.hits = 0

    def reset(self):

        self.x = random.uniform(
            WORLD_LEFT + 70,
            WORLD_RIGHT - 70
        )

        self.y = random.uniform(
            WORLD_BOTTOM + 70,
            WORLD_TOP - 70
        )

        self.vx = random.uniform(
            -1,
            1
        )

        self.vy = random.uniform(
            -1,
            1
        )

        self.energy = random.uniform(
            0.25,
            1.15
        )

        self.target_id = None

        self.hits = 0

    def nearest_agent(
        self,
        agents
    ):

        candidates = [
            agent
            for agent in agents
            if not agent.goal_reached
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda agent:
            distance(
                self.x,
                self.y,
                agent.x,
                agent.y
            )
        )

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

            limit = (
                self.radius + 24
            )

            if d < limit:

                if d < 0.01:

                    dx = random.uniform(
                        -1,
                        1
                    )

                    dy = random.uniform(
                        -1,
                        1
                    )

                else:

                    dx = (
                        self.x -
                        closest_x
                    ) / d

                    dy = (
                        self.y -
                        closest_y
                    ) / d

                force = (
                    1.0 -
                    d / limit
                ) * 0.65

                self.vx += dx * force
                self.vy += dy * force

    def update(
        self,
        agents,
        entities
    ):

        rc_field = self.world.rc_field

        local_energy = rc_field.energy_at(
            self.x,
            self.y
        )

        gx, gy = rc_field.gradient_at(
            self.x,
            self.y
        )

        self.energy += (
            local_energy -
            self.energy
        ) * 0.018

        self.energy = clamp(
            self.energy,
            0.08,
            RC_MAX_ENERGY
        )

        # RC濃度の高い方へ寄る

        self.vx += (
            gx * 15
        )

        self.vy += (
            gy * 15
        )

        self.vx += (
            rc_field.flow_x *
            0.025
        )

        self.vy += (
            rc_field.flow_y *
            0.025
        )

        target = self.nearest_agent(
            agents
        )

        if target is not None:

            self.target_id = target.id

            dx = target.x - self.x
            dy = target.y - self.y

            d = math.sqrt(
                dx * dx +
                dy * dy
            )

            if d > 0.01:

                if self.mode == "hunter":

                    if d < 240:

                        self.vx += (
                            dx / d *
                            CHIMIMORYO_ACCEL
                        )

                        self.vy += (
                            dy / d *
                            CHIMIMORYO_ACCEL
                        )

                elif self.mode == "flee":

                    if d < 170:

                        self.vx -= (
                            dx / d *
                            CHIMIMORYO_ACCEL
                        )

                        self.vy -= (
                            dy / d *
                            CHIMIMORYO_ACCEL
                        )

                elif self.mode == "drifter":

                    tangent_x = -dy / d
                    tangent_y = dx / d

                    self.vx += (
                        tangent_x * 0.08
                    )

                    self.vy += (
                        tangent_y * 0.08
                    )

        if self.mode == "swarm":

            for other in entities:

                if other.id == self.id:
                    continue

                d = distance(
                    self.x,
                    self.y,
                    other.x,
                    other.y
                )

                if 20 < d < 110:

                    self.vx += (
                        other.x -
                        self.x
                    ) / d * 0.015

                    self.vy += (
                        other.y -
                        self.y
                    ) / d * 0.015

                elif d <= 20 and d > 0.01:

                    self.vx -= (
                        other.x -
                        self.x
                    ) / d * 0.10

                    self.vy -= (
                        other.y -
                        self.y
                    ) / d * 0.10

        # 不規則な触手的運動

        self.vx += (
            math.sin(
                self.phase +
                self.world.time * 0.051
            ) * 0.055
        )

        self.vy += (
            math.cos(
                self.phase * 1.6 +
                self.world.time * 0.043
            ) * 0.055
        )

        self.avoid_obstacles()

        speed = math.sqrt(
            self.vx * self.vx +
            self.vy * self.vy
        )

        max_speed = (
            CHIMIMORYO_MAX_SPEED *
            clamp(
                0.55 +
                self.energy * 0.55,
                0.55,
                1.30
            )
        )

        if speed > max_speed:

            self.vx = (
                self.vx /
                speed *
                max_speed
            )

            self.vy = (
                self.vy /
                speed *
                max_speed
            )

        self.x += self.vx
        self.y += self.vy

        self.vx *= 0.972
        self.vy *= 0.972

        left, right, bottom, top = (
            self.world.dynamic_bounds()
        )

        if self.x < left + self.radius:

            self.x = left + self.radius
            self.vx = abs(
                self.vx
            ) * 0.8

        elif self.x > right - self.radius:

            self.x = right - self.radius
            self.vx = -abs(
                self.vx
            ) * 0.8

        if self.y < bottom + self.radius:

            self.y = bottom + self.radius
            self.vy = abs(
                self.vy
            ) * 0.8

        elif self.y > top - self.radius:

            self.y = top - self.radius
            self.vy = -abs(
                self.vy
            ) * 0.8

        self.radius = clamp(
            CHIMIMORYO_MIN_RADIUS +
            self.energy * 7 +
            math.sin(
                self.phase +
                self.world.time * 0.08
            ) * 3,
            CHIMIMORYO_MIN_RADIUS,
            CHIMIMORYO_MAX_RADIUS
        )

        for agent in agents:

            self.collide_agent(
                agent
            )

    def collide_agent(
        self,
        agent
    ):

        dx = agent.x - self.x
        dy = agent.y - self.y

        d = math.sqrt(
            dx * dx +
            dy * dy
        )

        collision_distance = (
            self.radius + 12
        )

        if (
            0.01 <
            d <
            collision_distance
        ):

            nx = dx / d
            ny = dy / d

            push = (
                CHIMIMORYO_PUSH *
                clamp(
                    self.energy,
                    0.4,
                    1.6
                )
            )

            agent.vx += nx * push
            agent.vy += ny * push

            self.vx -= nx * 0.7
            self.vy -= ny * 0.7

            penetration = (
                collision_distance - d
            )

            agent.x += (
                nx * penetration
            )

            agent.y += (
                ny * penetration
            )

            self.energy = clamp(
                self.energy +
                CHIMIMORYO_ENERGY_DRAIN,
                0,
                RC_MAX_ENERGY
            )

            self.world.rc_field.disturb(
                self.x,
                self.y,
                0.08
            )

            self.hits += 1


# ============================================================
# WORLD CLASS PATCH
# ============================================================
#
# 以下を既存の World クラスへ統合する
# ============================================================

# ------------------------------------------------------------
# World.__init__()
# ------------------------------------------------------------
#
# create_geometry() より前に追加:
#
#     self.rc_field = RCField()
#     self.chimimoryo = []
#
# create_geometry() の直後:
#
#     self.create_chimimoryo()


# ------------------------------------------------------------
# create_chimimoryo
# ------------------------------------------------------------

def world_create_chimimoryo(self):

    self.chimimoryo = [

        Chimimoryo(
            i,
            self
        )

        for i in range(
            NUM_CHIMIMORYO
        )
    ]


# ------------------------------------------------------------
# dynamic_bounds
# ------------------------------------------------------------

def world_dynamic_bounds(self):

    energy = self.rc_field.global_energy

    pulse_x = math.sin(
        self.time * 0.021
    )

    pulse_y = math.cos(
        self.time * 0.017
    )

    amount_x = (
        WORLD_BREATH_AMOUNT_X *
        clamp(
            energy,
            0.4,
            1.5
        )
    )

    amount_y = (
        WORLD_BREATH_AMOUNT_Y *
        clamp(
            energy,
            0.4,
            1.5
        )
    )

    left = (
        WORLD_LEFT +
        pulse_x * amount_x
    )

    right = (
        WORLD_RIGHT -
        math.sin(
            self.time * 0.019 + 1.8
        ) * amount_x
    )

    bottom = (
        WORLD_BOTTOM +
        pulse_y * amount_y
    )

    top = (
        WORLD_TOP -
        math.cos(
            self.time * 0.016 + 2.1
        ) * amount_y
    )

    if right - left < 900:

        center = (
            left + right
        ) * 0.5

        left = center - 450
        right = center + 450

    if top - bottom < 470:

        center = (
            top + bottom
        ) * 0.5

        bottom = center - 235
        top = center + 235

    return (
        left,
        right,
        bottom,
        top
    )


# ------------------------------------------------------------
# dynamic_tunnel_bounds
# ------------------------------------------------------------

def world_dynamic_tunnel_bounds(self):

    energy = self.rc_field.energy_at(
        0,
        0
    )

    pulse = math.sin(
        self.time * 0.032
    )

    deformation = (
        pulse *
        TUNNEL_BREATH_AMOUNT *
        clamp(
            energy,
            0.5,
            1.6
        )
    )

    lower = (
        TUNNEL_Y1 -
        deformation
    )

    upper = (
        TUNNEL_Y2 +
        deformation
    )

    if upper - lower < 46:

        center = (
            upper + lower
        ) * 0.5

        lower = center - 23
        upper = center + 23

    return lower, upper


# ------------------------------------------------------------
# update_living_world
# ------------------------------------------------------------

def world_update_living_world(
    self,
    agents
):

    self.rc_field.update(
        self.time
    )

    for obstacle in self.obstacles:

        obstacle.update(
            self.time,
            self.rc_field
        )

    for entity in self.chimimoryo:

        entity.update(
            agents,
            self.chimimoryo
        )


# ------------------------------------------------------------
# create_obstacles
# ------------------------------------------------------------

def world_create_obstacles(self):

    self.obstacles.clear()

    definitions = [

        (
            -400, -205,
            -275, -155,
            9, 0.11
        ),

        (
            -235, 70,
            -110, 115,
            13, 0.15
        ),

        (
            -470, -30,
            -370, 15,
            7, 0.18
        ),

        (
            -105, -210,
            -72, -75,
            5, 0.13
        ),

        (
            72, 75,
            105, 210,
            6, 0.16
        ),

        (
            170, -190,
            285, -135,
            15, 0.14
        ),

        (
            310, 35,
            440, 85,
            12, 0.17
        ),

        (
            150, 125,
            245, 165,
            18, 0.13
        ),

        (
            335, 135,
            385, 185,
            9, 0.21
        )
    ]

    for (
        x1,
        y1,
        x2,
        y2,
        movement,
        pulse
    ) in definitions:

        self.obstacles.append(
            Obstacle(
                x1,
                y1,
                x2,
                y2,
                movement=movement,
                pulse=pulse
            )
        )


# ------------------------------------------------------------
# inside_tunnel
# ------------------------------------------------------------

def world_inside_tunnel(
    self,
    x,
    y
):

    lower, upper = (
        self.dynamic_tunnel_bounds()
    )

    return (
        TUNNEL_X1 <= x <= TUNNEL_X2 and
        lower <= y <= upper
    )


# ------------------------------------------------------------
# enforce_boundary
# ------------------------------------------------------------

def world_enforce_boundary(
    self,
    agent
):

    left, right, bottom, top = (
        self.dynamic_bounds()
    )

    margin = 13

    if agent.x < left + margin:

        agent.x = left + 20

        agent.vx = (
            abs(
                agent.vx
            ) * 0.3
        )

        self.rc_field.disturb(
            agent.x,
            agent.y,
            0.035
        )

    if agent.x > right - margin:

        agent.x = right - 20

        agent.vx = (
            -abs(
                agent.vx
            ) * 0.3
        )

        self.rc_field.disturb(
            agent.x,
            agent.y,
            0.035
        )

    if agent.y < bottom + margin:

        agent.y = bottom + 20

        agent.vy = (
            abs(
                agent.vy
            ) * 0.3
        )

        self.rc_field.disturb(
            agent.x,
            agent.y,
            0.035
        )

    if agent.y > top - margin:

        agent.y = top - 20

        agent.vy = (
            -abs(
                agent.vy
            ) * 0.3
        )

        self.rc_field.disturb(
            agent.x,
            agent.y,
            0.035
        )


# ------------------------------------------------------------
# reset patch
# ------------------------------------------------------------

def world_reset_living_state(self):

    self.rc_field.reset()

    for entity in self.chimimoryo:

        entity.reset()


# ------------------------------------------------------------
# step_world
# ------------------------------------------------------------

def world_step_world(
    self,
    agents,
    actions
):

    self.time += 1

    # RCフィールド、壁、障害物、
    # 魑魅魍魎を同一tickで更新

    self.update_living_world(
        agents
    )

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
# WORLD CLASS METHOD ASSIGNMENTS
# ============================================================
#
# 既存 World クラス内へ直接メソッドを入れられない場合の
# パッチ方式。
# 既存の World 定義後に置く。
# ============================================================

World.create_chimimoryo = (
    world_create_chimimoryo
)

World.dynamic_bounds = (
    world_dynamic_bounds
)

World.dynamic_tunnel_bounds = (
    world_dynamic_tunnel_bounds
)

World.update_living_world = (
    world_update_living_world
)

World.create_obstacles = (
    world_create_obstacles
)

World.inside_tunnel = (
    world_inside_tunnel
)

World.enforce_boundary = (
    world_enforce_boundary
)

World.reset_living_state = (
    world_reset_living_state
)

World.step_world = (
    world_step_world
)


# ============================================================
# VISUAL FIELD PATCH
# ============================================================
#
# VisualField.capture() のMonster描画部分の直後へ追加
# ============================================================

def visualfield_draw_chimimoryo(
    self,
    img
):

    # --------------------------------------------------------
    # chimimoryo / living entities
    # --------------------------------------------------------

    for entity in self.world.chimimoryo:

        value = clamp(
            0.35 +
            entity.energy * 0.45,
            0.35,
            1.0
        )

        self.dot(
            img[:, :, 5],
            entity.x,
            entity.y,
            value
        )


# ============================================================
# DRAW WORLD - LIVING WORLD
# ============================================================

def draw_living_world(
    d,
    world,
    line,
    write=None
):

    # --------------------------------------------------------
    # living outer walls
    # --------------------------------------------------------

    (
        living_left,
        living_right,
        living_bottom,
        living_top
    ) = world.dynamic_bounds()

    wall_energy = world.rc_field.global_energy

    wall_color = (
        "#7d315f"
        if wall_energy > 0.85
        else "#49364f"
    )

    line(
        d,
        living_left,
        living_bottom,
        living_left,
        living_top,
        wall_color,
        7
    )

    line(
        d,
        living_right,
        living_bottom,
        living_right,
        living_top,
        wall_color,
        7
    )

    line(
        d,
        living_left,
        living_top,
        living_right,
        living_top,
        wall_color,
        7
    )

    line(
        d,
        living_left,
        living_bottom,
        living_right,
        living_bottom,
        wall_color,
        7
    )

    # --------------------------------------------------------
    # tunnel
    # --------------------------------------------------------

    tunnel_lower, tunnel_upper = (
        world.dynamic_tunnel_bounds()
    )

    tunnel_energy = (
        world.rc_field.energy_at(
            0,
            0
        )
    )

    tunnel_color = (
        "#ff3f91"
        if tunnel_energy > 0.90
        else "#8d3975"
    )

    line(
        d,
        TUNNEL_X1,
        tunnel_lower,
        TUNNEL_X2,
        tunnel_lower,
        tunnel_color,
        6
    )

    line(
        d,
        TUNNEL_X1,
        tunnel_upper,
        TUNNEL_X2,
        tunnel_upper,
        tunnel_color,
        6
    )

    # --------------------------------------------------------
    # central walls
    # --------------------------------------------------------

    line(
        d,
        -80,
        tunnel_upper,
        -80,
        living_top,
        "#713754",
        7
    )

    line(
        d,
        -80,
        living_bottom,
        -80,
        tunnel_lower,
        "#713754",
        7
    )

    line(
        d,
        80,
        tunnel_upper,
        80,
        living_top,
        "#713754",
        7
    )

    line(
        d,
        80,
        living_bottom,
        80,
        tunnel_lower,
        "#713754",
        7
    )

    # --------------------------------------------------------
    # RC energy nodes
    # --------------------------------------------------------

    for node in world.rc_field.nodes:

        node_size = int(
            clamp(
                4 +
                node.energy * 8,
                4,
                18
            )
        )

        d.goto(
            node.x,
            node.y
        )

        d.dot(
            node_size,
            "#8b245f"
        )

        d.dot(
            max(
                2,
                node_size // 3
            ),
            "#ff4da6"
        )

    # --------------------------------------------------------
    # chimimoryo
    # --------------------------------------------------------

    entity_colors = {
        "drifter": "#b94cff",
        "hunter": "#ff285d",
        "swarm": "#5bffca",
        "flee": "#677dff"
    }

    for entity in world.chimimoryo:

        color = entity_colors.get(
            entity.mode,
            "#d15cff"
        )

        pulse = math.sin(
            entity.phase +
            world.time * 0.09
        )

        radius = (
            entity.radius +
            pulse * 3
        )

        # 触手状の線

        tentacles = 5

        for i in range(
            tentacles
        ):

            angle = (
                entity.phase +
                world.time * 0.025 +
                i *
                math.tau /
                tentacles
            )

            length = (
                radius *
                (
                    1.3 +
                    0.45 *
                    math.sin(
                        world.time * 0.08 +
                        i * 1.7
                    )
                )
            )

            x2 = (
                entity.x +
                math.cos(
                    angle
                ) * length
            )

            y2 = (
                entity.y +
                math.sin(
                    angle
                ) * length
            )

            line(
                d,
                entity.x,
                entity.y,
                x2,
                y2,
                color,
                2
            )

        d.goto(
            entity.x,
            entity.y
        )

        d.dot(
            radius * 2,
            color
        )

        d.dot(
            max(
                4,
                radius * 0.55
            ),
            "#120617"
        )


# ============================================================
# UI / RC INFORMATION
# ============================================================

def draw_rc_ui(
    world,
    write
):

    write(
        170,
        218,
        (
            "RC Energy : "
            f"{world.rc_field.global_energy:.3f}"
        ),
        9,
        "#ff5bad"
    )

    write(
        170,
        200,
        (
            "Chimimoryo : "
            f"{len(world.chimimoryo)}"
        ),
        9,
        "#d26cff"
    )

    write(
        170,
        182,
        (
            "Entity hits : "
            f"{sum(e.hits for e in world.chimimoryo)}"
        ),
        9,
        "#ff789a"
    )

# ============================================================
# FINAL RUNTIME INTEGRATION
# ============================================================
#
# ここから先で、後半に追加された Living World / RC Field /
# Chimimoryo を実際の実行系へ統合する。
#
# 重要:
#   - World は Living Obstacle が有効になった後に生成する
#   - turtle.done() はファイルの最後で一度だけ呼ぶ
#   - VisualField と描画も Living World を反映する
# ============================================================


# ------------------------------------------------------------
# Patch World.__init__
# ------------------------------------------------------------

_original_world_init = World.__init__


def _world_init_integrated(self):
    self.rc_field = RCField()
    self.chimimoryo = []

    _original_world_init(self)

    self.create_chimimoryo()


World.__init__ = _world_init_integrated


# ------------------------------------------------------------
# Patch World.reset
# ------------------------------------------------------------

_original_world_reset = World.reset


def _world_reset_integrated(self):
    _original_world_reset(self)

    self.reset_living_state()

    # create_obstacles() が現在位置を再初期化するわけではないため、
    # 次の tick で必ず現在時刻から再計算される。
    for obstacle in self.obstacles:
        obstacle.update(
            self.time,
            self.rc_field
        )


World.reset = _world_reset_integrated


# ------------------------------------------------------------
# Patch VisualField.capture
# ------------------------------------------------------------

_original_visualfield_capture = VisualField.capture


def _visualfield_capture_integrated(
    self,
    agents,
    viewer_id,
    blackboard
):
    img = _original_visualfield_capture(
        self,
        agents,
        viewer_id,
        blackboard
    )

    # chimimoryo は channel 5 に追加する。
    # predicted target と同じ channel を共有することで、
    # 「予測された存在」と「実際の動的存在」の差を
    # world model が変化として観測できるようにする。
    img = np.asarray(
        img,
        dtype=np.float32
    ).reshape(
        OBS_H,
        OBS_W,
        CHANNELS
    )

    self.draw_chimimoryo(
        img
    )

    return img.reshape(-1)


VisualField.draw_chimimoryo = (
    visualfield_draw_chimimoryo
)

VisualField.capture = (
    _visualfield_capture_integrated
)


# ------------------------------------------------------------
# Patch draw_world
# ------------------------------------------------------------

_original_draw_world = draw_world


def _draw_world_integrated(
    world,
    monster
):
    _original_draw_world(
        world,
        monster
    )

    # Living World の動的境界、RC node、chimimoryo、
    # 動的 obstacle を既存描画の上に重ねる。
    draw_living_world(
        world.drawer,
        world,
        line,
        write
    )

    draw_rc_ui(
        world,
        write
    )


draw_world = _draw_world_integrated


# ------------------------------------------------------------
# Create actual runtime objects
# ------------------------------------------------------------

world = World()

monster = Monster(
    world
)

blackboard = SharedBlackboard()

vision = VisualField(
    world,
    monster
)

model = WorldModel()

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
    for i in range(NUM_AGENTS)
]


# ------------------------------------------------------------
# Start the simulation
# ------------------------------------------------------------

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


# ============================================================
# TURTLE MAIN LOOP
# ============================================================

turtle.done()


# ============================================================
# DYNAMIC WORLD EVOLUTION EXTENSION
# MovingPlatform / WindField / DynamicTerrain /
# WaterZone / WorldEventSystem / AdaptiveMonster
# ============================================================

class MovingPlatform:
    def __init__(self, x, y, width=80, amplitude=40, speed=0.03):
        self.base_x = x
        self.x = x
        self.y = y
        self.width = width
        self.amplitude = amplitude
        self.speed = speed

    def update(self, t):
        self.x = self.base_x + math.sin(t * self.speed) * self.amplitude

    def contains(self, x, y):
        return abs(x-self.x) < self.width/2 and abs(y-self.y) < 12


class WindField:
    def __init__(self):
        self.phase = 0

    def force(self, x, y, t):
        self.phase = t * 0.02
        return (
            math.sin(x*0.01 + self.phase) * 0.08,
            math.cos(y*0.01 + self.phase) * 0.05
        )


class DynamicTerrain:
    def __init__(self):
        self.offset = 0
        self.state = 0

    def update(self, t):
        self.offset = math.sin(t*0.01)
        if t % 500 == 0:
            self.state = 1 - self.state

    def danger(self, x, y):
        if self.state and -100 < x < 100 and y < -150:
            return True
        return False


class WaterZone:
    def __init__(self, x1, y1, x2, y2):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2

    def contains(self,x,y):
        return self.x1<x<self.x2 and self.y1<y<self.y2

    def physics(self,agent):
        agent.vx *= 0.93
        agent.vy *= 0.96


class WorldEventSystem:
    def __init__(self):
        self.event="normal"
        self.timer=0

    def update(self,t):
        if self.timer>0:
            self.timer-=1
            return

        if t>0 and t%700==0:
            self.event=random.choice(
                [
                    "wind",
                    "slow",
                    "gravity"
                ]
            )
            self.timer=120

    def apply(self,agent):
        if self.event=="slow":
            agent.vx*=0.97

        if self.event=="gravity":
            agent.vy*=0.96


class AdaptiveMonster(Monster):
    def __init__(self,world):
        super().__init__(world)
        self.strategy="predict"

    def update(self,agents):
        if self.attack_count>20:
            self.strategy="ambush"
        super().update(agents)


def attach_dynamic_world(world):
    world.moving_platforms=[
        MovingPlatform(-200,120),
        MovingPlatform(200,-80)
    ]

    world.wind_field=WindField()

    world.dynamic_terrain=DynamicTerrain()

    world.water_zones=[
        WaterZone(-50,-200,80,-100)
    ]

    world.events=WorldEventSystem()

    return world




# ============================================================
# ACTIVE CELL BALANCE SYSTEM
# Low activity agents / cells reactivation
# ============================================================

class ActivityHomeostasis:

    def __init__(self):
        self.activity = 1.0
        self.total_actions = 0
        self.idle_steps = 0

    def update(self, active=True):
        if active:
            self.activity = min(
                2.0,
                self.activity + 0.05
            )
            self.total_actions += 1
            self.idle_steps = 0
        else:
            self.activity *= 0.992
            self.idle_steps += 1

    def stimulation(self):
        if self.activity < 0.35:
            return 1.8

        if self.activity < 0.6:
            return 0.8

        return 0.0


class AgentRoleSystem:

    ROLES = [
        "explorer",
        "observer",
        "risk_monitor"
    ]

    def __init__(self, agent_id):
        self.role = self.ROLES[
            agent_id % len(self.ROLES)
        ]

    def bias(self):
        if self.role == "explorer":
            return {
                "novelty":1.4,
                "prediction":0.8
            }

        if self.role == "observer":
            return {
                "novelty":0.6,
                "prediction":1.5
            }

        return {
            "novelty":0.9,
            "prediction":1.1
        }


class SharedActivityController:

    def __init__(self, agents):
        self.activity = {
            a.id: ActivityHomeostasis()
            for a in agents
        }

        self.roles = {
            a.id: AgentRoleSystem(a.id)
            for a in agents
        }

    def update(self, agents):

        for agent in agents:

            active = (
                abs(agent.vx) > 0.2
                or abs(agent.vy) > 0.2
            )

            self.activity[
                agent.id
            ].update(active)


    def curiosity_boost(self, agent_id):

        return self.activity[
            agent_id
        ].stimulation()


    def role_bias(self, agent_id):

        return self.roles[
            agent_id
        ].bias()


# ============================================================
# END ACTIVE CELL BALANCE SYSTEM
# ============================================================

