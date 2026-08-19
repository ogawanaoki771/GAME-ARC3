import turtle
import random
import math
import numpy as np
from collections import deque


# ============================================================
# STRANGE PLACE / SELF-ORGANIZING WORLD MODEL v4
#
# v3:
#   Visual -> Place -> Transformation -> MetaTransformation
#
# v4:
#   Visual
#      ↓
#   Place
#      ↓
#   Transformation
#      ↓
#   Prediction
#      ↓
#   Counterfactual Action Evaluation
#      ↓
#   Real Outcome
#      ↓
#   Prediction Error / Surprise
#      ↓
#   Event Memory
#      ↓
#   MetaTransformation
#
# ============================================================


# ============================================================
# Screen / World
# ============================================================

SCREEN_W = 1240
SCREEN_H = 820

WORLD_LEFT = -560
WORLD_RIGHT = 560
WORLD_BOTTOM = -300
WORLD_TOP = 290


# ============================================================
# Simulation
# ============================================================

NUM_AGENTS = 3

STEPS_PER_EPISODE = 420
MAX_EPISODES = 14


# ============================================================
# Observation
# ============================================================

OBS_W = 72
OBS_H = 44
CHANNELS = 4

PATCH = 6


# ============================================================
# Similarity
# ============================================================

VISUAL_SIM_THRESHOLD = 0.74
PLACE_SIM_THRESHOLD = 0.80
TRANSFORM_SIM_THRESHOLD = 0.70
META_SIM_THRESHOLD = 0.72

EVENT_SIM_THRESHOLD = 0.78


# ============================================================
# Memory
# ============================================================

MEMORY_WINDOW = 140
TRANSFORM_HISTORY = 600


# ============================================================
# Physics
# ============================================================

BASE_GRAVITY = 0.72

GROUND_ACCEL = 1.05
AIR_ACCEL = 0.65

GROUND_FRICTION = 0.90
AIR_FRICTION = 0.985

JUMP_POWER = 11.2
MAX_SPEED = 11.0


# ============================================================
# Gravity
# ============================================================

GRAVITY_SWITCH_PERIOD = 170.0
GRAVITY_WARNING_TIME = 28

UPWARD_GRAVITY_BOUNCE = 0.88
IMPACT_HORIZONTAL_BOUNCE = 0.82


# ============================================================
# Strange world
# ============================================================

PHASE_PERIOD = 270.0
TIME_WARP_PERIOD = 220.0
TOPOLOGY_PERIOD = 340.0
PAST_LEAK_PERIOD = 260.0


# ============================================================
# Prediction
# ============================================================

PREDICTION_ERROR_SCALE = 5.0

PREDICTION_WEIGHT = 1.45
UNCERTAINTY_WEIGHT = 1.10
NOVELTY_WEIGHT = 0.75
BRANCH_WEIGHT = 0.25

COUNTERFACTUAL_WEIGHT = 0.70

RISK_WEIGHT = 1.20

EXPLORATION_RATE = 0.13


# ============================================================
# Actions
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
    ACTION_WAIT,
]


ACTION_NAMES = {
    ACTION_NONE: "NONE",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_JUMP: "JUMP",
    ACTION_BRAKE: "BRAKE",
    ACTION_WAIT: "WAIT",
}


ACTION_ACTIVITY_BIAS = {
    ACTION_NONE: -0.35,
    ACTION_LEFT: 0.30,
    ACTION_RIGHT: 0.30,
    ACTION_JUMP: 0.35,
    ACTION_BRAKE: -0.02,
    ACTION_WAIT: -0.20,
}


# ============================================================
# Utility
# ============================================================


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-9 or nb < 1e-9:
        return 0.0

    return float(
        np.dot(a, b) / (na * nb)
    )


def l1_distance(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    return float(
        np.mean(
            np.abs(a - b)
        )
    )


def soft_distance(a, b, scale=4.0):

    return math.exp(
        -scale
        * l1_distance(a, b)
    )


# ============================================================
# Event
# ============================================================


class WorldEvent:

    def __init__(
        self,
        name,
        time,
        x=0.0,
        y=0.0,
        intensity=1.0,
        source=-1,
    ):

        self.name = name

        self.time = time

        self.x = float(x)
        self.y = float(y)

        self.intensity = float(
            intensity
        )

        self.source = source

    def vector(self):

        event_types = [
            "GRAVITY_FLIP",
            "IMPACT",
            "WORMHOLE",
            "HAZARD",
            "TELEPORT",
            "BOUNDARY",
            "GROUND",
        ]

        v = np.zeros(
            len(event_types),
            dtype=np.float32,
        )

        if self.name in event_types:

            v[
                event_types.index(
                    self.name
                )
            ] = self.intensity

        return v


# ============================================================
# Event Memory
# ============================================================


class EventMemoryCell:

    def __init__(
        self,
        event_id,
        event_name,
        place_id,
        action,
        transform_id,
        context,
    ):

        self.id = event_id

        self.event_name = event_name

        self.place_id = place_id

        self.action = action

        self.transform_id = (
            transform_id
        )

        self.context = np.asarray(
            context,
            dtype=np.float32,
        )

        self.visits = 1

        self.energy = 1.0

        self.confidence = 0.1

        self.last_time = 0

    def similarity(
        self,
        event_name,
        place_id,
        action,
        context,
    ):

        if (
            self.event_name
            != event_name
        ):
            return 0.0

        place_score = (
            1.0
            if self.place_id
            == place_id
            else 0.25
        )

        action_score = (
            1.0
            if self.action
            == action
            else 0.35
        )

        context_score = (
            soft_distance(
                self.context,
                context,
                2.0,
            )
        )

        return (
            0.45 * place_score
            + 0.25 * action_score
            + 0.30 * context_score
        )

    def update(
        self,
        context,
        transform_id,
        time,
    ):

        context = np.asarray(
            context,
            dtype=np.float32,
        )

        self.context = (
            0.92 * self.context
            + 0.08 * context
        )

        self.transform_id = (
            transform_id
        )

        self.visits += 1

        self.energy = min(
            2.0,
            self.energy + 0.03,
        )

        self.confidence = clamp(
            self.confidence
            * 0.94
            + 0.06,
            0.0,
            1.0,
        )

        self.last_time = time

    def decay(self):

        self.energy *= 0.997


# ============================================================
# Impact Mark
# ============================================================


class ImpactMark:

    def __init__(
        self,
        x,
        y,
        gravity_direction,
    ):

        self.x = x
        self.y = y

        self.gravity_direction = (
            gravity_direction
        )

        self.life = 1.0
        self.radius = 42.0

        self.rotation = random.uniform(
            0,
            math.pi * 2,
        )

    def update(self):

        self.life -= 0.035

        self.radius += 1.8

    def alive(self):

        return self.life > 0.0

    def draw(self, drawer):

        if not self.alive():
            return

        alpha = self.life

        if (
            self.gravity_direction
            > 0
        ):

            color = "#ff4055"

        else:

            color = "#55ddff"

        r = self.radius

        drawer.color(color)

        drawer.pensize(
            max(
                2,
                int(
                    5 * alpha
                ),
            )
        )

        # ----------------------------------------------------
        # Circle
        # ----------------------------------------------------

        drawer.penup()

        drawer.goto(
            self.x + r,
            self.y,
        )

        drawer.setheading(90)

        drawer.pendown()

        for _ in range(36):

            drawer.forward(
                2
                * math.pi
                * r
                / 36
            )

            drawer.left(10)

        # ----------------------------------------------------
        # X
        # ----------------------------------------------------

        size = r * 0.65

        drawer.penup()

        drawer.goto(
            self.x - size,
            self.y - size,
        )

        drawer.pendown()

        drawer.goto(
            self.x + size,
            self.y + size,
        )

        drawer.penup()

        drawer.goto(
            self.x - size,
            self.y + size,
        )

        drawer.pendown()

        drawer.goto(
            self.x + size,
            self.y - size,
        )

        # ----------------------------------------------------
        # Cross
        # ----------------------------------------------------

        cross = r * 0.35

        drawer.penup()

        drawer.goto(
            self.x - cross,
            self.y,
        )

        drawer.pendown()

        drawer.goto(
            self.x + cross,
            self.y,
        )

        drawer.penup()

        drawer.goto(
            self.x,
            self.y - cross,
        )

        drawer.pendown()

        drawer.goto(
            self.x,
            self.y + cross,
        )

        # ----------------------------------------------------
        # Gravity arrow
        # ----------------------------------------------------

        arrow_len = r * 1.15

        drawer.penup()

        if (
            self.gravity_direction
            < 0
        ):

            drawer.goto(
                self.x,
                self.y - arrow_len,
            )

            drawer.pendown()

            drawer.goto(
                self.x,
                self.y + arrow_len,
            )

            drawer.goto(
                self.x - 12,
                self.y
                + arrow_len
                - 18,
            )

            drawer.penup()

            drawer.goto(
                self.x,
                self.y + arrow_len,
            )

            drawer.pendown()

            drawer.goto(
                self.x + 12,
                self.y
                + arrow_len
                - 18,
            )

        else:

            drawer.goto(
                self.x,
                self.y + arrow_len,
            )

            drawer.pendown()

            drawer.goto(
                self.x,
                self.y - arrow_len,
            )

            drawer.goto(
                self.x - 12,
                self.y
                - arrow_len
                + 18,
            )

            drawer.penup()

            drawer.goto(
                self.x,
                self.y - arrow_len,
            )

            drawer.pendown()

            drawer.goto(
                self.x + 12,
                self.y
                - arrow_len
                + 18,
            )

        drawer.penup()


# ============================================================
# Gravity Warning
# ============================================================


class GravityWarning:

    def __init__(
        self,
        x,
        y,
    ):

        self.x = x
        self.y = y
        self.phase = 0.0

    def update(self):

        self.phase += 0.22

    def draw(self, drawer):

        pulse = (
            1.0
            + 0.18
            * math.sin(
                self.phase
            )
        )

        r = 25 * pulse

        drawer.color(
            "#ffd84d"
        )

        drawer.pensize(4)

        drawer.penup()

        drawer.goto(
            self.x + r,
            self.y,
        )

        drawer.setheading(90)

        drawer.pendown()

        for _ in range(36):

            drawer.forward(
                2
                * math.pi
                * r
                / 36
            )

            drawer.left(10)

        s = r * 0.7

        drawer.penup()

        drawer.goto(
            self.x,
            self.y + s,
        )

        drawer.pendown()

        drawer.goto(
            self.x - s,
            self.y - s,
        )

        drawer.goto(
            self.x + s,
            self.y - s,
        )

        drawer.goto(
            self.x,
            self.y + s,
        )

        drawer.penup()

        drawer.goto(
            self.x,
            self.y - s * 0.5,
        )

        drawer.pendown()

        drawer.goto(
            self.x,
            self.y + s * 0.2,
        )

        drawer.penup()

        drawer.goto(
            self.x,
            self.y - s * 0.75,
        )

        drawer.dot(
            5,
            "#ffd84d",
        )


# ============================================================
# Strange World Regime
# ============================================================


class WorldRegime:

    def __init__(self):

        self.time = 0

        self.global_phase = 0.0
        self.topology_phase = 0.0
        self.time_phase = 0.0
        self.past_phase = 0.0

        self.gravity_sign = 1
        self.previous_gravity_sign = 1

        self.gravity_timer = 0

    def update(self):

        self.time += 1

        self.global_phase = (
            0.5
            + 0.5
            * math.sin(
                self.time
                * 2.0
                * math.pi
                / PHASE_PERIOD
            )
        )

        self.topology_phase = (
            0.5
            + 0.5
            * math.sin(
                self.time
                * 2.0
                * math.pi
                / TOPOLOGY_PERIOD
                + 1.4
            )
        )

        self.time_phase = (
            0.5
            + 0.5
            * math.sin(
                self.time
                * 2.0
                * math.pi
                / TIME_WARP_PERIOD
                - 0.8
            )
        )

        self.past_phase = (
            0.5
            + 0.5
            * math.sin(
                self.time
                * 2.0
                * math.pi
                / PAST_LEAK_PERIOD
                + 2.2
            )
        )

        self.previous_gravity_sign = (
            self.gravity_sign
        )

        self.gravity_timer += 1

        if (
            self.gravity_timer
            >= GRAVITY_SWITCH_PERIOD
        ):

            self.gravity_timer = 0

            self.gravity_sign *= -1

    def time_scale(self):

        return (
            0.30
            + 1.65
            * self.time_phase
        )

    def local_gravity(
        self,
        x,
        y,
    ):

        spatial = (
            math.sin(
                x * 0.012
                + self.time * 0.018
            )
            * 0.5
            + 0.5
        )

        local_sign = (
            1.0
            if spatial
            > (
                0.38
                + 0.22
                * self.global_phase
            )
            else -1.0
        )

        magnitude = (
            BASE_GRAVITY
            * (
                0.45
                + 1.15
                * abs(
                    math.sin(
                        y * 0.01
                        + self.time * 0.013
                    )
                )
            )
        )

        return (
            self.gravity_sign
            * local_sign
            * magnitude
        )

    def local_phase(
        self,
        x,
        y,
        agent_id,
    ):

        return (
            0.5
            + 0.5
            * math.sin(
                x * 0.010
                - y * 0.007
                + self.time * 0.019
                + agent_id * 1.7
            )
        )

    def mirror_strength(self):

        return (
            1.0
            if self.topology_phase > 0.82
            else 0.0
        )

    def wormhole_strength(self):

        return clamp(
            (
                self.global_phase
                - 0.76
            )
            / 0.24,
            0.0,
            1.0,
        )

    def past_leak_strength(self):

        return clamp(
            (
                self.past_phase
                - 0.72
            )
            / 0.28,
            0.0,
            1.0,
        )

    def gravity_is_up(self):

        return (
            self.gravity_sign < 0
        )

    def frames_until_gravity_flip(
        self,
    ):

        return (
            GRAVITY_SWITCH_PERIOD
            - self.gravity_timer
        )

    def gravity_flip_imminent(
        self,
    ):

        return (
            self.frames_until_gravity_flip()
            <= GRAVITY_WARNING_TIME
        )


# ============================================================
# Visual Cell
# ============================================================


class VisualCell:

    def __init__(
        self,
        cell_id,
        feature,
        gx,
        gy,
    ):

        self.id = cell_id

        self.feature = np.asarray(
            feature,
            dtype=np.float32,
        )

        self.gx = gx
        self.gy = gy

        self.visits = 1

        self.activation = 1.0
        self.energy = 1.0

        self.phase_counts = {}

    def similarity(
        self,
        feature,
    ):

        return soft_distance(
            self.feature,
            feature,
            4.3,
        )

    def update(
        self,
        feature,
        phase_bucket,
    ):

        feature = np.asarray(
            feature,
            dtype=np.float32,
        )

        self.feature = (
            0.91 * self.feature
            + 0.09 * feature
        )

        self.visits += 1

        self.activation = min(
            2.0,
            self.activation + 0.04,
        )

        self.energy = min(
            2.0,
            self.energy + 0.02,
        )

        self.phase_counts[
            phase_bucket
        ] = (
            self.phase_counts.get(
                phase_bucket,
                0,
            )
            + 1
        )

    def decay(self):

        self.activation *= 0.994
        self.energy *= 0.999


# ============================================================
# Spatial Representation
# ============================================================


class SpatialRepresentation:

    def __init__(self):

        self.cells = {}
        self.next_id = 0

    def patches(
        self,
        image,
    ):

        image = np.asarray(
            image,
            dtype=np.float32,
        ).reshape(
            OBS_H,
            OBS_W,
            CHANNELS,
        )

        out = []

        for y in range(
            0,
            OBS_H,
            PATCH,
        ):

            for x in range(
                0,
                OBS_W,
                PATCH,
            ):

                patch = image[
                    y:y + PATCH,
                    x:x + PATCH,
                ]

                feat = patch.mean(
                    axis=(0, 1)
                )

                out.append(
                    (
                        x // PATCH,
                        y // PATCH,
                        feat,
                    )
                )

        return out

    def encode(
        self,
        image,
        phase_bucket,
    ):

        active = []

        for (
            gx,
            gy,
            feat,
        ) in self.patches(
            image
        ):

            key = (
                gx,
                gy,
            )

            best = None
            best_score = 0.0

            for c in self.cells.get(
                key,
                [],
            ):

                s = c.similarity(
                    feat
                )

                if s > best_score:

                    best_score = s
                    best = c

            if (
                best is None
                or best_score
                < VISUAL_SIM_THRESHOLD
            ):

                c = VisualCell(
                    self.next_id,
                    feat,
                    gx,
                    gy,
                )

                self.next_id += 1

                self.cells.setdefault(
                    key,
                    [],
                ).append(c)

            else:

                best.update(
                    feat,
                    phase_bucket,
                )

                c = best

            active.append(c)

        return active

    def feature_vector(
        self,
        image,
    ):

        values = []

        for (
            _,
            _,
            feat,
        ) in self.patches(
            image
        ):

            values.extend(
                feat.tolist()
            )

        return np.asarray(
            values,
            dtype=np.float32,
        )

    def count(self):

        return sum(
            len(v)
            for v in self.cells.values()
        )

    def decay(self):

        for cells in (
            self.cells.values()
        ):

            for c in cells:

                c.decay()


# ============================================================
# Place State
# ============================================================


class PlaceState:

    def __init__(
        self,
        place_id,
        feature,
        signature,
    ):

        self.id = place_id

        self.center = np.asarray(
            feature,
            dtype=np.float32,
        )

        self.signature = np.asarray(
            signature,
            dtype=np.float32,
        )

        self.visits = 1

        self.energy = 1.0
        self.activation = 1.0

        self.transitions = {}

        self.state_history = deque(
            maxlen=MEMORY_WINDOW
        )

        self.phase_histogram = {}

        self.split_pressure = 0.0
        self.merge_pressure = 0.0

    def similarity(
        self,
        feature,
        signature,
    ):

        a = soft_distance(
            self.center,
            feature,
            5.0,
        )

        b = soft_distance(
            self.signature,
            signature,
            7.0,
        )

        return (
            0.65 * a
            + 0.35 * b
        )

    def absorb(
        self,
        feature,
        signature,
        phase_bucket,
        state_vector,
    ):

        feature = np.asarray(
            feature,
            dtype=np.float32,
        )

        signature = np.asarray(
            signature,
            dtype=np.float32,
        )

        self.center = (
            0.92 * self.center
            + 0.08 * feature
        )

        self.signature = (
            0.90 * self.signature
            + 0.10 * signature
        )

        self.visits += 1

        self.energy = min(
            2.0,
            self.energy + 0.015,
        )

        self.activation = min(
            2.0,
            self.activation + 0.05,
        )

        self.phase_histogram[
            phase_bucket
        ] = (
            self.phase_histogram.get(
                phase_bucket,
                0,
            )
            + 1
        )

        self.state_history.append(
            np.asarray(
                state_vector,
                dtype=np.float32,
            )
        )

    def connect(
        self,
        target_id,
    ):

        self.transitions[
            target_id
        ] = (
            self.transitions.get(
                target_id,
                0,
            )
            + 1
        )

    def decay(self):

        self.activation *= 0.994
        self.energy *= 0.999


# ============================================================
# Transformation Cell
#
# v4:
#   Transformation now becomes a predictive model.
# ============================================================


class TransformationCell:

    def __init__(
        self,
        transform_id,
        before_place,
        after_place,
        action,
        delta,
    ):

        self.id = transform_id

        self.before_place = (
            before_place
        )

        self.after_place = (
            after_place
        )

        self.action_counts = {
            action: 1
        }

        self.delta = np.asarray(
            delta,
            dtype=np.float32,
        )

        self.visits = 1

        self.energy = 1.0

        self.stability = 0.1
        self.error = 1.0

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        self.prediction_error = 1.0

        self.prediction_success = 0.0

        self.prediction_history = deque(
            maxlen=MEMORY_WINDOW
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        self.contexts = deque(
            maxlen=80
        )

        self.history = deque(
            maxlen=MEMORY_WINDOW
        )

        # ----------------------------------------------------
        # Counterfactual statistics
        # ----------------------------------------------------

        self.counterfactual_count = 0

        self.counterfactual_energy = 0.0

    def similarity(
        self,
        delta,
        before_place,
    ):

        if (
            self.before_place
            != before_place
        ):

            return 0.0

        return soft_distance(
            self.delta,
            delta,
            2.8,
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        state,
    ):

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        # x + delta
        return (
            state
            + self.delta
        )

    def predict_delta(
        self,
    ):

        return self.delta.copy()

    def prediction_score(
        self,
        predicted,
        actual,
    ):

        err = l1_distance(
            predicted,
            actual,
        )

        self.prediction_error = (
            0.92
            * self.prediction_error
            + 0.08 * err
        )

        success = math.exp(
            -PREDICTION_ERROR_SCALE
            * err
        )

        self.prediction_success = (
            0.92
            * self.prediction_success
            + 0.08
            * success
        )

        self.prediction_history.append(
            float(err)
        )

        return err

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        delta,
        action,
        context,
    ):

        delta = np.asarray(
            delta,
            dtype=np.float32,
        )

        err = l1_distance(
            self.delta,
            delta,
        )

        self.delta = (
            0.88
            * self.delta
            + 0.12 * delta
        )

        self.error = (
            0.90
            * self.error
            + 0.10 * err
        )

        self.stability = clamp(
            0.97
            * self.stability
            + 0.03
            * (1.0 - err),
            0.0,
            1.0,
        )

        self.visits += 1

        self.energy = min(
            2.0,
            self.energy + 0.02,
        )

        self.action_counts[
            action
        ] = (
            self.action_counts.get(
                action,
                0,
            )
            + 1
        )

        self.contexts.append(
            np.asarray(
                context,
                dtype=np.float32,
            )
        )

        self.history.append(
            float(err)
        )

    # ========================================================
    # Counterfactual
    # ========================================================

    def counterfactual_value(
        self,
    ):

        self.counterfactual_count += 1

        uncertainty = (
            1.0
            - self.prediction_success
        )

        novelty = (
            1.0
            / math.sqrt(
                max(
                    1,
                    self.visits,
                )
            )
        )

        value = (
            0.55 * uncertainty
            + 0.45 * novelty
        )

        self.counterfactual_energy = (
            0.95
            * self.counterfactual_energy
            + 0.05
            * value
        )

        return value

    # ========================================================
    # Curiosity
    # ========================================================

    def curiosity(self):

        novelty = (
            1.0
            / math.sqrt(
                max(
                    1,
                    self.visits,
                )
            )
        )

        instability = (
            self.prediction_error
        )

        uncertainty = (
            1.0
            - self.prediction_success
        )

        isolation = (
            1.0
            if len(
                self.action_counts
            ) == 1
            else 0.25
        )

        return (
            0.45 * novelty
            + 1.2 * instability
            + 0.8 * uncertainty
            + 0.35 * isolation
            + 0.6
            * (
                1.0
                - self.stability
            )
        )

    def decay(self):

        self.energy *= 0.998


# ============================================================
# Meta Transformation
# ============================================================


class MetaTransformationCell:

    def __init__(
        self,
        meta_id,
        delta_a,
        delta_b,
    ):

        self.id = meta_id

        self.delta_a = np.asarray(
            delta_a,
            dtype=np.float32,
        )

        self.delta_b = np.asarray(
            delta_b,
            dtype=np.float32,
        )

        self.meta_delta = (
            self.delta_b
            - self.delta_a
        )

        self.visits = 1

        self.energy = 1.0

        self.stability = 0.2

    def similarity(
        self,
        a,
        b,
    ):

        d1 = soft_distance(
            self.delta_a,
            a,
            2.0,
        )

        d2 = soft_distance(
            self.delta_b,
            b,
            2.0,
        )

        return (
            0.5
            * (
                d1 + d2
            )
        )

    def update(
        self,
        a,
        b,
    ):

        a = np.asarray(
            a,
            dtype=np.float32,
        )

        b = np.asarray(
            b,
            dtype=np.float32,
        )

        new_meta = b - a

        err = l1_distance(
            self.meta_delta,
            new_meta,
        )

        self.delta_a = (
            0.92
            * self.delta_a
            + 0.08 * a
        )

        self.delta_b = (
            0.92
            * self.delta_b
            + 0.08 * b
        )

        self.meta_delta = (
            0.92
            * self.meta_delta
            + 0.08
            * new_meta
        )

        self.stability = clamp(
            0.96
            * self.stability
            + 0.04
            * (1.0 - err),
            0.0,
            1.0,
        )

        self.visits += 1

        self.energy = min(
            2.0,
            self.energy + 0.02,
        )


# ============================================================
# Temporal Trace
# ============================================================


class TemporalTrace:

    def __init__(
        self,
        feature,
        body,
        place_id,
        time_index,
        action,
    ):

        self.feature = np.asarray(
            feature,
            dtype=np.float32,
        ).copy()

        self.body = np.asarray(
            body,
            dtype=np.float32,
        ).copy()

        self.place_id = place_id

        self.time_index = (
            time_index
        )

        self.action = action

        self.strength = 1.0

    def decay(self):

        self.strength *= 0.995


# ============================================================
# World Model
# ============================================================


class WorldModel:

    def __init__(self):

        self.visual = (
            SpatialRepresentation()
        )

        self.places = []

        self.transforms = []

        self.meta_transforms = []

        self.events = []

        self.traces = deque(
            maxlen=700
        )

        self.next_place = 0
        self.next_transform = 0
        self.next_meta = 0
        self.next_event = 0

        self.previous_state = {}

        self.previous_transform = {}

        self.error_history = deque(
            maxlen=500
        )

        self.prediction_error_history = (
            deque(
                maxlen=500
            )
        )

        self.surprise_history = deque(
            maxlen=500
        )

        self.phase_history = deque(
            maxlen=200
        )

        self.event_history = deque(
            maxlen=300
        )

        self.step_count = 0

        self.split_events = 0
        self.merge_events = 0

    # ========================================================
    # State representation
    # ========================================================

    def make_state(
        self,
        image,
        body,
        local_context,
    ):

        visual = (
            self.visual.feature_vector(
                image
            )
        )

        return np.concatenate(
            [
                visual[::4],
                body,
                np.asarray(
                    local_context,
                    dtype=np.float32,
                ),
            ]
        )

    # ========================================================
    # Place signature
    # ========================================================

    def place_signature(
        self,
        image,
        body,
        local_context,
    ):

        visual = (
            self.visual.feature_vector(
                image
            )
        )

        sig = np.concatenate(
            [
                visual[::4],
                body,
                np.asarray(
                    local_context,
                    dtype=np.float32,
                ),
            ]
        )

        return visual, sig

    # ========================================================
    # Encode Place
    # ========================================================

    def encode_place(
        self,
        image,
        body,
        local_context,
        phase_bucket,
        state_vector,
    ):

        feature, signature = (
            self.place_signature(
                image,
                body,
                local_context,
            )
        )

        best = None
        best_score = 0.0

        for place in self.places:

            s = place.similarity(
                feature,
                signature,
            )

            if s > best_score:

                best_score = s
                best = place

        if (
            best is None
            or best_score
            < PLACE_SIM_THRESHOLD
        ):

            place = PlaceState(
                self.next_place,
                feature,
                signature,
            )

            self.next_place += 1

            place.phase_histogram[
                phase_bucket
            ] = 1

            place.state_history.append(
                np.asarray(
                    state_vector,
                    dtype=np.float32,
                )
            )

            self.places.append(
                place
            )

            created = True

        else:

            nearest_phase = (
                max(
                    best.phase_histogram.values()
                )
                if best.phase_histogram
                else 0
            )

            total_phase = (
                sum(
                    best.phase_histogram.values()
                )
                if best.phase_histogram
                else 1
            )

            phase_purity = (
                nearest_phase
                / max(
                    1,
                    total_phase,
                )
            )

            if (
                phase_purity < 0.35
                and best.visits > 12
            ):

                best.split_pressure += (
                    0.015
                )

            best.absorb(
                feature,
                signature,
                phase_bucket,
                state_vector,
            )

            place = best

            created = False

        return (
            place,
            feature,
            signature,
            created,
        )

    # ========================================================
    # Place Split
    # ========================================================

    def maybe_split_place(
        self,
        place,
    ):

        if (
            place.split_pressure < 1.0
            or len(
                place.state_history
            ) < 15
        ):

            return None

        history = list(
            place.state_history
        )

        tail = np.mean(
            np.stack(
                history[-7:]
            ),
            axis=0,
        )

        old_center = np.mean(
            np.stack(
                history[:7]
            ),
            axis=0,
        )

        if (
            l1_distance(
                tail,
                old_center,
            )
            < 0.10
        ):

            place.split_pressure *= (
                0.5
            )

            return None

        new_place = PlaceState(
            self.next_place,
            place.center.copy(),
            place.signature.copy(),
        )

        self.next_place += 1

        new_place.state_history.extend(
            history[-7:]
        )

        new_place.visits = max(
            1,
            place.visits // 5,
        )

        new_place.activation = (
            place.activation
            * 0.8
        )

        place.split_pressure *= (
            0.35
        )

        self.places.append(
            new_place
        )

        self.split_events += 1

        return new_place

    # ========================================================
    # Transform search
    # ========================================================

    def find_transform(
        self,
        before_place,
        delta,
    ):

        best = None
        best_score = 0.0

        for t in self.transforms:

            s = t.similarity(
                delta,
                before_place.id,
            )

            if s > best_score:

                best_score = s
                best = t

        return (
            best,
            best_score,
        )

    # ========================================================
    # Form transform
    # ========================================================

    def form_transform(
        self,
        before,
        after,
        action,
        delta,
        context,
    ):

        t, score = (
            self.find_transform(
                before,
                delta,
            )
        )

        if (
            t is None
            or score
            < TRANSFORM_SIM_THRESHOLD
        ):

            t = TransformationCell(
                self.next_transform,
                before.id,
                after.id,
                action,
                delta,
            )

            self.next_transform += 1

            t.contexts.append(
                np.asarray(
                    context,
                    dtype=np.float32,
                )
            )

            self.transforms.append(
                t
            )

            error = 1.0

        else:

            before_after_distance = (
                float(
                    after.id
                    != t.after_place
                )
            )

            t.update(
                delta,
                action,
                context,
            )

            error = (
                0.5 * t.error
                + 0.5
                * before_after_distance
            )

            if (
                t.after_place
                != after.id
            ):

                before.merge_pressure += (
                    0.02
                )

                t.after_place = (
                    after.id
                )

        before.connect(
            after.id
        )

        self.error_history.append(
            error
        )

        return (
            t,
            error,
        )

    # ========================================================
    # Meta
    # ========================================================

    def form_meta(
        self,
        transform_a,
        transform_b,
    ):

        if (
            transform_a is None
            or transform_b is None
        ):

            return

        best = None
        score = 0.0

        for m in self.meta_transforms:

            s = m.similarity(
                transform_a.delta,
                transform_b.delta,
            )

            if s > score:

                score = s
                best = m

        if (
            best is None
            or score
            < META_SIM_THRESHOLD
        ):

            m = MetaTransformationCell(
                self.next_meta,
                transform_a.delta,
                transform_b.delta,
            )

            self.next_meta += 1

            self.meta_transforms.append(
                m
            )

        else:

            best.update(
                transform_a.delta,
                transform_b.delta,
            )

    # ========================================================
    # Event Memory
    # ========================================================

    def remember_event(
        self,
        event,
        place,
        action,
        transform,
        context,
    ):

        best = None
        best_score = 0.0

        for e in self.events:

            score = e.similarity(
                event.name,
                place.id,
                action,
                context,
            )

            if score > best_score:

                best_score = score
                best = e

        if (
            best is None
            or best_score
            < EVENT_SIM_THRESHOLD
        ):

            cell = EventMemoryCell(
                self.next_event,
                event.name,
                place.id,
                action,
                (
                    transform.id
                    if transform is not None
                    else -1
                ),
                context,
            )

            self.next_event += 1

            cell.last_time = (
                event.time
            )

            self.events.append(
                cell
            )

        else:

            best.update(
                context,
                (
                    transform.id
                    if transform is not None
                    else -1
                ),
                event.time,
            )

            cell = best

        self.event_history.append(
            (
                event.name,
                place.id,
                action,
                cell.id,
            )
        )

        return cell

    # ========================================================
    # Event prediction
    # ========================================================

    def predict_events(
        self,
        place_id,
        action,
    ):

        candidates = [
            e
            for e in self.events
            if (
                e.place_id
                == place_id
                and e.action
                == action
            )
        ]

        candidates.sort(
            key=lambda e:
                e.confidence,
            reverse=True,
        )

        return candidates[:5]

    # ========================================================
    # Learn
    # ========================================================

    def learn(
        self,
        agent_id,
        image_before,
        body_before,
        action,
        image_after,
        body_after,
        local_context_before,
        local_context_after,
        phase_bucket,
        events,
    ):

        self.step_count += 1

        # ----------------------------------------------------
        # Visual encoding
        # ----------------------------------------------------

        self.visual.encode(
            image_before,
            phase_bucket,
        )

        self.visual.encode(
            image_after,
            phase_bucket,
        )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        state_before = self.make_state(
            image_before,
            body_before,
            local_context_before,
        )

        state_after = self.make_state(
            image_after,
            body_after,
            local_context_after,
        )

        # ----------------------------------------------------
        # Places
        # ----------------------------------------------------

        (
            before,
            _,
            _,
            created_before,
        ) = self.encode_place(
            image_before,
            body_before,
            local_context_before,
            phase_bucket,
            state_before,
        )

        (
            after,
            _,
            _,
            created_after,
        ) = self.encode_place(
            image_after,
            body_after,
            local_context_after,
            phase_bucket,
            state_after,
        )

        # ----------------------------------------------------
        # Actual transformation
        # ----------------------------------------------------

        delta = (
            state_after
            - state_before
        )

        context = np.concatenate(
            [
                body_before,
                body_after,
                np.asarray(
                    local_context_before,
                    dtype=np.float32,
                ),
            ]
        )

        (
            transform,
            error,
        ) = self.form_transform(
            before,
            after,
            action,
            delta,
            context,
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted_state = (
            transform.predict(
                state_before
            )
        )

        prediction_error = (
            transform.prediction_score(
                predicted_state,
                state_after,
            )
        )

        self.prediction_error_history.append(
            prediction_error
        )

        # ----------------------------------------------------
        # Surprise
        # ----------------------------------------------------

        surprise = (
            prediction_error
            * (
                1.0
                + transform.curiosity()
            )
        )

        self.surprise_history.append(
            surprise
        )

        # ----------------------------------------------------
        # Meta transformation
        # ----------------------------------------------------

        prev_t = (
            self.previous_transform.get(
                agent_id
            )
        )

        if prev_t is not None:

            self.form_meta(
                prev_t,
                transform,
            )

        self.previous_transform[
            agent_id
        ] = transform

        # ----------------------------------------------------
        # Temporal trace
        # ----------------------------------------------------

        trace = TemporalTrace(
            state_after[::4],
            body_after,
            after.id,
            self.step_count,
            action,
        )

        self.traces.append(
            trace
        )

        self.previous_state[
            agent_id
        ] = after.id

        self.phase_history.append(
            phase_bucket
        )

        # ----------------------------------------------------
        # Event memory
        # ----------------------------------------------------

        event_cells = []

        for event in events:

            event_cell = (
                self.remember_event(
                    event,
                    after,
                    action,
                    transform,
                    local_context_after,
                )
            )

            event_cells.append(
                event_cell
            )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        split = self.maybe_split_place(
            before
        )

        return {
            "before": before,
            "after": after,
            "transform": transform,
            "error": error,
            "prediction_error":
                prediction_error,
            "surprise": surprise,
            "events":
                event_cells,
            "created_before":
                created_before,
            "created_after":
                created_after,
            "split": split,
        }

    # ========================================================
    # Find action transforms
    # ========================================================

    def action_transforms(
        self,
        place_id,
        action,
    ):

        return [
            t
            for t in self.transforms
            if (
                t.before_place
                == place_id
                and action
                in t.action_counts
            )
        ]

    # ========================================================
    # Predict Action
    # ========================================================

    def predict_action(
        self,
        image,
        body,
        local_context,
        action,
    ):

        feature, signature = (
            self.place_signature(
                image,
                body,
                local_context,
            )
        )

        best_place = None
        best_score = 0.0

        for p in self.places:

            s = p.similarity(
                feature,
                signature,
            )

            if s > best_score:

                best_score = s
                best_place = p

        if best_place is None:

            return {
                "place": None,
                "transform": None,
                "predicted_state": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "novelty": 1.0,
                "risk": 0.0,
                "event_predictions": [],
            }

        candidates = (
            self.action_transforms(
                best_place.id,
                action,
            )
        )

        # ----------------------------------------------------
        # 未知のAction
        # ----------------------------------------------------

        if not candidates:

            return {
                "place": best_place,
                "transform": None,
                "predicted_state": None,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "novelty": 1.0,
                "risk": 0.0,
                "event_predictions":
                    self.predict_events(
                        best_place.id,
                        action,
                    ),
            }

        # ----------------------------------------------------
        # 複数モデルの中から最も信頼できるもの
        # ----------------------------------------------------

        t = max(
            candidates,
            key=lambda x:
                (
                    x.prediction_success
                    * 0.65
                    + x.stability
                    * 0.35
                ),
        )

        state = np.concatenate(
            [
                feature[::4],
                body,
                np.asarray(
                    local_context,
                    dtype=np.float32,
                ),
            ]
        )

        predicted = t.predict(
            state
        )

        confidence = (
            t.prediction_success
        )

        uncertainty = (
            1.0 - confidence
        )

        novelty = (
            1.0
            / math.sqrt(
                max(
                    1,
                    t.visits,
                )
            )
        )

        # ----------------------------------------------------
        # Body prediction
        #
        # state:
        #   [visual..., body(8), context(7)]
        # ----------------------------------------------------

        body_start = (
            len(state) - 15
        )

        body_end = (
            len(state) - 7
        )

        predicted_body = (
            predicted[
                body_start:body_end
            ]
        )

        # ----------------------------------------------------
        # 危険度
        # ----------------------------------------------------

        risk = 0.0

        predicted_y = (
            predicted_body[1]
        )

        predicted_vy = (
            predicted_body[3]
        )

        predicted_grounded = (
            predicted_body[4]
        )

        # 画面端
        predicted_x = (
            predicted_body[0]
        )

        if (
            abs(predicted_x)
            > 0.94
        ):

            risk += 0.25

        # 高すぎる / 低すぎる
        if (
            abs(predicted_y)
            > 0.92
        ):

            risk += 0.25

        # 急激な速度
        if (
            abs(predicted_vy)
            > 0.90
        ):

            risk += 0.20

        # ----------------------------------------------------
        # Event prediction
        # ----------------------------------------------------

        predicted_events = (
            self.predict_events(
                best_place.id,
                action,
            )
        )

        for event in predicted_events:

            if (
                event.event_name
                in (
                    "HAZARD",
                    "IMPACT",
                    "BOUNDARY",
                )
            ):

                risk += (
                    0.10
                    * event.confidence
                )

        return {
            "place": best_place,
            "transform": t,
            "predicted_state": predicted,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "novelty": novelty,
            "risk": clamp(
                risk,
                0.0,
                1.0,
            ),
            "event_predictions":
                predicted_events,
        }

    # ========================================================
    # Counterfactual
    #
    # 「今ここで別のActionを選んだら？」
    # ========================================================

    def counterfactual(
        self,
        image,
        body,
        local_context,
        action,
    ):

        prediction = (
            self.predict_action(
                image,
                body,
                local_context,
                action,
            )
        )

        if (
            prediction["transform"]
            is None
        ):

            return {
                "action": action,
                "known": False,
                "value": 2.0,
                "risk": 0.0,
                "uncertainty": 1.0,
                "novelty": 1.0,
                "confidence": 0.0,
            }

        t = prediction[
            "transform"
        ]

        curiosity = (
            t.curiosity()
        )

        uncertainty = (
            prediction[
                "uncertainty"
            ]
        )

        novelty = (
            prediction[
                "novelty"
            ]
        )

        risk = (
            prediction["risk"]
        )

        # ----------------------------------------------------
        # Counterfactual value
        # ----------------------------------------------------

        exploration_value = (
            NOVELTY_WEIGHT
            * novelty
            + UNCERTAINTY_WEIGHT
            * uncertainty
            + PREDICTION_WEIGHT
            * curiosity
        )

        # 危険すぎる未来は少し避ける
        value = (
            exploration_value
            - RISK_WEIGHT
            * risk
        )

        t.counterfactual_value()

        return {
            "action": action,
            "known": True,
            "value": value,
            "risk": risk,
            "uncertainty":
                uncertainty,
            "novelty":
                novelty,
            "confidence":
                prediction[
                    "confidence"
                ],
            "transform": t,
            "prediction":
                prediction,
        }

    # ========================================================
    # Action score
    # ========================================================

    def score_action(
        self,
        image,
        body,
        local_context,
        action,
        recent_action,
    ):

        score = (
            ACTION_ACTIVITY_BIAS.get(
                action,
                0.0,
            )
        )

        cf = self.counterfactual(
            image,
            body,
            local_context,
            action,
        )

        # ----------------------------------------------------
        # Unknown action
        # ----------------------------------------------------

        if not cf["known"]:

            score += 2.5

            score += (
                0.8
            )

        else:

            score += (
                COUNTERFACTUAL_WEIGHT
                * cf["value"]
            )

            score += (
                PREDICTION_WEIGHT
                * cf["uncertainty"]
            )

            score += (
                NOVELTY_WEIGHT
                * cf["novelty"]
            )

            # 安定しているモデル
            score += (
                0.45
                * cf["confidence"]
            )

            # 危険度
            score -= (
                RISK_WEIGHT
                * cf["risk"]
            )

            t = cf[
                "transform"
            ]

            # 分岐の多いPlaceを少し優先
            if (
                t is not None
                and cf.get(
                    "prediction"
                ) is not None
            ):

                place = cf[
                    "prediction"
                ]["place"]

                if place is not None:

                    branch = len(
                        place.transitions
                    )

                    score += min(
                        1.0,
                        branch
                        * BRANCH_WEIGHT,
                    )

        # ----------------------------------------------------
        # 連続行動
        # ----------------------------------------------------

        if (
            recent_action
            in (
                ACTION_LEFT,
                ACTION_RIGHT,
            )
            and action
            == recent_action
        ):

            score += 0.12

        if (
            recent_action is not None
            and action
            == recent_action
        ):

            score -= 0.025

        return score

    # ========================================================
    # Select Action
    # ========================================================

    def select_action(
        self,
        image,
        body,
        local_context,
        recent_action,
    ):

        scores = [
            self.score_action(
                image,
                body,
                local_context,
                action,
                recent_action,
            )
            for action in ACTIONS
        ]

        order = np.argsort(
            scores
        )[::-1]

        # ----------------------------------------------------
        # 少し探索
        # ----------------------------------------------------

        if (
            len(order) >= 3
            and random.random()
            < 0.28
        ):

            selected = random.choice(
                order[:3]
            )

            return ACTIONS[
                int(selected)
            ]

        return ACTIONS[
            int(order[0])
        ]

    # ========================================================
    # Replay
    # ========================================================

    def replay(
        self,
        count=300,
    ):

        if (
            not self.traces
            or not self.transforms
        ):

            return

        weights = np.asarray(
            [
                1.0
                + 1.5 * t.error
                + 1.0
                * t.curiosity()
                + 0.8
                * (
                    1.0
                    - t.stability
                )
                + 0.7
                * (
                    1.0
                    - t.prediction_success
                )
                for t in self.transforms
            ],
            dtype=np.float64,
        )

        weights = np.maximum(
            weights,
            1e-8,
        )

        weights /= weights.sum()

        for _ in range(count):

            t = np.random.choice(
                self.transforms,
                p=weights,
            )

            t.energy = min(
                2.0,
                t.energy + 0.02,
            )

            if (
                t.error
                > 0.35
            ):

                t.stability *= 0.999

            else:

                t.stability = clamp(
                    t.stability + 0.001,
                    0.0,
                    1.0,
                )

            # 予測誤差の高いモデルを
            # replayによって再活性化
            if (
                t.prediction_error
                > 0.25
            ):

                t.energy = min(
                    2.0,
                    t.energy + 0.01,
                )

        for trace in self.traces:

            trace.decay()

    # ========================================================
    # Decay
    # ========================================================

    def decay(self):

        self.visual.decay()

        for p in self.places:

            p.decay()

        for t in self.transforms:

            t.decay()

        for m in self.meta_transforms:

            m.energy *= 0.998

        for e in self.events:

            e.decay()

    # ========================================================
    # Statistics
    # ========================================================

    def statistics(self):

        mean_error = (
            float(
                np.mean(
                    self.error_history
                )
            )
            if self.error_history
            else 0.0
        )

        mean_prediction_error = (
            float(
                np.mean(
                    self.prediction_error_history
                )
            )
            if self.prediction_error_history
            else 0.0
        )

        mean_surprise = (
            float(
                np.mean(
                    self.surprise_history
                )
            )
            if self.surprise_history
            else 0.0
        )

        avg_stability = (
            float(
                np.mean(
                    [
                        t.stability
                        for t in self.transforms
                    ]
                )
            )
            if self.transforms
            else 0.0
        )

        avg_prediction_success = (
            float(
                np.mean(
                    [
                        t.prediction_success
                        for t in self.transforms
                    ]
                )
            )
            if self.transforms
            else 0.0
        )

        avg_split = (
            float(
                np.mean(
                    [
                        p.split_pressure
                        for p in self.places
                    ]
                )
            )
            if self.places
            else 0.0
        )

        return {
            "visual":
                self.visual.count(),

            "places":
                len(self.places),

            "transforms":
                len(self.transforms),

            "meta":
                len(
                    self.meta_transforms
                ),

            "events":
                len(self.events),

            "mean_error":
                mean_error,

            "prediction_error":
                mean_prediction_error,

            "prediction_success":
                avg_prediction_success,

            "surprise":
                mean_surprise,

            "stability":
                avg_stability,

            "split_pressure":
                avg_split,

            "splits":
                self.split_events,

            "merges":
                self.merge_events,
        }


# ============================================================
# Physical Strange World
# ============================================================


class World:

    def __init__(self):

        self.drawer = turtle.Turtle(
            visible=False
        )

        self.drawer.penup()
        self.drawer.speed(0)

        self.regime = (
            WorldRegime()
        )

        self.time = 0

        # ----------------------------------------------------
        # Platforms
        # ----------------------------------------------------

        self.base_platforms = [
            (-530, -220, -220),
            (-470, -300, -105),
            (-340, -250, 40),
            (-230, -80, -40),
            (-50, 40, 75),
            (80, 240, -20),
            (180, 350, 100),
            (315, 510, -90),
            (430, 550, 50),
        ]

        self.ceiling_platforms = [
            (-500, -350, 235),
            (-330, -190, 195),
            (-150, -20, 250),
            (30, 170, 205),
            (190, 360, 245),
            (380, 520, 185),
        ]

        self.extra_platforms = [
            (-510, -450, 155),
            (-160, -30, 170),
            (110, 220, 175),
            (260, 390, 205),
        ]

        self.hazards = [
            (-390, -340, -130),
            (-235, -190, -65),
            (125, 180, -45),
            (350, 400, -115),
        ]

        self.moving_x = 0.0
        self.moving_y = 145.0

        self.ghost_x = 0.0
        self.ghost_y = -10.0

        self.impact_marks = []

        self.gravity_warning = None

        self.last_gravity_sign = (
            self.regime.gravity_sign
        )

        # ----------------------------------------------------
        # Event memory buffer
        # ----------------------------------------------------

        self.events = deque(
            maxlen=500
        )

        self.event_serial = 0

        self.draw()

    # ========================================================
    # Events
    # ========================================================

    def emit_event(
        self,
        name,
        x=0.0,
        y=0.0,
        intensity=1.0,
        source=-1,
    ):

        event = WorldEvent(
            name,
            self.time,
            x,
            y,
            intensity,
            source,
        )

        self.events.append(
            event
        )

        self.event_serial += 1

        return event

    def events_since(
        self,
        last_serial,
    ):

        # event_serial is global.
        #
        # deque itself is intentionally kept
        # small, so events from the current
        # episode remain available.

        if (
            last_serial
            >= self.event_serial
        ):

            return []

        # For simplicity, return recent events.
        # Agents filter by time/source.

        return list(
            self.events
        )[-20:]

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):

        self.time = 0

        self.regime = (
            WorldRegime()
        )

        self.moving_x = 0.0
        self.moving_y = 145.0

        self.ghost_x = 0.0
        self.ghost_y = -10.0

        self.impact_marks.clear()

        self.gravity_warning = None

        self.last_gravity_sign = (
            self.regime.gravity_sign
        )

        self.events.clear()

        self.event_serial = 0

        self.draw()

    # ========================================================
    # Update
    # ========================================================

    def update(self):

        self.regime.update()

        self.time = (
            self.regime.time
        )

        scale = (
            self.regime.time_scale()
        )

        self.moving_x = (
            245.0
            * math.sin(
                self.time
                * 0.028
                * scale
            )
        )

        self.moving_y = (
            145.0
            + 65.0
            * math.sin(
                self.time * 0.047
                + self.regime.global_phase
                * math.pi
            )
        )

        self.ghost_x = (
            170.0
            * math.sin(
                self.time * 0.015
                - 1.1
            )
        )

        self.ghost_y = (
            45.0
            + 100.0
            * math.cos(
                self.time * 0.021
            )
        )

        # ----------------------------------------------------
        # Gravity warning
        # ----------------------------------------------------

        if (
            self.regime.gravity_flip_imminent()
        ):

            self.gravity_warning = (
                GravityWarning(
                    0,
                    150,
                )
            )

        else:

            self.gravity_warning = None

        # ----------------------------------------------------
        # Gravity flip
        # ----------------------------------------------------

        if (
            self.regime.gravity_sign
            != self.last_gravity_sign
        ):

            self.impact_marks.append(
                ImpactMark(
                    0,
                    0,
                    self.regime.gravity_sign,
                )
            )

            self.emit_event(
                "GRAVITY_FLIP",
                0,
                0,
                1.0,
            )

            self.last_gravity_sign = (
                self.regime.gravity_sign
            )

        # ----------------------------------------------------
        # Marks
        # ----------------------------------------------------

        for mark in (
            self.impact_marks
        ):

            mark.update()

        self.impact_marks = [
            mark
            for mark
            in self.impact_marks
            if mark.alive()
        ]

        if self.gravity_warning:

            self.gravity_warning.update()

    # ========================================================
    # Draw line
    # ========================================================

    def draw_line(
        self,
        x1,
        y1,
        x2,
        y2,
        width,
        color,
    ):

        d = self.drawer

        d.color(color)

        d.pensize(width)

        d.penup()

        d.goto(
            x1,
            y1,
        )

        d.pendown()

        d.goto(
            x2,
            y2,
        )

        d.penup()

    # ========================================================
    # Effective platforms
    # ========================================================

    def effective_platforms(self):

        phase = (
            self.regime.topology_phase
        )

        out = []

        for i, (
            x1,
            x2,
            y,
        ) in enumerate(
            self.base_platforms
        ):

            wobble = (
                28.0
                * math.sin(
                    self.time
                    * 0.026
                    + i * 1.3
                )
                * phase
            )

            shear = (
                18.0
                * math.sin(
                    self.time
                    * 0.019
                    + i
                )
            )

            out.append(
                (
                    x1 + shear,
                    x2 - shear * 0.4,
                    y + wobble,
                )
            )

        if phase > 0.48:

            out.extend(
                self.extra_platforms[
                    :int(
                        1
                        + phase
                        * len(
                            self.extra_platforms
                        )
                    )
                ]
            )

        return out

    # ========================================================
    # Ceiling
    # ========================================================

    def effective_ceiling_platforms(
        self,
    ):

        out = []

        for i, (
            x1,
            x2,
            y,
        ) in enumerate(
            self.ceiling_platforms
        ):

            wobble = (
                16.0
                * math.sin(
                    self.time
                    * 0.034
                    + i * 1.7
                )
            )

            out.append(
                (
                    x1,
                    x2,
                    y + wobble,
                )
            )

        return out

    # ========================================================
    # Platform
    # ========================================================

    def platform_y(
        self,
        x,
        y,
    ):

        for (
            x1,
            x2,
            py,
        ) in self.effective_platforms():

            if (
                x1 <= x <= x2
                and abs(y - py) < 24
            ):

                return py

        return None

    # ========================================================
    # Ceiling
    # ========================================================

    def ceiling_y(
        self,
        x,
        y,
    ):

        for (
            x1,
            x2,
            py,
        ) in self.effective_ceiling_platforms():

            if (
                x1 <= x <= x2
                and abs(y - py) < 24
            ):

                return py

        return None

    # ========================================================
    # Hazard
    # ========================================================

    def hazard_at(
        self,
        x,
        y,
    ):

        phase = (
            self.regime.global_phase
        )

        for (
            x1,
            x2,
            hy,
        ) in self.hazards:

            shift = (
                22.0
                * math.sin(
                    self.time
                    * 0.031
                    + x1
                )
                * phase
            )

            if (
                x1 + shift
                <= x
                <= x2 + shift
                and y <= hy + 20
            ):

                return True

        return False

    # ========================================================
    # Wormhole
    # ========================================================

    def wormhole(
        self,
        x,
        y,
    ):

        s = (
            self.regime.wormhole_strength()
        )

        if s <= 0.2:

            return False

        return (
            abs(x) < 28
            and abs(y - 130) < 42
        )

    # ========================================================
    # Local context
    # ========================================================

    def local_context(
        self,
        x,
        y,
        agent_id,
    ):

        gx = (
            self.regime.local_gravity(
                x,
                y,
            )
        )

        phase = (
            self.regime.local_phase(
                x,
                y,
                agent_id,
            )
        )

        return np.asarray(
            [
                gx
                / max(
                    0.1,
                    BASE_GRAVITY,
                ),

                phase,

                self.regime.time_scale()
                / 2.0,

                self.regime.wormhole_strength(),

                self.regime.past_leak_strength(),

                self.regime.mirror_strength(),

                float(
                    self.regime.gravity_sign
                ),
            ],
            dtype=np.float32,
        )

    def all_platforms(self):

        return (
            self.effective_platforms()
            + self.effective_ceiling_platforms()
        )

    # ========================================================
    # PHYSICS
    # ========================================================

    def step(
        self,
        agent,
        action,
    ):

        dt = (
            self.regime.time_scale()
        )

        x = agent.x
        y = agent.y

        vx = agent.vx
        vy = agent.vy

        grounded = agent.grounded

        local_phase = (
            self.regime.local_phase(
                x,
                y,
                agent.id,
            )
        )

        gravity = (
            self.regime.local_gravity(
                x,
                y,
            )
        )

        mirror = (
            self.regime.mirror_strength()
        )

        # ----------------------------------------------------
        # Action
        # ----------------------------------------------------

        if action == ACTION_LEFT:

            vx -= (
                GROUND_ACCEL
                if grounded
                else AIR_ACCEL
            ) * dt

            agent.heading = -1

        elif action == ACTION_RIGHT:

            vx += (
                GROUND_ACCEL
                if grounded
                else AIR_ACCEL
            ) * dt

            agent.heading = 1

        elif action == ACTION_JUMP:

            if grounded:

                if gravity > 0:

                    vy = (
                        JUMP_POWER
                        * (
                            0.65
                            + 0.55
                            * local_phase
                        )
                    )

                else:

                    vy = (
                        -JUMP_POWER
                        * (
                            0.65
                            + 0.55
                            * local_phase
                        )
                    )

                grounded = False

                agent.jumps = 1

            elif agent.jumps < 2:

                if gravity > 0:

                    vy = (
                        JUMP_POWER
                        * 0.82
                    )

                else:

                    vy = (
                        -JUMP_POWER
                        * 0.82
                    )

                agent.jumps += 1

        elif action == ACTION_BRAKE:

            vx *= (
                0.22
                + 0.22
                * local_phase
            )

        elif action == ACTION_WAIT:

            vx *= 0.90

        # ----------------------------------------------------
        # Mirror
        # ----------------------------------------------------

        if (
            mirror > 0
            and -170 < x < 170
        ):

            if self.time % 14 == 0:

                vx *= -1.0

                self.emit_event(
                    "BOUNDARY",
                    x,
                    y,
                    0.15,
                    agent.id,
                )

        vx = float(
            np.clip(
                vx,
                -MAX_SPEED,
                MAX_SPEED,
            )
        )

        vx *= (
            GROUND_FRICTION
            if grounded
            else AIR_FRICTION
        )

        # ----------------------------------------------------
        # Gravity
        # ----------------------------------------------------

        if not grounded:

            vy -= (
                gravity * dt
            )

            y += (
                vy * dt
            )

        x += (
            vx * dt
        )

        # ----------------------------------------------------
        # World bounds
        # ----------------------------------------------------

        if (
            x
            < WORLD_LEFT + 12
        ):

            x = (
                WORLD_LEFT + 12
            )

            vx *= -0.35

            self.emit_event(
                "BOUNDARY",
                x,
                y,
                0.5,
                agent.id,
            )

        if (
            x
            > WORLD_RIGHT - 12
        ):

            x = (
                WORLD_RIGHT - 12
            )

            vx *= -0.35

            self.emit_event(
                "BOUNDARY",
                x,
                y,
                0.5,
                agent.id,
            )

        # ----------------------------------------------------
        # Wormhole
        # ----------------------------------------------------

        if self.wormhole(
            x,
            y,
        ):

            old_x = x

            x = (
                430.0
                if x < 0
                else -430.0
            )

            y = (
                50.0
                + 100.0
                * local_phase
            )

            vx *= -0.5
            vy *= 0.35

            grounded = False

            self.emit_event(
                "WORMHOLE",
                old_x,
                y,
                1.0,
                agent.id,
            )

            self.emit_event(
                "TELEPORT",
                x,
                y,
                1.0,
                agent.id,
            )

        # ====================================================
        # DOWNWARD GRAVITY
        # ====================================================

        if gravity > 0:

            py = self.platform_y(
                x,
                y,
            )

            if (
                py is not None
                and vy <= 0
                and y <= py + 23
            ):

                y = (
                    py + 23
                )

                vy = 0.0

                grounded = True

                agent.jumps = 0

                self.emit_event(
                    "GROUND",
                    x,
                    y,
                    0.25,
                    agent.id,
                )

            else:

                grounded = False

        # ====================================================
        # UPWARD GRAVITY
        # ====================================================

        else:

            ceiling = (
                self.ceiling_y(
                    x,
                    y,
                )
            )

            if (
                ceiling is not None
                and vy >= 0
                and y >= ceiling - 23
            ):

                y = (
                    ceiling - 23
                )

                vy = (
                    -abs(vy)
                    * UPWARD_GRAVITY_BOUNCE
                )

                vx *= (
                    IMPACT_HORIZONTAL_BOUNCE
                )

                grounded = True

                agent.jumps = 0

                self.impact_marks.append(
                    ImpactMark(
                        x,
                        ceiling - 23,
                        -1,
                    )
                )

                agent.last_impact = (
                    self.time
                )

                self.emit_event(
                    "IMPACT",
                    x,
                    ceiling - 23,
                    clamp(
                        abs(vy)
                        / 13.0,
                        0.2,
                        1.0,
                    ),
                    agent.id,
                )

            else:

                grounded = False

        # ----------------------------------------------------
        # Bottom
        # ----------------------------------------------------

        if (
            y
            < WORLD_BOTTOM + 5
        ):

            x = (
                -480
                + agent.id * 24
            )

            y = -195

            vx = 0.0
            vy = 0.0

            grounded = True

            agent.jumps = 0

            self.emit_event(
                "BOUNDARY",
                x,
                y,
                0.7,
                agent.id,
            )

        # ----------------------------------------------------
        # Top
        # ----------------------------------------------------

        if (
            y
            > WORLD_TOP + 15
        ):

            y = (
                WORLD_TOP + 15
            )

            vy = min(
                vy,
                -2.0,
            )

            self.emit_event(
                "BOUNDARY",
                x,
                y,
                0.7,
                agent.id,
            )

        # ----------------------------------------------------
        # Hazard
        # ----------------------------------------------------

        if self.hazard_at(
            x,
            y,
        ):

            self.emit_event(
                "HAZARD",
                x,
                y,
                1.0,
                agent.id,
            )

            x = (
                -480
                + agent.id * 24
            )

            y = -195

            vx = 0.0
            vy = 0.0

            grounded = True

            agent.jumps = 0

        # ----------------------------------------------------
        # Write back
        # ----------------------------------------------------

        agent.x = x
        agent.y = y

        agent.vx = vx
        agent.vy = vy

        agent.grounded = grounded

    # ========================================================
    # DRAW
    # ========================================================

    def draw(self):

        d = self.drawer

        d.clear()

        # ----------------------------------------------------
        # Border
        # ----------------------------------------------------

        for args in [
            (
                WORLD_LEFT,
                WORLD_BOTTOM,
                WORLD_RIGHT,
                WORLD_BOTTOM,
            ),
            (
                WORLD_RIGHT,
                WORLD_BOTTOM,
                WORLD_RIGHT,
                WORLD_TOP,
            ),
            (
                WORLD_RIGHT,
                WORLD_TOP,
                WORLD_LEFT,
                WORLD_TOP,
            ),
            (
                WORLD_LEFT,
                WORLD_TOP,
                WORLD_LEFT,
                WORLD_BOTTOM,
            ),
        ]:

            self.draw_line(
                *args,
                3,
                "#3b3d4d",
            )

        # ----------------------------------------------------
        # Gravity color
        # ----------------------------------------------------

        if (
            self.regime.gravity_is_up()
        ):

            gravity_color = (
                "#ff4d68"
            )

        else:

            gravity_color = (
                "#4da6ff"
            )

        # ----------------------------------------------------
        # Platforms
        # ----------------------------------------------------

        phase = (
            self.regime.topology_phase
        )

        structure_color = (
            "#5f9ea0"
            if phase < 0.5
            else "#ad7be8"
        )

        for (
            x1,
            x2,
            y,
        ) in self.effective_platforms():

            self.draw_line(
                x1,
                y,
                x2,
                y,
                6,
                structure_color,
            )

        # ----------------------------------------------------
        # Ceiling
        # ----------------------------------------------------

        for (
            x1,
            x2,
            y,
        ) in self.effective_ceiling_platforms():

            self.draw_line(
                x1,
                y,
                x2,
                y,
                8,
                gravity_color,
            )

        # ----------------------------------------------------
        # Hazards
        # ----------------------------------------------------

        for (
            x1,
            x2,
            y,
        ) in self.hazards:

            shift = (
                22.0
                * math.sin(
                    self.time
                    * 0.031
                    + x1
                )
                * self.regime.global_phase
            )

            self.draw_line(
                x1 + shift,
                y,
                x2 + shift,
                y,
                10,
                "#ff4055",
            )

        # ----------------------------------------------------
        # Moving object
        # ----------------------------------------------------

        d.goto(
            self.moving_x,
            self.moving_y,
        )

        d.dot(
            18,
            "#f0a63a",
        )

        # ----------------------------------------------------
        # Ghost
        # ----------------------------------------------------

        if (
            self.regime.past_leak_strength()
            > 0.35
        ):

            d.goto(
                self.ghost_x,
                self.ghost_y,
            )

            d.dot(
                14,
                "#bb88ff",
            )

        # ----------------------------------------------------
        # Wormhole
        # ----------------------------------------------------

        if (
            self.regime.wormhole_strength()
            > 0.2
        ):

            d.goto(
                0,
                130,
            )

            d.dot(
                34,
                "#6b55ff",
            )

            d.goto(
                0,
                130,
            )

            d.dot(
                13,
                "#f4eaff",
            )

        # ----------------------------------------------------
        # Warning
        # ----------------------------------------------------

        if self.gravity_warning:

            self.gravity_warning.draw(
                d
            )

        # ----------------------------------------------------
        # Impact
        # ----------------------------------------------------

        for mark in (
            self.impact_marks
        ):

            mark.draw(d)

        # ----------------------------------------------------
        # Gravity indicator
        # ----------------------------------------------------

        indicator_x = -430
        indicator_y = 215

        d.color(
            gravity_color
        )

        d.pensize(5)

        if (
            self.regime.gravity_is_up()
        ):

            d.penup()

            d.goto(
                indicator_x,
                indicator_y - 35,
            )

            d.pendown()

            d.goto(
                indicator_x,
                indicator_y + 35,
            )

            d.goto(
                indicator_x - 13,
                indicator_y + 18,
            )

            d.penup()

            d.goto(
                indicator_x,
                indicator_y + 35,
            )

            d.pendown()

            d.goto(
                indicator_x + 13,
                indicator_y + 18,
            )

        else:

            d.penup()

            d.goto(
                indicator_x,
                indicator_y + 35,
            )

            d.pendown()

            d.goto(
                indicator_x,
                indicator_y - 35,
            )

            d.goto(
                indicator_x - 13,
                indicator_y - 18,
            )

            d.penup()

            d.goto(
                indicator_x,
                indicator_y - 35,
            )

            d.pendown()

            d.goto(
                indicator_x + 13,
                indicator_y - 18,
            )

        d.penup()

        # ----------------------------------------------------
        # World text
        # ----------------------------------------------------

        d.color(
            gravity_color
        )

        d.goto(
            -500,
            250,
        )

        direction_text = (
            "GRAVITY ↑ UP"
            if self.regime.gravity_is_up()
            else "GRAVITY ↓ DOWN"
        )

        d.write(
            direction_text,
            font=(
                "Arial",
                13,
                "bold",
            ),
        )

        d.goto(
            -500,
            230,
        )

        d.write(
            f"FLIP IN "
            f"{self.regime.frames_until_gravity_flip()}",
            font=(
                "Arial",
                10,
                "normal",
            ),
        )


# ============================================================
# Visual Field
# ============================================================


class VisualField:

    def __init__(
        self,
        world,
    ):

        self.world = world

    def to_grid(
        self,
        x,
        y,
    ):

        gx = int(
            (
                x - WORLD_LEFT
            )
            / (
                WORLD_RIGHT
                - WORLD_LEFT
            )
            * (
                OBS_W - 1
            )
        )

        gy = int(
            (
                y - WORLD_BOTTOM
            )
            / (
                WORLD_TOP
                - WORLD_BOTTOM
            )
            * (
                OBS_H - 1
            )
        )

        return (
            clamp(
                gx,
                0,
                OBS_W - 1,
            ),
            clamp(
                gy,
                0,
                OBS_H - 1,
            ),
        )

    def seg(
        self,
        channel,
        x1,
        y1,
        x2,
        y2,
        value,
    ):

        gx1, gy1 = (
            self.to_grid(
                x1,
                y1,
            )
        )

        gx2, gy2 = (
            self.to_grid(
                x2,
                y2,
            )
        )

        n = max(
            abs(gx2 - gx1),
            abs(gy2 - gy1),
            1,
        )

        for i in range(
            n + 1
        ):

            t = i / n

            gx = int(
                gx1
                + (
                    gx2 - gx1
                )
                * t
            )

            gy = int(
                gy1
                + (
                    gy2 - gy1
                )
                * t
            )

            for dx in (
                -1,
                0,
                1,
            ):

                for dy in (
                    -1,
                    0,
                    1,
                ):

                    xx = gx + dx
                    yy = gy + dy

                    if (
                        0 <= xx < OBS_W
                        and 0 <= yy < OBS_H
                    ):

                        channel[
                            yy,
                            xx
                        ] = max(
                            channel[
                                yy,
                                xx
                            ],
                            value,
                        )

    def dot(
        self,
        channel,
        x,
        y,
        value,
    ):

        gx, gy = (
            self.to_grid(
                x,
                y,
            )
        )

        for dx in (
            -1,
            0,
            1,
        ):

            for dy in (
                -1,
                0,
                1,
            ):

                xx = gx + dx
                yy = gy + dy

                if (
                    0 <= xx < OBS_W
                    and 0 <= yy < OBS_H
                ):

                    channel[
                        yy,
                        xx
                    ] = max(
                        channel[
                            yy,
                            xx
                        ],
                        value,
                    )

    def capture(
        self,
        agents,
        viewer_id=None,
    ):

        img = np.zeros(
            (
                OBS_H,
                OBS_W,
                CHANNELS,
            ),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Structures
        # ----------------------------------------------------

        for (
            x1,
            x2,
            y,
        ) in self.world.effective_platforms():

            self.seg(
                img[:, :, 0],
                x1,
                y,
                x2,
                y,
                0.75,
            )

        # ----------------------------------------------------
        # Ceiling
        # ----------------------------------------------------

        for (
            x1,
            x2,
            y,
        ) in (
            self.world
            .effective_ceiling_platforms()
        ):

            self.seg(
                img[:, :, 0],
                x1,
                y,
                x2,
                y,
                0.90,
            )

        # ----------------------------------------------------
        # Hazards
        # ----------------------------------------------------

        for (
            x1,
            x2,
            y,
        ) in self.world.hazards:

            shift = (
                22.0
                * math.sin(
                    self.world.time
                    * 0.031
                    + x1
                )
                * self.world.regime.global_phase
            )

            self.seg(
                img[:, :, 1],
                x1 + shift,
                y,
                x2 + shift,
                y,
                1.0,
            )

        self.dot(
            img[:, :, 1],
            self.world.moving_x,
            self.world.moving_y,
            0.9,
        )

        # ----------------------------------------------------
        # Ghost
        # ----------------------------------------------------

        if (
            self.world.regime.past_leak_strength()
            > 0.35
        ):

            self.dot(
                img[:, :, 1],
                self.world.ghost_x,
                self.world.ghost_y,
                0.7,
            )

        # ----------------------------------------------------
        # Wormhole
        # ----------------------------------------------------

        if (
            self.world.regime.wormhole_strength()
            > 0.2
        ):

            self.dot(
                img[:, :, 1],
                0,
                130,
                1.0,
            )

        # ----------------------------------------------------
        # Agents
        # ----------------------------------------------------

        for a in agents:

            if (
                viewer_id is not None
                and a.id
                == viewer_id
            ):

                self.dot(
                    img[:, :, 2],
                    a.x,
                    a.y,
                    1.0,
                )

            else:

                self.dot(
                    img[:, :, 2],
                    a.x,
                    a.y,
                    0.75,
                )

        # ----------------------------------------------------
        # Gravity information
        # ----------------------------------------------------

        gravity_value = (
            0.95
            if self.world.regime.gravity_is_up()
            else 0.25
        )

        for i in range(
            0,
            OBS_W,
            9,
        ):

            x = (
                WORLD_LEFT
                + (
                    WORLD_RIGHT
                    - WORLD_LEFT
                )
                * i
                / (
                    OBS_W - 1
                )
            )

            y = 210

            gx, gy = (
                self.to_grid(
                    x,
                    y,
                )
            )

            img[
                gy,
                gx,
                3
            ] = gravity_value

        # ----------------------------------------------------
        # Temporal texture
        # ----------------------------------------------------

        for i in range(
            0,
            OBS_W,
            7,
        ):

            x = (
                WORLD_LEFT
                + (
                    WORLD_RIGHT
                    - WORLD_LEFT
                )
                * i
                / (
                    OBS_W - 1
                )
            )

            y = (
                195
                + 12
                * math.sin(
                    self.world.time
                    * 0.02
                    + i * 0.7
                )
            )

            gx, gy = (
                self.to_grid(
                    x,
                    y,
                )
            )

            img[
                gy,
                gx,
                3
            ] = max(
                img[
                    gy,
                    gx,
                    3
                ],
                0.15
                + 0.45
                * self.world.regime.global_phase,
            )

        return img.reshape(-1)


# ============================================================
# Agent
# ============================================================


class Agent:

    def __init__(
        self,
        agent_id,
        color,
        world,
        model,
        vision,
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

        self.last_impact = -999

        # ----------------------------------------------------
        # Individual personality
        # ----------------------------------------------------

        self.exploration = (
            EXPLORATION_RATE
            * random.uniform(
                0.70,
                1.35,
            )
        )

        self.risk_tolerance = (
            random.uniform(
                0.65,
                1.25,
            )
        )

        self.curiosity_bias = (
            random.uniform(
                0.80,
                1.30,
            )
        )

        # ----------------------------------------------------
        # Event cursor
        # ----------------------------------------------------

        self.event_cursor = 0

        self.reset()

    def reset(self):

        self.x = (
            -480
            + self.id * 24
        )

        self.y = -195

        self.vx = 0.0
        self.vy = 0.0

        self.grounded = True

        self.jumps = 0

        self.heading = 1

        self.last_action = (
            ACTION_NONE
        )

        self.last_error = 0.0

        self.last_prediction_error = (
            0.0
        )

        self.last_surprise = 0.0

        self.last_counterfactuals = []

        self.steps = 0

        self.last_impact = -999

        self.event_cursor = (
            self.world.event_serial
        )

        self.turtle.goto(
            self.x,
            self.y,
        )

    # ========================================================
    # Body
    # ========================================================

    def body_state(self):

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

                float(
                    self.world.time
                    - self.last_impact
                    < 10
                ),
            ],
            dtype=np.float32,
        )

    # ========================================================
    # Context
    # ========================================================

    def local_context(self):

        return (
            self.world.local_context(
                self.x,
                self.y,
                self.id,
            )
        )

    # ========================================================
    # Events
    # ========================================================

    def recent_events(self):

        events = (
            self.world.events_since(
                self.event_cursor
            )
        )

        self.event_cursor = (
            self.world.event_serial
        )

        # Agent自身に関係するイベント
        # + グローバルイベント
        filtered = []

        for event in events:

            if (
                event.source
                in (
                    -1,
                    self.id,
                )
            ):

                filtered.append(
                    event
                )

        return filtered

    # ========================================================
    # Step
    # ========================================================

    def step(
        self,
        agents,
    ):

        # ----------------------------------------------------
        # Before observation
        # ----------------------------------------------------

        image_before = (
            self.vision.capture(
                agents,
                self.id,
            )
        )

        body_before = (
            self.body_state()
        )

        ctx_before = (
            self.local_context()
        )

        # ----------------------------------------------------
        # Counterfactual preview
        # ----------------------------------------------------

        cf_values = []

        for action in ACTIONS:

            cf = (
                self.model.counterfactual(
                    image_before,
                    body_before,
                    ctx_before,
                    action,
                )
            )

            cf_values.append(
                cf
            )

        self.last_counterfactuals = (
            cf_values
        )

        # ----------------------------------------------------
        # Action selection
        # ----------------------------------------------------

        if (
            random.random()
            < self.exploration
        ):

            action = random.choice(
                [
                    ACTION_LEFT,
                    ACTION_RIGHT,
                    ACTION_JUMP,
                    ACTION_WAIT,
                    ACTION_BRAKE,
                ]
            )

        else:

            action = (
                self.model.select_action(
                    image_before,
                    body_before,
                    ctx_before,
                    self.last_action,
                )
            )

        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        self.world.step(
            self,
            action,
        )

        # ----------------------------------------------------
        # World draw
        # ----------------------------------------------------

        self.world.draw()

        # ----------------------------------------------------
        # After
        # ----------------------------------------------------

        image_after = (
            self.vision.capture(
                agents,
                self.id,
            )
        )

        body_after = (
            self.body_state()
        )

        ctx_after = (
            self.local_context()
        )

        # ----------------------------------------------------
        # Events
        # ----------------------------------------------------

        events = (
            self.recent_events()
        )

        # ----------------------------------------------------
        # Phase
        # ----------------------------------------------------

        phase_bucket = int(
            self.world.regime.local_phase(
                self.x,
                self.y,
                self.id,
            )
            * 8.0
        )

        phase_bucket = int(
            clamp(
                phase_bucket,
                0,
                7,
            )
        )

        # ----------------------------------------------------
        # Learning
        # ----------------------------------------------------

        result = self.model.learn(
            self.id,
            image_before,
            body_before,
            action,
            image_after,
            body_after,
            ctx_before,
            ctx_after,
            phase_bucket,
            events,
        )

        # ----------------------------------------------------
        # Stats
        # ----------------------------------------------------

        self.last_action = action

        self.last_error = (
            result["error"]
        )

        self.last_prediction_error = (
            result[
                "prediction_error"
            ]
        )

        self.last_surprise = (
            result["surprise"]
        )

        self.steps += 1

        self.turtle.goto(
            self.x,
            self.y,
        )


# ============================================================
# UI
# ============================================================


ui = turtle.Turtle(
    visible=False
)

ui.penup()
ui.speed(0)


mem = turtle.Turtle(
    visible=False
)

mem.penup()
mem.speed(0)


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
    color="#dddddd",
):

    ui.goto(
        x,
        y,
    )

    ui.color(
        color
    )

    ui.write(
        text,
        font=(
            "Arial",
            size,
            "normal",
        ),
    )


# ============================================================
# Model UI
# ============================================================


def draw_model(
    model,
    agents,
    episode,
    step,
    world,
):

    ui.clear()
    mem.clear()
    graph.clear()

    s = (
        model.statistics()
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    write(
        -540,
        325,
        "STRANGE SELF-ORGANIZING WORLD v4",
        15,
        "#ffffff",
    )

    write(
        -540,
        302,
        f"EPISODE "
        f"{episode + 1}/"
        f"{MAX_EPISODES}"
        f"   STEP "
        f"{step}/"
        f"{STEPS_PER_EPISODE}",
        10,
        "#55ffbb",
    )

    # --------------------------------------------------------
    # Gravity
    # --------------------------------------------------------

    gravity_color = (
        "#ff4055"
        if world.regime.gravity_is_up()
        else "#4da6ff"
    )

    gravity_text = (
        "↑ UPWARD GRAVITY"
        if world.regime.gravity_is_up()
        else "↓ DOWNWARD GRAVITY"
    )

    write(
        -540,
        278,
        gravity_text,
        14,
        gravity_color,
    )

    write(
        -540,
        258,
        f"Gravity flip in "
        f"{world.regime.frames_until_gravity_flip()}",
        10,
        "#ffd84d",
    )

    if (
        world.regime.gravity_flip_imminent()
    ):

        write(
            -540,
            240,
            "⚠ GRAVITY REVERSAL IMMINENT",
            11,
            "#ffd84d",
        )

    # --------------------------------------------------------
    # Main stats
    # --------------------------------------------------------

    write(
        560,
        265,
        f"VisualCells : {s['visual']}",
        9,
    )

    write(
        560,
        247,
        f"PlaceStates : {s['places']}",
        9,
    )

    write(
        560,
        229,
        f"Transforms  : {s['transforms']}",
        9,
    )

    write(
        560,
        211,
        f"Meta        : {s['meta']}",
        9,
    )

    write(
        560,
        193,
        f"Events      : {s['events']}",
        9,
        "#ff99dd",
    )

    write(
        560,
        175,
        f"Err         : "
        f"{s['mean_error']:.3f}",
        9,
        "#ffba45",
    )

    write(
        560,
        157,
        f"PredErr     : "
        f"{s['prediction_error']:.3f}",
        9,
        "#ff8866",
    )

    write(
        560,
        139,
        f"PredSuccess : "
        f"{s['prediction_success']:.3f}",
        9,
        "#66ffbb",
    )

    write(
        560,
        121,
        f"Surprise    : "
        f"{s['surprise']:.3f}",
        9,
        "#ffaaee",
    )

    write(
        560,
        103,
        f"Stability   : "
        f"{s['stability']:.3f}",
        9,
        "#88ccff",
    )

    write(
        560,
        85,
        f"Splits      : "
        f"{s['splits']}",
        9,
        "#ff77aa",
    )

    # --------------------------------------------------------
    # Place graph
    # --------------------------------------------------------

    write(
        560,
        60,
        "PLACE STATES",
        9,
        "#ffffff",
    )

    base_x = 560
    base_y = 25

    maxp = min(
        42,
        len(model.places),
    )

    coords = {}

    for i in range(
        maxp
    ):

        p = model.places[i]

        x = (
            base_x
            + (i % 7)
            * 42
        )

        y = (
            base_y
            - (i // 7)
            * 30
        )

        coords[i] = (
            x,
            y,
        )

        r = min(
            15,
            5
            + int(
                math.log1p(
                    p.visits
                )
            ),
        )

        mem.goto(
            x,
            y,
        )

        mem.dot(
            r,
            "#5ea8ff",
        )

    # --------------------------------------------------------
    # Place connections
    # --------------------------------------------------------

    for i in range(
        maxp
    ):

        p = model.places[i]

        x1, y1 = coords[i]

        for (
            target,
            count,
        ) in list(
            p.transitions.items()
        )[-8:]:

            if (
                target
                not in coords
            ):

                continue

            x2, y2 = coords[target]

            mem.goto(
                x1,
                y1,
            )

            mem.pendown()

            mem.goto(
                x2,
                y2,
            )

            mem.penup()

    # --------------------------------------------------------
    # Transform curiosity
    # --------------------------------------------------------

    tx = 560
    ty = -150

    write(
        tx,
        ty + 45,
        "PREDICTIVE TRANSFORMATIONS",
        9,
        "#ffffff",
    )

    ranked = sorted(
        model.transforms,
        key=lambda t:
            (
                t.prediction_error
                + t.curiosity()
            ),
        reverse=True,
    )[:8]

    for i, t in enumerate(
        ranked
    ):

        y = (
            ty
            + 18
            - i * 18
        )

        action_text = "?"

        if t.action_counts:

            action_id = max(
                t.action_counts,
                key=t.action_counts.get,
            )

            action_text = (
                ACTION_NAMES.get(
                    action_id,
                    "?",
                )
            )

        write(
            tx,
            y,
            f"T{t.id:03d} "
            f"{action_text:<5} "
            f"v={t.visits:<3d} "
            f"p={t.prediction_success:.2f} "
            f"e={t.prediction_error:.2f}",
            8,
            "#ffdd66",
        )

    # --------------------------------------------------------
    # Event memory
    # --------------------------------------------------------

    ex = -540
    ey = -245

    write(
        ex,
        ey,
        "EVENT MEMORY",
        9,
        "#ffffff",
    )

    recent_event_cells = sorted(
        model.events,
        key=lambda e:
            e.last_time,
        reverse=True,
    )[:7]

    for i, e in enumerate(
        recent_event_cells
    ):

        write(
            ex,
            ey - 18 - i * 16,
            f"E{e.id:03d} "
            f"{e.event_name:<13} "
            f"v={e.visits:<3d} "
            f"c={e.confidence:.2f}",
            8,
            "#ff99dd",
        )

    # --------------------------------------------------------
    # Agent status
    # --------------------------------------------------------

    y = 225

    for a in agents:

        impact_text = ""

        if (
            world.time
            - a.last_impact
            < 12
        ):

            impact_text = (
                " IMPACT!"
            )

        write(
            -540,
            y,
            f"A{a.id} "
            f"{ACTION_NAMES[a.last_action]:<5} "
            f"x={a.x:+.0f} "
            f"y={a.y:+.0f} "
            f"vx={a.vx:+.1f} "
            f"vy={a.vy:+.1f} "
            f"err={a.last_error:.2f} "
            f"pred={a.last_prediction_error:.2f}"
            f"{impact_text}",
            8,
            a.color,
        )

        y -= 19

    # --------------------------------------------------------
    # Counterfactual panel
    # --------------------------------------------------------

    if agents:

        a = agents[0]

        write(
            170,
            325,
            "COUNTERFACTUAL FUTURES / A0",
            10,
            "#ffffff",
        )

        cf_y = 305

        for cf in (
            a.last_counterfactuals
        ):

            action_name = (
                ACTION_NAMES[
                    cf["action"]
                ]
            )

            value = cf[
                "value"
            ]

            risk = cf[
                "risk"
            ]

            uncertainty = cf[
                "uncertainty"
            ]

            color = (
                "#55ffbb"
                if value > 0
                else "#ff6677"
            )

            write(
                170,
                cf_y,
                f"{action_name:<5} "
                f"value={value:+.2f} "
                f"risk={risk:.2f} "
                f"unc={uncertainty:.2f}",
                8,
                color,
            )

            cf_y -= 16


# ============================================================
# Main
# ============================================================


screen = turtle.Screen()

screen.setup(
    SCREEN_W,
    SCREEN_H,
)

screen.bgcolor(
    "#080910"
)

screen.title(
    "Strange Place / Predictive Self-Organizing World v4"
)

screen.tracer(
    False
)


# ============================================================
# Create world
# ============================================================


world = World()

vision = VisualField(
    world
)

model = WorldModel()


colors = [
    "#00eaff",
    "#57ff9b",
    "#ff9f43",
]


agents = [
    Agent(
        i,
        colors[i],
        world,
        model,
        vision,
    )
    for i in range(
        NUM_AGENTS
    )
]


episode = 0
step = 0

finished = False


# ============================================================
# Episode reset
# ============================================================


def reset_episode():

    global step

    step = 0

    world.reset()

    for a in agents:

        a.reset()


# ============================================================
# Episode sleep
# ============================================================


def sleep_phase():

    global episode
    global finished

    # --------------------------------------------------------
    # Replay
    # --------------------------------------------------------

    model.replay(
        520
    )

    # --------------------------------------------------------
    # Decay
    # --------------------------------------------------------

    model.decay()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    draw_model(
        model,
        agents,
        episode,
        step,
        world,
    )

    screen.update()

    episode += 1

    if (
        episode
        >= MAX_EPISODES
    ):

        finished = True

        write(
            -180,
            -335,
            "SIMULATION FINISHED",
            17,
            "#ffffff",
        )

        screen.update()

        return

    screen.ontimer(
        start_episode,
        800,
    )


# ============================================================
# Run step
# ============================================================


def run_step():

    global step

    if finished:

        return

    # --------------------------------------------------------
    # World
    # --------------------------------------------------------

    world.update()

    world.draw()

    # --------------------------------------------------------
    # Agents
    # --------------------------------------------------------

    for agent in agents:

        agent.step(
            agents
        )

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    draw_model(
        model,
        agents,
        episode,
        step,
        world,
    )

    screen.update()

    step += 1

    # --------------------------------------------------------
    # Continue
    # --------------------------------------------------------

    if (
        step
        < STEPS_PER_EPISODE
    ):

        screen.ontimer(
            run_step,
            14,
        )

    else:

        screen.ontimer(
            sleep_phase,
            180,
        )


# ============================================================
# Start episode
# ============================================================


def start_episode():

    if finished:

        return

    reset_episode()

    run_step()


# ============================================================
# Start
# ============================================================


reset_episode()

world.draw()

draw_model(
    model,
    agents,
    episode,
    step,
    world,
)

screen.update()

screen.ontimer(
    start_episode,
    600,
)

turtle.done()
