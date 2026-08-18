import turtle
import random
import math
import numpy as np
from collections import deque

# ============================================================
# STRANGE PLACE / SELF-ORGANIZING WORLD MODEL v2
# ============================================================
#
# 中心思想
# ------------------------------------------------------------
# 「同じ場所とは何か？」を固定しない。
#
# 同じ座標でも、
#   - 世界位相
#   - 局所重力
#   - 時間位相
#   - 過去の漏出
#   - Agentごとの位相
#   - 地形状態
# が違えば、別の PlaceState になる。
#
# AIへ semantic label は渡さない。
# AIが受け取るのは FULL IMAGE + BODY STATE だけ。
#
# 学習の中心:
#
#   観測
#     ↓
#   Spatial field
#     ↓
#   PlaceState
#     ↓
#   Transformation
#     ↓
#   Transformation graph
#     ↓
#   Meta-transformation
#     ↓
#   Place split / merge
#
# 「未来を当てる予測器」ではなく、
# 「世界がどう変化したかの構造」を自己形成する。
# ============================================================

# ============================================================
# Configuration
# ============================================================

SCREEN_W = 1240
SCREEN_H = 820

WORLD_LEFT = -560
WORLD_RIGHT = 560
WORLD_BOTTOM = -300
WORLD_TOP = 290

NUM_AGENTS = 3
STEPS_PER_EPISODE = 340
MAX_EPISODES = 14

OBS_W = 72
OBS_H = 44
CHANNELS = 4
PATCH = 6

VISUAL_SIM_THRESHOLD = 0.74
PLACE_SIM_THRESHOLD = 0.80
TRANSFORM_SIM_THRESHOLD = 0.70
META_SIM_THRESHOLD = 0.72

MEMORY_WINDOW = 140
TRANSFORM_HISTORY = 600

# Physics
BASE_GRAVITY = 0.62
GROUND_ACCEL = 1.05
AIR_ACCEL = 0.58
GROUND_FRICTION = 0.90
AIR_FRICTION = 0.985
JUMP_POWER = 11.2
MAX_SPEED = 11.0

# Strange-world event scales
PHASE_PERIOD = 420.0
TIME_WARP_PERIOD = 300.0
TOPOLOGY_PERIOD = 510.0
PAST_LEAK_PERIOD = 360.0

# Actions
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
    return float(np.dot(a, b) / (na * nb))


def l1_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(a - b)))


def soft_distance(a, b, scale=4.0):
    return math.exp(-scale * l1_distance(a, b))

# ============================================================
# Strange World Regime
# ============================================================


class WorldRegime:
    """
    世界そのものの局所的な状態。

    semantic な名前は持つが、それは WORLD 側の内部実装であり、
    AI へ直接渡さない。
    """

    def __init__(self):
        self.time = 0
        self.global_phase = 0.0
        self.topology_phase = 0.0
        self.time_phase = 0.0
        self.past_phase = 0.0

    def update(self):
        self.time += 1
        self.global_phase = (
            0.5 + 0.5 * math.sin(self.time * 2.0 * math.pi / PHASE_PERIOD)
        )
        self.topology_phase = (
            0.5 + 0.5 * math.sin(self.time * 2.0 * math.pi / TOPOLOGY_PERIOD + 1.4)
        )
        self.time_phase = (
            0.5 + 0.5 * math.sin(self.time * 2.0 * math.pi / TIME_WARP_PERIOD - 0.8)
        )
        self.past_phase = (
            0.5 + 0.5 * math.sin(self.time * 2.0 * math.pi / PAST_LEAK_PERIOD + 2.2)
        )

    def time_scale(self):
        # 極端に速くも遅くもなる。
        return 0.30 + 1.65 * self.time_phase

    def local_gravity(self, x, y):
        spatial = math.sin(x * 0.012 + self.time * 0.018) * 0.5 + 0.5
        sign_field = 1.0 if spatial > (0.38 + 0.22 * self.global_phase) else -1.0
        magnitude = BASE_GRAVITY * (0.45 + 1.15 * abs(math.sin(y * 0.01 + self.time * 0.013)))
        return sign_field * magnitude

    def local_phase(self, x, y, agent_id):
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
        return 1.0 if self.topology_phase > 0.82 else 0.0

    def wormhole_strength(self):
        return clamp((self.global_phase - 0.76) / 0.24, 0.0, 1.0)

    def past_leak_strength(self):
        return clamp((self.past_phase - 0.72) / 0.28, 0.0, 1.0)

# ============================================================
# Visual memory
# ============================================================


class VisualCell:
    def __init__(self, cell_id, feature, gx, gy):
        self.id = cell_id
        self.feature = np.asarray(feature, dtype=np.float32)
        self.gx = gx
        self.gy = gy
        self.visits = 1
        self.activation = 1.0
        self.energy = 1.0
        self.phase_counts = {}

    def similarity(self, feature):
        return soft_distance(self.feature, feature, 4.3)

    def update(self, feature, phase_bucket):
        feature = np.asarray(feature, dtype=np.float32)
        self.feature = 0.91 * self.feature + 0.09 * feature
        self.visits += 1
        self.activation = min(2.0, self.activation + 0.04)
        self.energy = min(2.0, self.energy + 0.02)
        self.phase_counts[phase_bucket] = self.phase_counts.get(phase_bucket, 0) + 1

    def decay(self):
        self.activation *= 0.994
        self.energy *= 0.999


class SpatialRepresentation:
    def __init__(self):
        self.cells = {}
        self.next_id = 0

    def patches(self, image):
        image = np.asarray(image, dtype=np.float32).reshape(OBS_H, OBS_W, CHANNELS)
        out = []
        for y in range(0, OBS_H, PATCH):
            for x in range(0, OBS_W, PATCH):
                patch = image[y:y + PATCH, x:x + PATCH]
                feat = patch.mean(axis=(0, 1))
                out.append((x // PATCH, y // PATCH, feat))
        return out

    def encode(self, image, phase_bucket):
        active = []
        for gx, gy, feat in self.patches(image):
            key = (gx, gy)
            best = None
            best_score = 0.0
            for c in self.cells.get(key, []):
                s = c.similarity(feat)
                if s > best_score:
                    best_score = s
                    best = c
            if best is None or best_score < VISUAL_SIM_THRESHOLD:
                c = VisualCell(self.next_id, feat, gx, gy)
                self.next_id += 1
                self.cells.setdefault(key, []).append(c)
            else:
                best.update(feat, phase_bucket)
                c = best
            active.append(c)
        return active

    def feature_vector(self, image):
        values = []
        for _, _, feat in self.patches(image):
            values.extend(feat.tolist())
        return np.asarray(values, dtype=np.float32)

    def count(self):
        return sum(len(v) for v in self.cells.values())

    def decay(self):
        for cells in self.cells.values():
            for c in cells:
                c.decay()

# ============================================================
# PlaceState
# ============================================================


class PlaceState:
    """
    「場所」そのものではない。

    同じ空間座標でも、別の状態なら別の PlaceState になり得る。
    """

    def __init__(self, place_id, feature, signature):
        self.id = place_id
        self.center = np.asarray(feature, dtype=np.float32)
        self.signature = np.asarray(signature, dtype=np.float32)
        self.visits = 1
        self.energy = 1.0
        self.activation = 1.0
        self.transitions = {}
        self.state_history = deque(maxlen=MEMORY_WINDOW)
        self.phase_histogram = {}
        self.split_pressure = 0.0
        self.merge_pressure = 0.0

    def similarity(self, feature, signature):
        a = soft_distance(self.center, feature, 5.0)
        b = soft_distance(self.signature, signature, 7.0)
        return 0.65 * a + 0.35 * b

    def absorb(self, feature, signature, phase_bucket, state_vector):
        feature = np.asarray(feature, dtype=np.float32)
        signature = np.asarray(signature, dtype=np.float32)
        self.center = 0.92 * self.center + 0.08 * feature
        self.signature = 0.90 * self.signature + 0.10 * signature
        self.visits += 1
        self.energy = min(2.0, self.energy + 0.015)
        self.activation = min(2.0, self.activation + 0.05)
        self.phase_histogram[phase_bucket] = self.phase_histogram.get(phase_bucket, 0) + 1
        self.state_history.append(np.asarray(state_vector, dtype=np.float32))

    def connect(self, target_id):
        self.transitions[target_id] = self.transitions.get(target_id, 0) + 1

    def decay(self):
        self.activation *= 0.994
        self.energy *= 0.999

# ============================================================
# Transformation
# ============================================================


class TransformationCell:
    """
    世界状態の差分を記憶する。

    予測結果ではなく、実際に観測された変化を圧縮する。
    """

    def __init__(self, transform_id, before_place, after_place, action, delta):
        self.id = transform_id
        self.before_place = before_place
        self.after_place = after_place
        self.action_counts = {action: 1}
        self.delta = np.asarray(delta, dtype=np.float32)
        self.visits = 1
        self.energy = 1.0
        self.stability = 0.1
        self.error = 1.0
        self.contexts = deque(maxlen=80)
        self.history = deque(maxlen=MEMORY_WINDOW)

    def similarity(self, delta, before_place):
        if self.before_place != before_place:
            return 0.0
        return soft_distance(self.delta, delta, 2.8)

    def update(self, delta, action, context):
        delta = np.asarray(delta, dtype=np.float32)
        err = l1_distance(self.delta, delta)
        self.delta = 0.88 * self.delta + 0.12 * delta
        self.error = 0.90 * self.error + 0.10 * err
        self.stability = clamp(0.97 * self.stability + 0.03 * (1.0 - err), 0.0, 1.0)
        self.visits += 1
        self.energy = min(2.0, self.energy + 0.02)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.contexts.append(np.asarray(context, dtype=np.float32))
        self.history.append(float(err))

    def curiosity(self):
        novelty = 1.0 / math.sqrt(max(1, self.visits))
        instability = self.error
        isolation = 1.0 if len(self.action_counts) == 1 else 0.25
        return 0.8 * novelty + 1.1 * instability + 0.35 * isolation + 0.6 * (1.0 - self.stability)

    def decay(self):
        self.energy *= 0.998


class MetaTransformationCell:
    """
    「変化」から「変化の変化」を形成する。
    """

    def __init__(self, meta_id, delta_a, delta_b):
        self.id = meta_id
        self.delta_a = np.asarray(delta_a, dtype=np.float32)
        self.delta_b = np.asarray(delta_b, dtype=np.float32)
        self.meta_delta = self.delta_b - self.delta_a
        self.visits = 1
        self.energy = 1.0
        self.stability = 0.2

    def similarity(self, a, b):
        d1 = soft_distance(self.delta_a, a, 2.0)
        d2 = soft_distance(self.delta_b, b, 2.0)
        return 0.5 * (d1 + d2)

    def update(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        new_meta = b - a
        err = l1_distance(self.meta_delta, new_meta)
        self.delta_a = 0.92 * self.delta_a + 0.08 * a
        self.delta_b = 0.92 * self.delta_b + 0.08 * b
        self.meta_delta = 0.92 * self.meta_delta + 0.08 * new_meta
        self.stability = clamp(0.96 * self.stability + 0.04 * (1.0 - err), 0.0, 1.0)
        self.visits += 1
        self.energy = min(2.0, self.energy + 0.02)

# ============================================================
# Temporal memory / past leakage
# ============================================================


class TemporalTrace:
    def __init__(self, feature, body, place_id, time_index):
        self.feature = np.asarray(feature, dtype=np.float32).copy()
        self.body = np.asarray(body, dtype=np.float32).copy()
        self.place_id = place_id
        self.time_index = time_index
        self.strength = 1.0

    def decay(self):
        self.strength *= 0.995

# ============================================================
# Self-organizing World Model
# ============================================================


class WorldModel:
    def __init__(self):
        self.visual = SpatialRepresentation()
        self.places = []
        self.transforms = []
        self.meta_transforms = []
        self.traces = deque(maxlen=700)

        self.next_place = 0
        self.next_transform = 0
        self.next_meta = 0

        self.previous_state = {}
        self.previous_transform = {}
        self.error_history = deque(maxlen=500)
        self.phase_history = deque(maxlen=200)
        self.step_count = 0
        self.split_events = 0
        self.merge_events = 0

    # --------------------------------------------------------
    # Place identity is context dependent
    # --------------------------------------------------------

    def place_signature(self, image, body, local_context):
        visual = self.visual.feature_vector(image)
        # 強い次元圧縮を行わず、場所の見た目・身体・局所位相を同時に保つ。
        sig = np.concatenate([
            visual[::4],
            body,
            np.asarray(local_context, dtype=np.float32),
        ])
        return visual, sig

    def encode_place(self, image, body, local_context, phase_bucket, state_vector):
        feature, signature = self.place_signature(image, body, local_context)
        best = None
        best_score = 0.0

        for place in self.places:
            s = place.similarity(feature, signature)
            if s > best_score:
                best_score = s
                best = place

        if best is None or best_score < PLACE_SIM_THRESHOLD:
            place = PlaceState(self.next_place, feature, signature)
            self.next_place += 1
            place.phase_histogram[phase_bucket] = 1
            place.state_history.append(np.asarray(state_vector, dtype=np.float32))
            self.places.append(place)
            created = True
        else:
            # 同じ場所に見えるが、文脈が安定しない場合は split pressure を上げる。
            nearest_phase = max(best.phase_histogram.values()) if best.phase_histogram else 0
            total_phase = sum(best.phase_histogram.values()) if best.phase_histogram else 1
            phase_purity = nearest_phase / max(1, total_phase)
            if phase_purity < 0.35 and best.visits > 12:
                best.split_pressure += 0.015
            best.absorb(feature, signature, phase_bucket, state_vector)
            place = best
            created = False

        return place, feature, signature, created

    def maybe_split_place(self, place):
        if place.split_pressure < 1.0 or len(place.state_history) < 15:
            return None

        history = list(place.state_history)
        # 最後の半分を別の中心として再編成する。
        tail = np.mean(np.stack(history[-7:]), axis=0)
        old_center = np.mean(np.stack(history[:7]), axis=0)
        if l1_distance(tail, old_center) < 0.10:
            place.split_pressure *= 0.5
            return None

        new_place = PlaceState(
            self.next_place,
            place.center.copy(),
            place.signature.copy(),
        )
        self.next_place += 1
        new_place.center = place.center.copy()
        new_place.signature = place.signature.copy()
        new_place.state_history.extend(history[-7:])
        new_place.visits = max(1, place.visits // 5)
        new_place.activation = place.activation * 0.8
        place.split_pressure *= 0.35
        self.places.append(new_place)
        self.split_events += 1
        return new_place

    # --------------------------------------------------------
    # Transform formation
    # --------------------------------------------------------

    def find_transform(self, before_place, delta):
        best = None
        best_score = 0.0
        for t in self.transforms:
            s = t.similarity(delta, before_place.id)
            if s > best_score:
                best_score = s
                best = t
        return best, best_score

    def form_transform(self, before, after, action, delta, context):
        t, score = self.find_transform(before, delta)
        if t is None or score < TRANSFORM_SIM_THRESHOLD:
            t = TransformationCell(
                self.next_transform,
                before.id,
                after.id,
                action,
                delta,
            )
            self.next_transform += 1
            t.contexts.append(np.asarray(context, dtype=np.float32))
            self.transforms.append(t)
            error = 1.0
        else:
            before_after_distance = float(before.id != t.after_place)
            t.update(delta, action, context)
            error = 0.5 * t.error + 0.5 * before_after_distance
            if before.id != t.before_place:
                t.before_place = before.id
            # 同じ transformation が複数の結果へ向かうなら分岐圧を上げる。
            if t.after_place != after.id:
                before.merge_pressure += 0.02
                t.after_place = after.id

        before.connect(after.id)
        self.error_history.append(error)
        return t, error

    # --------------------------------------------------------
    # Meta transformation
    # --------------------------------------------------------

    def form_meta(self, transform_a, transform_b):
        if transform_a is None or transform_b is None:
            return
        best = None
        score = 0.0
        for m in self.meta_transforms:
            s = m.similarity(transform_a.delta, transform_b.delta)
            if s > score:
                score = s
                best = m
        if best is None or score < META_SIM_THRESHOLD:
            m = MetaTransformationCell(
                self.next_meta,
                transform_a.delta,
                transform_b.delta,
            )
            self.next_meta += 1
            self.meta_transforms.append(m)
        else:
            best.update(transform_a.delta, transform_b.delta)

    # --------------------------------------------------------
    # Learn one experience
    # --------------------------------------------------------

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
    ):
        self.step_count += 1

        self.visual.encode(image_before, phase_bucket)
        self.visual.encode(image_after, phase_bucket)

        state_before = np.concatenate([
            self.visual.feature_vector(image_before)[::4],
            body_before,
            np.asarray(local_context_before, dtype=np.float32),
        ])
        state_after = np.concatenate([
            self.visual.feature_vector(image_after)[::4],
            body_after,
            np.asarray(local_context_after, dtype=np.float32),
        ])

        before, _, _, created_before = self.encode_place(
            image_before,
            body_before,
            local_context_before,
            phase_bucket,
            state_before,
        )
        after, _, _, created_after = self.encode_place(
            image_after,
            body_after,
            local_context_after,
            phase_bucket,
            state_after,
        )

        delta = state_after - state_before
        context = np.concatenate([
            body_before,
            body_after,
            np.asarray(local_context_before, dtype=np.float32),
        ])

        transform, error = self.form_transform(
            before,
            after,
            action,
            delta,
            context,
        )

        prev_t = self.previous_transform.get(agent_id)
        if prev_t is not None:
            self.form_meta(prev_t, transform)
        self.previous_transform[agent_id] = transform

        trace = TemporalTrace(
            state_after[::4],
            body_after,
            after.id,
            self.step_count,
        )
        self.traces.append(trace)

        self.previous_state[agent_id] = after.id
        self.phase_history.append(phase_bucket)

        split = self.maybe_split_place(before)

        return {
            "before": before,
            "after": after,
            "transform": transform,
            "error": error,
            "created_before": created_before,
            "created_after": created_after,
            "split": split,
        }

    # --------------------------------------------------------
    # Strange memory replay
    # --------------------------------------------------------

    def replay(self, count=300):
        """
        過去の記憶をそのまま復唱するのではなく、
        現在の Place 構造との一致/不一致を再評価する。
        """
        if not self.traces or not self.transforms:
            return

        weights = np.asarray([
            1.0
            + 1.5 * t.error
            + 1.0 * t.curiosity()
            + 0.8 * (1.0 - t.stability)
            for t in self.transforms
        ], dtype=np.float64)
        weights = np.maximum(weights, 1e-8)
        weights /= weights.sum()

        for _ in range(count):
            t = np.random.choice(self.transforms, p=weights)
            t.energy = min(2.0, t.energy + 0.02)

            # 高誤差変化ほど統合よりも「保持」を優先する。
            if t.error > 0.35:
                t.stability *= 0.999
            else:
                t.stability = clamp(t.stability + 0.001, 0.0, 1.0)

        for trace in self.traces:
            trace.decay()

    # --------------------------------------------------------
    # Action choice by structural curiosity
    # --------------------------------------------------------

    def score_action(self, image, body, local_context, action, recent_action):
        feature, signature = self.place_signature(image, body, local_context)
        best_place = None
        best_score = 0.0
        for p in self.places:
            s = p.similarity(feature, signature)
            if s > best_score:
                best_score = s
                best_place = p

        score = ACTION_ACTIVITY_BIAS.get(action, 0.0)

        if best_place is None:
            return score + 2.5 + random.random() * 0.3

        candidates = [
            t for t in self.transforms
            if t.before_place == best_place.id
            and action in t.action_counts
        ]

        if not candidates:
            # 未観測の作用を優先。
            score += 2.2
            score += random.random() * 0.4
        else:
            curiosity = max(t.curiosity() for t in candidates)
            score += curiosity

            # その場所が分岐し始めているなら、
            # 異なる action を観測する価値が上がる。
            branch = len(best_place.transitions)
            score += min(1.0, branch * 0.08)

            unstable = np.mean([t.error for t in candidates])
            score += 0.9 * unstable

        if recent_action in (ACTION_LEFT, ACTION_RIGHT) and action == recent_action:
            score += 0.12

        if recent_action is not None and action == recent_action:
            score -= 0.025

        return score

    def select_action(self, image, body, local_context, recent_action):
        scores = [
            self.score_action(image, body, local_context, a, recent_action)
            for a in ACTIONS
        ]
        order = np.argsort(scores)[::-1]
        if len(order) >= 3 and random.random() < 0.28:
            return ACTIONS[int(random.choice(order[:3]))]
        return ACTIONS[int(order[0])]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def decay(self):
        self.visual.decay()
        for p in self.places:
            p.decay()
        for t in self.transforms:
            t.decay()
        for m in self.meta_transforms:
            m.energy *= 0.998

    def statistics(self):
        mean_error = float(np.mean(self.error_history)) if self.error_history else 0.0
        avg_stability = float(np.mean([t.stability for t in self.transforms])) if self.transforms else 0.0
        avg_split = float(np.mean([p.split_pressure for p in self.places])) if self.places else 0.0
        return {
            "visual": self.visual.count(),
            "places": len(self.places),
            "transforms": len(self.transforms),
            "meta": len(self.meta_transforms),
            "mean_error": mean_error,
            "stability": avg_stability,
            "split_pressure": avg_split,
            "splits": self.split_events,
            "merges": self.merge_events,
        }

# ============================================================
# Physical strange world
# ============================================================


class World:
    def __init__(self):
        self.drawer = turtle.Turtle(visible=False)
        self.drawer.penup()
        self.drawer.speed(0)
        self.regime = WorldRegime()
        self.time = 0

        self.base_platforms = [
            (-530, -400, -220),
            (-470, -300, -105),
            (-340, -250, 40),
            (-230, -80, -40),
            (-50, 40, 75),
            (80, 240, -20),
            (180, 350, 100),
            (315, 510, -90),
            (430, 550, 50),
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
        self.draw()

    def reset(self):
        self.time = 0
        self.regime = WorldRegime()
        self.moving_x = 0.0
        self.moving_y = 145.0
        self.ghost_x = 0.0
        self.ghost_y = -10.0
        self.draw()

    def update(self):
        self.regime.update()
        self.time = self.regime.time
        scale = self.regime.time_scale()
        self.moving_x = 245.0 * math.sin(self.time * 0.028 * scale)
        self.moving_y = 145.0 + 65.0 * math.sin(self.time * 0.047 + self.regime.global_phase * math.pi)
        self.ghost_x = 170.0 * math.sin(self.time * 0.015 - 1.1)
        self.ghost_y = 45.0 + 100.0 * math.cos(self.time * 0.021)

    def draw_line(self, x1, y1, x2, y2, width, color):
        d = self.drawer
        d.color(color)
        d.pensize(width)
        d.goto(x1, y1)
        d.pendown()
        d.goto(x2, y2)
        d.penup()

    def effective_platforms(self):
        phase = self.regime.topology_phase
        out = []
        for i, (x1, x2, y) in enumerate(self.base_platforms):
            wobble = 28.0 * math.sin(self.time * 0.026 + i * 1.3) * phase
            shear = 18.0 * math.sin(self.time * 0.019 + i)
            out.append((x1 + shear, x2 - shear * 0.4, y + wobble))
        if phase > 0.48:
            out.extend(self.extra_platforms[:int(1 + phase * len(self.extra_platforms))])
        return out

    def platform_y(self, x, y):
        for x1, x2, py in self.effective_platforms():
            if x1 <= x <= x2 and abs(y - py) < 24:
                return py
        return None

    def hazard_at(self, x, y):
        phase = self.regime.global_phase
        for x1, x2, hy in self.hazards:
            shift = 22.0 * math.sin(self.time * 0.031 + x1) * phase
            if x1 + shift <= x <= x2 + shift and y <= hy + 20:
                return True
        return False

    def wormhole(self, x, y):
        s = self.regime.wormhole_strength()
        if s <= 0.2:
            return False
        return abs(x - 0.0) < 28 and abs(y - 130) < 42

    def local_context(self, x, y, agent_id):
        gx = self.regime.local_gravity(x, y)
        phase = self.regime.local_phase(x, y, agent_id)
        return np.asarray([
            gx / max(0.1, BASE_GRAVITY),
            phase,
            self.regime.time_scale() / 2.0,
            self.regime.wormhole_strength(),
            self.regime.past_leak_strength(),
            self.regime.mirror_strength(),
        ], dtype=np.float32)

    def all_platforms(self):
        return self.effective_platforms()

    def step(self, agent, action):
        dt = self.regime.time_scale()
        x, y = agent.x, agent.y
        vx, vy = agent.vx, agent.vy
        grounded = agent.grounded

        local_phase = self.regime.local_phase(x, y, agent.id)
        gravity = self.regime.local_gravity(x, y)
        mirror = self.regime.mirror_strength()

        if action == ACTION_LEFT:
            vx -= (GROUND_ACCEL if grounded else AIR_ACCEL) * dt
            agent.heading = -1
        elif action == ACTION_RIGHT:
            vx += (GROUND_ACCEL if grounded else AIR_ACCEL) * dt
            agent.heading = 1
        elif action == ACTION_JUMP:
            if grounded:
                vy = JUMP_POWER * (0.65 + 0.55 * local_phase)
                grounded = False
                agent.jumps = 1
            elif agent.jumps < 2:
                vy = JUMP_POWER * 0.82
                agent.jumps += 1
        elif action == ACTION_BRAKE:
            vx *= 0.22 + 0.22 * local_phase
        elif action == ACTION_WAIT:
            vx *= 0.90

        # 局所鏡像領域では水平運動が反転する。
        if mirror > 0 and -170 < x < 170:
            vx *= -1.0 if self.time % 14 == 0 else 1.0

        vx = float(np.clip(vx, -MAX_SPEED, MAX_SPEED))
        vx *= GROUND_FRICTION if grounded else AIR_FRICTION

        if not grounded:
            # 重力の符号自体が場所依存。
            vy -= gravity * dt
            y += vy * dt

        x += vx * dt

        if x < WORLD_LEFT + 12:
            x = WORLD_LEFT + 12
            vx *= -0.35
        if x > WORLD_RIGHT - 12:
            x = WORLD_RIGHT - 12
            vx *= -0.35

        if self.wormhole(x, y):
            # 遠距離転移。単なる移動では説明しにくい変化。
            x = 430.0 if x < 0 else -430.0
            y = 50.0 + 100.0 * local_phase
            vx *= -0.5
            vy *= 0.35
            grounded = False

        py = self.platform_y(x, y)
        if py is not None and vy <= 0 and y <= py + 23:
            y = py + 23
            vy = 0.0
            grounded = True
            agent.jumps = 0
        elif y < WORLD_BOTTOM + 5:
            x = -480 + agent.id * 24
            y = -195
            vx = 0.0
            vy = 0.0
            grounded = True
            agent.jumps = 0
        else:
            grounded = False

        if self.hazard_at(x, y):
            # reset, no reward and no penalty.
            x = -480 + agent.id * 24
            y = -195
            vx = 0.0
            vy = 0.0
            grounded = True
            agent.jumps = 0

        agent.x, agent.y = x, y
        agent.vx, agent.vy = vx, vy
        agent.grounded = grounded

    def draw(self):
        d = self.drawer
        d.clear()
        # border
        for args in [
            (WORLD_LEFT, WORLD_BOTTOM, WORLD_RIGHT, WORLD_BOTTOM),
            (WORLD_RIGHT, WORLD_BOTTOM, WORLD_RIGHT, WORLD_TOP),
            (WORLD_RIGHT, WORLD_TOP, WORLD_LEFT, WORLD_TOP),
            (WORLD_LEFT, WORLD_TOP, WORLD_LEFT, WORLD_BOTTOM),
        ]:
            self.draw_line(*args, 3, "#3b3d4d")

        # topology phase changes color/intensity of structural lines
        phase = self.regime.topology_phase
        structure_color = "#5f9ea0" if phase < 0.5 else "#ad7be8"
        for x1, x2, y in self.effective_platforms():
            self.draw_line(x1, y, x2, y, 6, structure_color)

        for x1, x2, y in self.hazards:
            shift = 22.0 * math.sin(self.time * 0.031 + x1) * self.regime.global_phase
            self.draw_line(x1 + shift, y, x2 + shift, y, 10, "#ff4055")

        d.goto(self.moving_x, self.moving_y)
        d.dot(18, "#f0a63a")

        if self.regime.past_leak_strength() > 0.35:
            d.goto(self.ghost_x, self.ghost_y)
            d.dot(14, "#bb88ff")

        # strange wells / portals
        if self.regime.wormhole_strength() > 0.2:
            d.goto(0, 130)
            d.dot(34, "#6b55ff")
            d.goto(0, 130)
            d.dot(13, "#f4eaff")

        # faint phase markings. They are visual, not semantic inputs.
        d.color("#66697d")
        d.goto(-500, 250)
        d.write(f"WORLD PHASE {self.regime.global_phase:.2f}", font=("Arial", 10, "normal"))
        d.goto(-500, 230)
        d.write(f"TIME SCALE {self.regime.time_scale():.2f}", font=("Arial", 10, "normal"))

# ============================================================
# Visual field
# ============================================================


class VisualField:
    """
    World drawingから画像テンソルを作る。
    semantic label をAIへ渡さない。
    """

    def __init__(self, world):
        self.world = world

    def to_grid(self, x, y):
        gx = int((x - WORLD_LEFT) / (WORLD_RIGHT - WORLD_LEFT) * (OBS_W - 1))
        gy = int((y - WORLD_BOTTOM) / (WORLD_TOP - WORLD_BOTTOM) * (OBS_H - 1))
        return clamp(gx, 0, OBS_W - 1), clamp(gy, 0, OBS_H - 1)

    def seg(self, channel, x1, y1, x2, y2, value):
        gx1, gy1 = self.to_grid(x1, y1)
        gx2, gy2 = self.to_grid(x2, y2)
        n = max(abs(gx2 - gx1), abs(gy2 - gy1), 1)
        for i in range(n + 1):
            t = i / n
            gx = int(gx1 + (gx2 - gx1) * t)
            gy = int(gy1 + (gy2 - gy1) * t)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = gx + dx, gy + dy
                    if 0 <= xx < OBS_W and 0 <= yy < OBS_H:
                        channel[yy, xx] = max(channel[yy, xx], value)

    def dot(self, channel, x, y, value):
        gx, gy = self.to_grid(x, y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx, yy = gx + dx, gy + dy
                if 0 <= xx < OBS_W and 0 <= yy < OBS_H:
                    channel[yy, xx] = max(channel[yy, xx], value)

    def capture(self, agents, viewer_id=None):
        img = np.zeros((OBS_H, OBS_W, CHANNELS), dtype=np.float32)

        # structural
        for x1, x2, y in self.world.effective_platforms():
            self.seg(img[:, :, 0], x1, y, x2, y, 0.75)

        # dynamic
        for x1, x2, y in self.world.hazards:
            shift = 22.0 * math.sin(self.world.time * 0.031 + x1) * self.world.regime.global_phase
            self.seg(img[:, :, 1], x1 + shift, y, x2 + shift, y, 1.0)
        self.dot(img[:, :, 1], self.world.moving_x, self.world.moving_y, 0.9)
        if self.world.regime.past_leak_strength() > 0.35:
            self.dot(img[:, :, 1], self.world.ghost_x, self.world.ghost_y, 0.7)
        if self.world.regime.wormhole_strength() > 0.2:
            self.dot(img[:, :, 1], 0, 130, 1.0)

        # agents
        for a in agents:
            if viewer_id is not None and a.id == viewer_id:
                self.dot(img[:, :, 2], a.x, a.y, 1.0)
            else:
                self.dot(img[:, :, 2], a.x, a.y, 0.75)

        # hidden-looking phase texture: still part of the observed image.
        # This creates non-semantic temporal texture.
        for i in range(0, OBS_W, 7):
            x = WORLD_LEFT + (WORLD_RIGHT - WORLD_LEFT) * i / (OBS_W - 1)
            y = 195 + 12 * math.sin(self.world.time * 0.02 + i * 0.7)
            gx, gy = self.to_grid(x, y)
            img[gy, gx, 3] = max(
                img[gy, gx, 3],
                0.15 + 0.45 * self.world.regime.global_phase,
            )

        return img.reshape(-1)

# ============================================================
# Agent
# ============================================================


class Agent:
    def __init__(self, agent_id, color, world, model, vision):
        self.id = agent_id
        self.color = color
        self.world = world
        self.model = model
        self.vision = vision
        self.turtle = turtle.Turtle(shape="turtle")
        self.turtle.color(color)
        self.turtle.penup()
        self.turtle.speed(0)
        self.reset()

    def reset(self):
        self.x = -480 + self.id * 24
        self.y = -195
        self.vx = 0.0
        self.vy = 0.0
        self.grounded = True
        self.jumps = 0
        self.heading = 1
        self.last_action = ACTION_NONE
        self.last_error = 0.0
        self.steps = 0
        self.turtle.goto(self.x, self.y)

    def body_state(self):
        return np.asarray([
            self.x / 560.0,
            self.y / 300.0,
            self.vx / MAX_SPEED,
            self.vy / 13.0,
            float(self.grounded),
            self.jumps / 2.0,
            self.heading,
        ], dtype=np.float32)

    def local_context(self):
        return self.world.local_context(self.x, self.y, self.id)

    def step(self, agents):
        image_before = self.vision.capture(agents, self.id)
        body_before = self.body_state()
        ctx_before = self.local_context()

        if random.random() < 0.08:
            action = random.choice([ACTION_LEFT, ACTION_RIGHT, ACTION_JUMP, ACTION_WAIT])
        else:
            action = self.model.select_action(
                image_before,
                body_before,
                ctx_before,
                self.last_action,
            )

        self.world.step(self, action)
        self.world.draw()

        image_after = self.vision.capture(agents, self.id)
        body_after = self.body_state()
        ctx_after = self.local_context()

        phase_bucket = int(self.world.regime.local_phase(self.x, self.y, self.id) * 8.0)
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
        )

        self.last_action = action
        self.last_error = result["error"]
        self.steps += 1
        self.turtle.goto(self.x, self.y)

# ============================================================
# UI
# ============================================================


ui = turtle.Turtle(visible=False)
ui.penup()
ui.speed(0)
mem = turtle.Turtle(visible=False)
mem.penup()
mem.speed(0)


def write(x, y, text, size=10, color="#dddddd"):
    ui.goto(x, y)
    ui.color(color)
    ui.write(text, font=("Arial", size, "normal"))


def draw_model(model, agents, episode, step, world):
    ui.clear()
    mem.clear()
    s = model.statistics()

    write(-540, 325, "STRANGE SELF-ORGANIZING WORLD", 15, "#ffffff")
    write(-540, 302, f"EPISODE {episode + 1}/{MAX_EPISODES}   STEP {step}/{STEPS_PER_EPISODE}", 10, "#55ffbb")

    write(560, 265, f"VisualCells : {s['visual']}", 10)
    write(560, 245, f"PlaceStates : {s['places']}", 10)
    write(560, 225, f"Transforms  : {s['transforms']}", 10)
    write(560, 205, f"Meta        : {s['meta']}", 10)
    write(560, 185, f"Err         : {s['mean_error']:.3f}", 10, "#ffba45")
    write(560, 165, f"Stability   : {s['stability']:.3f}", 10, "#88ccff")
    write(560, 145, f"Splits      : {s['splits']}", 10, "#ff77aa")

    write(560, 120, "PLACE STATES", 10, "#ffffff")
    base_x, base_y = 560, 85
    maxp = min(42, len(model.places))
    coords = {}
    for i in range(maxp):
        p = model.places[i]
        x = base_x + (i % 7) * 42
        y = base_y - (i // 7) * 38
        coords[i] = (x, y)
        r = min(15, 5 + int(math.log1p(p.visits)))
        mem.goto(x, y)
        mem.dot(r, "#5ea8ff")

    for i in range(maxp):
        p = model.places[i]
        x1, y1 = coords[i]
        for target, count in list(p.transitions.items())[-8:]:
            if target not in coords:
                continue
            x2, y2 = coords[target]
            mem.goto(x1, y1)
            mem.pendown()
            mem.goto(x2, y2)
            mem.penup()

    # transformation graph at far right/bottom
    tx, ty = 560, -150
    write(tx, ty + 45, "TRANSFORMATION CURIOSITY", 10, "#ffffff")
    ranked = sorted(model.transforms, key=lambda t: t.curiosity(), reverse=True)[:10]
    for i, t in enumerate(ranked):
        y = ty + 18 - i * 18
        write(tx, y, f"T{t.id:03d}  v={t.visits:<3d}  c={t.curiosity():.2f}  s={t.stability:.2f}", 8, "#ffdd66")

    y = 225
    for a in agents:
        write(-540, y, f"A{a.id} {ACTION_NAMES[a.last_action]:<5} x={a.x:+.0f} y={a.y:+.0f} vx={a.vx:+.1f} err={a.last_error:.2f}", 9, a.color)
        y -= 19

# ============================================================
# Main
# ============================================================


screen = turtle.Screen()
screen.setup(SCREEN_W, SCREEN_H)
screen.bgcolor("#080910")
screen.title("Strange Place / Self-Organizing World Model v2")
screen.tracer(False)

world = World()
vision = VisualField(world)
model = WorldModel()
colors = ["#00eaff", "#57ff9b", "#ff9f43"]
agents = [Agent(i, colors[i], world, model, vision) for i in range(NUM_AGENTS)]

episode = 0
step = 0
finished = False


def reset_episode():
    global step
    step = 0
    world.reset()
    for a in agents:
        a.reset()


def sleep_phase():
    global episode, finished
    model.replay(520)
    model.decay()
    draw_model(model, agents, episode, step, world)
    screen.update()
    episode += 1
    if episode >= MAX_EPISODES:
        finished = True
        write(-180, -335, "SIMULATION FINISHED", 17, "#ffffff")
        screen.update()
        return
    screen.ontimer(start_episode, 800)


def run_step():
    global step
    if finished:
        return

    world.update()
    world.draw()

    for agent in agents:
        agent.step(agents)

    draw_model(model, agents, episode, step, world)
    screen.update()
    step += 1

    if step < STEPS_PER_EPISODE:
        screen.ontimer(run_step, 20)
    else:
        screen.ontimer(sleep_phase, 180)


def start_episode():
    if finished:
        return
    reset_episode()
    run_step()


reset_episode()
world.draw()
draw_model(model, agents, episode, step, world)
screen.update()
screen.ontimer(start_episode, 600)
turtle.done()
