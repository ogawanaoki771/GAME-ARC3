# ============================================================
# EVOLVED EMBODIED SELF-ORGANIZING WORLD MODEL v5.0
# ============================================================
#
# 既存のv3脳を維持し、巨大な物理実験場と物理概念層を追加
#
# EventCell
#      ↓
# TransitionCell
#      ↓
# PlaceCell
#      ↓
# ConceptCell
#      ↓
# DreamSimulator
#
#                 ↓ v3 UPPER SYSTEMS ↓
#
# SelfModel
# CuriosityEngine
# CausalModel
# GoalSystem
# SkillSystem
# Planner
# CounterfactualDream
# ReflectionSystem
# MultiAgentSpecialization
# SleepConsolidation
#
# 3 AGENTS
#      ↓
# SHARED HIPPOCAMPUS
#      ↓
# SHARED WORLD MODEL
#      ↓
# SHARED CAUSAL / SKILL KNOWLEDGE
#
# ============================================================

import turtle
import random
import math
import numpy as np
from collections import defaultdict, deque


# ============================================================
# 1. CONFIG
# ============================================================

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 760

NUM_AGENTS = 3
STEPS_PER_EPISODE = 180
MAX_EPISODES = 15

DAY_DELAY = 35
SLEEP_DELAY = 900

NUM_ACTIONS = 11

ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3
ACTION_DASH = 4
ACTION_BRAKE = 5
ACTION_WAIT = 6
ACTION_SLIDE = 7
ACTION_WALL_JUMP = 8
ACTION_GRAPPLE = 9
ACTION_RELEASE = 10

ACTION_NAMES = {
    ACTION_NONE: "NONE",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_JUMP: "JUMP",
    ACTION_DASH: "DASH",
    ACTION_BRAKE: "BRAKE",
    ACTION_WAIT: "WAIT",
    ACTION_SLIDE: "SLIDE",
    ACTION_WALL_JUMP: "WALL_JUMP",
    ACTION_GRAPPLE: "GRAPPLE",
    ACTION_RELEASE: "RELEASE",
}

# v5.1 physical environment tuning
SLOPE_ACCELERATION = 0.12
AIR_WIND_SCALE = 0.55
PLATFORM_CARRY_SCALE = 0.65
SLIPPERY_FRICTION = 0.985
ROUGH_FRICTION = 0.62
SLIDE_FRICTION = 0.995
WALL_JUMP_POWER = 11.5
GRAPPLE_RANGE = 190.0
GRAPPLE_PULL = 1.8
GRAPPLE_DAMP = 0.92

# v5 physical-chain parameters
CHAIN_MAX = 4
CHAIN_MEMORY = 1600
SWING_INHERIT = 0.72
PENDULUM_TANGENT_IMPULSE = 6.0
CRUMBLE_LOAD_SCALE = 0.035
RAMP_MOMENTUM_SCALE = 0.65
WIND_FEEDBACK_SCALE = 0.18
PLATFORM_REACTION_SCALE = 0.30


# ============================================================
# 2. PHYSICS
# ============================================================

MAX_SPEED = 8.0
GROUND_ACCEL = 0.9
AIR_ACCEL = 0.45

GROUND_FRICTION = 0.82
AIR_FRICTION = 0.96

GRAVITY = 0.65

JUMP_POWER = 11.0
DOUBLE_JUMP_POWER = 9.0
MAX_JUMPS = 2

DASH_SPEED = 18.0
DASH_DURATION = 4


# ============================================================
# 3. MEMORY CONFIG
# ============================================================

MAX_EVENTS = 1200
MAX_TRANSITIONS = 4000
MAX_REPLAY = 4000

EVENT_SIM_THRESHOLD = 0.88
PLACE_RADIUS = 0.70

MAX_ERROR_HISTORY = 1500

REPLAY_COUNT = 600
CONCEPT_REPLAY_COUNT = 300
SKILL_REPLAY_COUNT = 200

DREAM_SOURCES = 20
DREAM_SAMPLES = 25
DREAM_STEPS = 12

PLANNING_DEPTH = 5


# ============================================================
# 4. UTILS
# ============================================================

def clamp(x, a, b):
    return max(a, min(b, x))


def softmax(values, temperature=1.0):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    temperature = max(0.05, temperature)

    values = values / temperature
    values -= np.max(values)

    exp_values = np.exp(values)
    total = exp_values.sum()

    if total <= 0:
        return np.ones(len(values)) / len(values)

    return exp_values / total


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na <= 1e-9 or nb <= 1e-9:
        return 0.0

    return float(np.dot(a, b) / (na * nb))


# ============================================================
# 5. REPLAY MEMORY
# ============================================================

class ReplayMemory:

    def __init__(self, max_size=MAX_REPLAY):
        self.memory = []
        self.max_size = max_size

    def add(
        self,
        previous_event,
        current_event,
        action,
        reward,
        error,
        agent_id=0,
    ):
        self.memory.append({
            "previous": previous_event,
            "current": current_event,
            "action": int(action),
            "reward": float(reward),
            "error": float(error),
            "agent_id": int(agent_id),
        })

        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def sample_batch(self, n):

        if not self.memory:
            return []

        n = min(n, len(self.memory))

        weights = np.array([
            1.0 + abs(item["error"])
            for item in self.memory
        ])

        weights /= max(weights.sum(), 1e-9)

        indices = np.random.choice(
            len(self.memory),
            size=n,
            replace=False,
            p=weights,
        )

        return [self.memory[i] for i in indices]

    def __len__(self):
        return len(self.memory)


# ============================================================
# 6. EVENT CELL
# ============================================================

class EventCell:

    # v3.1: EventCell = 状態ノード。
    # action はメタデータとして保持し、状態検索には使わない。
    # 遷移の意味は TransitionCell に一本化する。

    def __init__(
        self,
        event_id,
        state,
        action=ACTION_NONE,
        agent_id=0,
        event_name="NORMAL",
    ):
        self.id = event_id
        self.state = np.array(state, dtype=float)
        self.action = int(action)
        self.agent_id = agent_id
        self.event_name = str(event_name)

        self.visits = 1
        self.energy = 1.0
        self.activation = 1.0
        self.reward_mean = 0.0
        self.error_mean = 1.0

        self.place_id = None
        self.concepts = []

    def similarity(self, state, action=None):
        state = np.asarray(state, dtype=float)
        distance = np.linalg.norm(self.state - state)
        return math.exp(-distance * 1.8)

    def reinforce(
        self,
        state,
        reward,
        error,
        action=None,
        event_name=None,
    ):
        self.visits += 1
        self.energy = min(3.0, self.energy + 0.05)
        self.activation = min(2.0, self.activation + 0.10)

        state = np.asarray(state, dtype=float)
        self.state = 0.90 * self.state + 0.10 * state

        if action is not None:
            self.action = int(action)
        if event_name is not None:
            self.event_name = str(event_name)

        self.reward_mean = 0.90 * self.reward_mean + 0.10 * reward
        self.error_mean = 0.90 * self.error_mean + 0.10 * error

    def decay(self):
        self.energy *= 0.997
        self.activation *= 0.985


# ============================================================
# 7. TRANSITION CELL
# ============================================================
# 7. TRANSITION CELL
# ============================================================

class TransitionCell:

    def __init__(
        self,
        transition_id,
        source_id,
        action,
        target_id,
        next_state,
        reward,
        error,
    ):

        self.id = transition_id
        self.source_id = source_id
        self.action = int(action)

        self.target_counts = {
            target_id: 1
        }

        self.predicted_next_state = np.array(
            next_state,
            dtype=float,
        )

        self.reward_mean = float(reward)
        self.error_mean = float(error)

        self.visits = 1

        self.success_count = (
            1 if reward > 0 else 0
        )

        self.failure_count = (
            1 if reward < -3 else 0
        )

        self.energy = 1.0

        self.confidence = 0.2

    def update(
        self,
        target_id,
        next_state,
        reward,
        error,
    ):

        self.target_counts[target_id] = (
            self.target_counts.get(
                target_id,
                0,
            )
            + 1
        )

        self.visits += 1

        self.predicted_next_state = (
            0.85 * self.predicted_next_state
            +
            0.15 * np.asarray(
                next_state,
                dtype=float,
            )
        )

        self.reward_mean = (
            0.85 * self.reward_mean
            +
            0.15 * reward
        )

        self.error_mean = (
            0.85 * self.error_mean
            +
            0.15 * error
        )

        if reward > 0:
            self.success_count += 1

        if reward < -3:
            self.failure_count += 1

        self.energy = min(
            3.0,
            self.energy + 0.04,
        )

        self.confidence = (
            1.0
            -
            math.exp(
                -self.visits / 8.0
            )
        )

    def best_target(self):

        if not self.target_counts:
            return None

        return max(
            self.target_counts,
            key=self.target_counts.get,
        )

    def target_probability(self, target_id):

        total = sum(
            self.target_counts.values()
        )

        if total <= 0:
            return 0.0

        return (
            self.target_counts.get(
                target_id,
                0,
            )
            /
            total
        )

    def risk(self):

        failure_rate = (
            self.failure_count
            /
            max(1, self.visits)
        )

        return (
            failure_rate
            +
            self.error_mean * 0.15
        )

    def decay(self):

        self.energy *= 0.998


# ============================================================
# 8. PLACE CELL
# ============================================================

class PlaceCell:

    def __init__(self, place_id):

        self.id = place_id
        self.center = None

        self.events = []

        self.visits = 0

        self.mean_reward = 0.0
        self.mean_error = 0.0

        self.energy = 1.0

        self.concepts = []

    def add_event(self, event):

        if event.id not in self.events:
            self.events.append(event.id)

        if self.center is None:

            self.center = event.state.copy()

        else:

            self.center = (
                0.90 * self.center
                +
                0.10 * event.state
            )

        self.visits += 1

        self.mean_reward = (
            0.90 * self.mean_reward
            +
            0.10 * event.reward_mean
        )

        self.mean_error = (
            0.90 * self.mean_error
            +
            0.10 * event.error_mean
        )

        self.energy = min(
            2.5,
            self.energy + 0.02,
        )

        event.place_id = self.id

    def distance(self, state):

        if self.center is None:
            return 999999.0

        return float(
            np.linalg.norm(
                self.center
                -
                np.asarray(
                    state,
                    dtype=float,
                )
            )
        )

    def decay(self):

        self.energy *= 0.998


# ============================================================
# 9. CONCEPT CELL
# ============================================================

class ConceptCell:

    def __init__(
        self,
        concept_id,
        name,
    ):

        self.id = concept_id
        self.name = name

        self.events = []
        self.places = []

        self.links = {}

        self.visits = 0

        self.value = 0.0
        self.activation = 0.0

        self.energy = 1.0

    def absorb(self, event):

        if event.id not in self.events:
            self.events.append(event.id)

        if self.id not in event.concepts:
            event.concepts.append(self.id)

        self.visits += 1

        self.activation = (
            0.80 * self.activation
            +
            0.20
        )

        self.value = (
            0.90 * self.value
            +
            0.10 * event.reward_mean
        )

        self.energy = min(
            2.5,
            self.energy + 0.02,
        )

    def strengthen(
        self,
        target_id,
        amount=0.05,
    ):

        self.links[target_id] = min(
            8.0,
            self.links.get(
                target_id,
                0.0,
            )
            +
            amount
        )

    def decay(self):

        self.activation *= 0.97
        self.energy *= 0.998


# ============================================================
# 9.5 PHYSICS CONCEPT
# ============================================================

class PhysicsConcept:
    """
    Action -> Physics Event -> Consequence abstraction.

    This is deliberately separate from the legacy ConceptCell so the
    original v3 architecture remains intact. The new concept layer learns
    reusable physical relations without requiring external reward.
    """

    def __init__(self, concept_id, action_name, event_name, consequence):
        self.id = concept_id
        self.action = str(action_name)
        self.event = str(event_name)
        self.consequence = str(consequence)
        self.key = (self.action, self.event, self.consequence)

        self.visits = 0
        self.energy = 1.0
        self.strength = 0.0
        self.mean_error = 0.0
        self.mean_dx = 0.0
        self.mean_dy = 0.0
        self.mean_dvx = 0.0
        self.mean_dvy = 0.0

    def observe(self, dx, dy, dvx, dvy, error):
        self.visits += 1
        a = 1.0 / min(self.visits, 20)
        self.mean_dx = (1-a) * self.mean_dx + a * dx
        self.mean_dy = (1-a) * self.mean_dy + a * dy
        self.mean_dvx = (1-a) * self.mean_dvx + a * dvx
        self.mean_dvy = (1-a) * self.mean_dvy + a * dvy
        self.mean_error = (1-a) * self.mean_error + a * error
        self.strength = min(1.0, self.strength + 0.04 + 0.01 * min(error, 5.0))
        self.energy = min(3.0, self.energy + 0.03)

    def decay(self):
        self.energy *= 0.998
        self.strength *= 0.999



# ============================================================
# 10. HIPPOCAMPUS
# ============================================================

class PhysicsChainConcept:
    """
    Learned multi-step physical chain:

        ACTION -> EVENT -> EVENT -> CONSEQUENCE

    The chain is discovered from observed event sequences, never supplied as
    a scripted solution.  It is intentionally value-free: reward is ignored.
    """

    def __init__(self, concept_id, action_name, events, consequence):
        self.id = concept_id
        self.action = str(action_name)
        self.events = tuple(str(e) for e in events)
        self.consequence = str(consequence)
        self.key = (self.action, self.events, self.consequence)
        self.visits = 0
        self.energy = 1.0
        self.strength = 0.0
        self.mean_error = 0.0

    def observe(self, error):
        self.visits += 1
        a = 1.0 / min(self.visits, 25)
        self.mean_error = (1.0 - a) * self.mean_error + a * float(error)
        self.strength = min(1.0, self.strength + 0.025 + 0.004 * min(float(error), 5.0))
        self.energy = min(3.0, self.energy + 0.035)

    def decay(self):
        self.energy *= 0.998
        self.strength *= 0.999


class Hippocampus:

    def __init__(self):

        self.events = []
        self.transitions = []
        self.places = []
        self.concepts = []

        self.next_event_id = 0
        self.next_transition_id = 0
        self.next_place_id = 0
        self.next_concept_id = 0

        self.total_encodes = 0
        self.total_transitions = 0

    def get_event(self, event_id):

        for event in self.events:

            if event.id == event_id:
                return event

        return None

    def get_transition(
        self,
        source_id,
        action,
    ):

        for transition in self.transitions:

            if (
                transition.source_id == source_id
                and
                transition.action == action
            ):
                return transition

        return None

    def get_concept(self, concept_id):

        for concept in self.concepts:

            if concept.id == concept_id:
                return concept

        return None

    def encode_event(
        self,
        state,
        action=ACTION_NONE,
        reward=0.0,
        error=1.0,
        agent_id=0,
        event_name="NORMAL",
    ):
        # v3.1: 状態クラスタとしてEventを検索する。
        self.total_encodes += 1

        best = None
        best_score = 0.0

        for event in self.events:
            score = event.similarity(state)
            if score > best_score:
                best_score = score
                best = event

        if best is not None and best_score >= EVENT_SIM_THRESHOLD:
            best.reinforce(
                state, reward, error, action, event_name
            )
            return best, False

        event = EventCell(
            self.next_event_id,
            state,
            action,
            agent_id,
            event_name,
        )
        event.reward_mean = float(reward)
        event.error_mean = float(error)

        self.next_event_id += 1
        self.events.append(event)
        return event, True

    def encode_transition(
        self,
        source,
        action,
        target,
        next_state,
        reward,
        error,
    ):

        transition = self.get_transition(
            source.id,
            action,
        )

        if transition is None:

            transition = TransitionCell(
                self.next_transition_id,
                source.id,
                action,
                target.id,
                next_state,
                reward,
                error,
            )

            self.next_transition_id += 1

            self.transitions.append(
                transition
            )

            self.total_transitions += 1

            return transition, True

        transition.update(
            target.id,
            next_state,
            reward,
            error,
        )

        return transition, False

    def assign_place(self, event):

        best = None
        best_distance = 999999.0

        for place in self.places:

            distance = place.distance(
                event.state
            )

            if distance < best_distance:

                best_distance = distance
                best = place

        if (
            best is not None
            and
            best_distance < PLACE_RADIUS
        ):

            best.add_event(event)

            return best

        place = PlaceCell(
            self.next_place_id
        )

        self.next_place_id += 1

        place.add_event(event)

        self.places.append(place)

        return place

    def classify(self, event):

        # 環境イベントを明示的に保存していた場合は、それを優先。
        if event.event_name not in (None, "", "NORMAL"):
            return event.event_name

        if event.reward_mean >= 4:
            return "SUCCESS"
        if event.reward_mean <= -5:
            return "FAILURE"
        if event.error_mean >= 6:
            return "SURPRISE"
        if event.action == ACTION_JUMP:
            return "JUMP"
        if event.action == ACTION_DASH:
            return "DASH"
        if event.action == ACTION_BRAKE:
            return "BRAKE"
        if event.action == ACTION_LEFT:
            return "TURN_LEFT"
        if event.action == ACTION_RIGHT:
            return "TURN_RIGHT"
        if event.action == ACTION_WAIT:
            return "WAIT"
        if event.action == ACTION_SLIDE:
            return "SLIDE"
        if event.action == ACTION_WALL_JUMP:
            return "WALL_JUMP"
        if event.action == ACTION_GRAPPLE:
            return "GRAPPLE"
        if event.action == ACTION_RELEASE:
            return "RELEASE"
        return "MOVEMENT"

    def assign_concept(
        self,
        event,
        place,
    ):

        name = self.classify(event)

        concept = None

        for c in self.concepts:

            if c.name == name:

                concept = c
                break

        if concept is None:

            concept = ConceptCell(
                self.next_concept_id,
                name,
            )

            self.next_concept_id += 1

            self.concepts.append(concept)

        concept.absorb(event)

        if place.id not in concept.places:
            concept.places.append(place.id)

        if concept.id not in place.concepts:
            place.concepts.append(concept.id)

        return concept

    def organize(self, event):

        place = self.assign_place(event)

        concept = self.assign_concept(
            event,
            place,
        )

        return place, concept

    def link_concepts(
        self,
        previous,
        current,
    ):

        if previous is None or current is None:
            return

        for p_id in previous.concepts:

            p = self.get_concept(p_id)

            if p is None:
                continue

            for c_id in current.concepts:

                if p_id != c_id:

                    p.strengthen(c_id)

    def replay(
        self,
        replay_memory,
        count,
    ):

        batch = replay_memory.sample_batch(
            count
        )

        for item in batch:

            previous = self.get_event(
                item["previous"]
            )

            current = self.get_event(
                item["current"]
            )

            if previous is None or current is None:
                continue

            transition = self.get_transition(
                previous.id,
                item["action"],
            )

            if transition is not None:

                transition.update(
                    current.id,
                    current.state,
                    item["reward"],
                    item["error"],
                )

    def replay_concepts(self, count):

        if not self.concepts:
            return

        for _ in range(count):

            concept = random.choice(
                self.concepts
            )

            concept.energy = min(
                2.5,
                concept.energy + 0.015,
            )

            if concept.links:

                target_id = random.choice(
                    list(concept.links.keys())
                )

                concept.links[target_id] = min(
                    8.0,
                    concept.links[target_id] * 1.01
                )

    def metabolize(self):

        for event in self.events:
            event.decay()

        for transition in self.transitions:
            transition.decay()

        for place in self.places:
            place.decay()

        for concept in self.concepts:
            concept.decay()

        self.events = [
            e for e in self.events
            if (
                e.visits >= 5
                or abs(e.reward_mean) >= 3
                or e.error_mean >= 6
                or e.energy > 0.18
            )
        ]

        if len(self.events) > MAX_EVENTS:

            self.events.sort(
                key=lambda e:
                    e.visits
                    +
                    e.energy
                    +
                    abs(e.reward_mean)
                    +
                    e.error_mean * 0.25,
                reverse=True,
            )

            self.events = self.events[:MAX_EVENTS]

        self.transitions = [
            t for t in self.transitions
            if (
                t.visits >= 2
                or t.energy > 0.25
                or t.reward_mean > 1
            )
        ]

        if len(self.transitions) > MAX_TRANSITIONS:

            self.transitions.sort(
                key=lambda t:
                    t.visits
                    +
                    t.energy
                    +
                    abs(t.reward_mean)
                    +
                    t.confidence,
                reverse=True,
            )

            self.transitions = (
                self.transitions[:MAX_TRANSITIONS]
            )

    def statistics(self):

        return {
            "events": len(self.events),
            "transitions": len(self.transitions),
            "places": len(self.places),
            "concepts": len(self.concepts),
            "encodes": self.total_encodes,
        }


# ============================================================
# 11. DREAM SIMULATOR
# ============================================================

class DreamSimulator:

    def __init__(self, hippocampus):

        self.hippo = hippocampus

    def candidates(self, event):

        result = []

        for transition in self.hippo.transitions:

            if transition.source_id != event.id:
                continue

            target_id = transition.best_target()

            target = self.hippo.get_event(
                target_id
            )

            if target is not None:

                result.append(
                    (
                        transition,
                        target,
                    )
                )

        return result

    def rollout(
        self,
        start,
        steps,
    ):

        path = []
        current = start

        total_reward = 0.0

        for _ in range(steps):

            path.append(current)

            candidates = self.candidates(
                current
            )

            if not candidates:
                break

            scores = []

            for transition, target in candidates:

                score = (
                    transition.reward_mean
                    +
                    transition.confidence * 0.5
                    -
                    transition.risk() * 2.0
                    -
                    transition.error_mean * 0.1
                )

                score += (
                    1.0 /
                    math.sqrt(
                        transition.visits + 1
                    )
                )

                scores.append(score)

            probabilities = softmax(
                scores,
                temperature=0.8,
            )

            index = np.random.choice(
                len(candidates),
                p=probabilities,
            )

            transition, current = candidates[index]

            total_reward += transition.reward_mean

        return path, total_reward

    def best_dream(
        self,
        start,
        samples,
        steps,
    ):

        best_path = []
        best_score = -999999.0

        for _ in range(samples):

            path, reward = self.rollout(
                start,
                steps,
            )

            if not path:
                continue

            score = reward

            for event in path:

                score += event.energy * 0.2
                score -= event.error_mean * 0.05

            if score > best_score:

                best_score = score
                best_path = path

        return best_path, best_score


# ============================================================
# 12. SELF MODEL
# ============================================================

class SelfModel:

    def __init__(self, agent_id):

        self.agent_id = agent_id

        self.total_steps = 0

        self.total_reward = 0.0

        self.successes = 0
        self.failures = 0

        self.prediction_error = 1.0

        self.confidence = 0.2

        self.energy = 1.0

        self.action_values = np.zeros(
            NUM_ACTIONS,
            dtype=float,
        )

        self.action_visits = np.zeros(
            NUM_ACTIONS,
            dtype=float,
        )

        self.recent_errors = deque(
            maxlen=100
        )

        self.recent_rewards = deque(
            maxlen=100
        )

        self.identity = {
            "explorer": 0.0,
            "collector": 0.0,
            "survivor": 0.0,
            "controller": 0.0,
        }

    def observe(
        self,
        action,
        reward,
        error,
        event_name,
    ):

        self.total_steps += 1
        self.total_reward += reward

        self.prediction_error = (
            0.90 * self.prediction_error
            +
            0.10 * error
        )

        self.recent_errors.append(error)
        self.recent_rewards.append(reward)

        self.action_visits[action] += 1

        self.action_values[action] = (
            0.90 * self.action_values[action]
            +
            0.10 * reward
        )

        if event_name == "SUCCESS":
            self.successes += 1

        if event_name in ("DANGER", "FALL"):
            self.failures += 1

        self.confidence = clamp(
            1.0 /
            (
                1.0
                +
                self.prediction_error
            ),
            0.05,
            1.0,
        )

        self.energy = clamp(
            self.energy
            +
            reward * 0.002
            -
            0.001,
            0.1,
            2.0,
        )

    def update_identity(
        self,
        exploration,
        collection,
        survival,
        control,
    ):

        signals = {
            "explorer": exploration,
            "collector": collection,
            "survivor": survival,
            "controller": control,
        }

        for key, value in signals.items():

            self.identity[key] = (
                0.90 * self.identity[key]
                +
                0.10 * value
            )

    def dominant_role(self):

        return max(
            self.identity,
            key=self.identity.get,
        )

    def score_action(self, action):

        visits = self.action_visits[action]

        if visits <= 0:
            novelty = 1.0
        else:
            novelty = 1.0 / math.sqrt(
                visits + 1
            )

        return (
            self.action_values[action]
            +
            novelty * 0.5
            +
            self.confidence * 0.2
        )

    def consolidate(self):

        self.energy *= 0.995

        delta = (
            0.01
            if self.prediction_error < 1.0
            else -0.005
        )

        self.confidence = clamp(
            self.confidence + delta,
            0.05,
            1.0,
        )


# ============================================================
# 13. CURIOSITY ENGINE
# ============================================================

class CuriosityEngine:

    def __init__(self):

        self.novelty_weight = 1.0
        self.error_weight = 0.7
        self.information_weight = 0.8
        self.risk_penalty = 0.7

        self.exploration_history = deque(
            maxlen=200
        )

    def score(
        self,
        prediction,
        state,
        action,
        self_model,
    ):

        transition = prediction.get(
            "transition"
        )

        if transition is None:

            novelty = 1.0
            uncertainty = 1.0
            information = 1.0
            risk = 0.0

        else:

            novelty = (
                1.0 /
                math.sqrt(
                    transition.visits + 1
                )
            )

            uncertainty = clamp(
                prediction.get(
                    "uncertainty",
                    1.0,
                ),
                0.0,
                3.0,
            )

            information = (
                transition.error_mean
                /
                max(
                    1.0,
                    math.sqrt(
                        transition.visits
                    ),
                )
            )

            risk = transition.risk()

        curiosity = (
            novelty * self.novelty_weight
            +
            uncertainty * self.error_weight
            +
            information * self.information_weight
            -
            risk * self.risk_penalty
        )

        curiosity *= (
            0.7
            +
            0.3 * self_model.confidence
        )

        self.exploration_history.append(
            curiosity
        )

        return float(curiosity)

    def action_bonus(
        self,
        model,
        state,
        action,
        self_model,
    ):

        prediction = model.predict(
            state,
            action,
        )

        return self.score(
            prediction,
            state,
            action,
            self_model,
        )


# ============================================================
# 14. CAUSAL MODEL
# ============================================================

class CausalRelation:

    def __init__(
        self,
        cause,
        effect,
    ):

        self.cause = cause
        self.effect = effect

        self.count = 0

        self.success = 0
        self.failure = 0

        self.strength = 0.0

        self.mean_reward = 0.0

        self.mean_error = 0.0

    def observe(
        self,
        reward,
        error,
    ):

        self.count += 1

        if reward > 0:
            self.success += 1

        if reward < -3:
            self.failure += 1

        self.mean_reward = (
            0.9 * self.mean_reward
            +
            0.1 * reward
        )

        self.mean_error = (
            0.9 * self.mean_error
            +
            0.1 * error
        )

        frequency = min(
            1.0,
            self.count / 20.0
        )

        outcome = (
            self.success
            -
            self.failure
        ) / max(
            1,
            self.count
        )

        self.strength = clamp(
            frequency * 0.6
            +
            (outcome + 1.0) * 0.2
            +
            math.tanh(
                self.mean_reward * 0.1
            ) * 0.2,
            -1.0,
            1.0,
        )


class CausalModel:

    def __init__(self):

        self.relations = {}

        self.action_effects = defaultdict(
            lambda: defaultdict(int)
        )

        self.concept_effects = defaultdict(
            lambda: defaultdict(int)
        )

    def key(self, cause, effect):

        return (
            str(cause),
            str(effect),
        )

    def observe(
        self,
        previous_event,
        current_event,
        action,
        reward,
        error,
    ):

        if previous_event is None:
            return

        cause = (
            "ACTION",
            action,
            previous_event.place_id,
        )

        effect = (
            "EVENT",
            current_event.place_id,
            current_event.action,
        )

        key = self.key(
            cause,
            effect,
        )

        if key not in self.relations:

            self.relations[key] = CausalRelation(
                cause,
                effect,
            )

        relation = self.relations[key]

        relation.observe(
            reward,
            error,
        )

        self.action_effects[
            action
        ][
            current_event.action
        ] += 1

        for concept_id in current_event.concepts:

            self.concept_effects[
                action
            ][
                concept_id
            ] += 1

    def predict_action_effect(
        self,
        action,
    ):

        effects = self.action_effects.get(
            action,
            {}
        )

        if not effects:
            return None

        return max(
            effects,
            key=effects.get,
        )

    def causal_strength(
        self,
        action,
    ):

        values = []

        for key, relation in self.relations.items():

            cause = relation.cause

            if (
                isinstance(cause, tuple)
                and
                len(cause) > 1
                and
                cause[0] == "ACTION"
                and
                cause[1] == action
            ):

                values.append(
                    relation.strength
                )

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    def consolidate(self):

        weak = []

        for key, relation in self.relations.items():

            relation.strength *= 0.995

            if (
                relation.count < 2
                and
                abs(relation.strength) < 0.1
            ):

                weak.append(key)

        for key in weak:

            self.relations.pop(
                key,
                None,
            )

    def statistics(self):

        if not self.relations:
            return {
                "relations": 0,
                "strong": 0,
            }

        strong = sum(
            1
            for relation in self.relations.values()
            if abs(relation.strength) > 0.5
        )

        return {
            "relations": len(
                self.relations
            ),
            "strong": strong,
        }


# ============================================================
# 15. GOAL SYSTEM
# ============================================================

class Goal:

    def __init__(
        self,
        name,
        priority,
        target,
    ):

        self.name = name
        self.priority = priority
        self.target = target

        self.progress = 0.0
        self.successes = 0
        self.failures = 0

        self.energy = 1.0

    def update(
        self,
        progress,
        success=False,
        failure=False,
    ):

        self.progress = clamp(
            0.9 * self.progress
            +
            0.1 * progress,
            -1.0,
            1.0,
        )

        if success:
            self.successes += 1

        if failure:
            self.failures += 1

        self.energy = min(
            2.5,
            self.energy + 0.02
        )


class GoalSystem:

    def __init__(self):

        self.goals = {}

        self.create_default_goals()

    def create_default_goals(self):

        self.goals["SURVIVE"] = Goal(
            "SURVIVE",
            1.0,
            "avoid_failure",
        )

        self.goals["EXPLORE"] = Goal(
            "EXPLORE",
            0.7,
            "discover",
        )

        self.goals["COLLECT"] = Goal(
            "COLLECT",
            0.9,
            "orb",
        )

        self.goals["CONTROL"] = Goal(
            "CONTROL",
            0.8,
            "switch",
        )

    def generate(
        self,
        state,
        world,
        self_model,
        curiosity,
    ):

        scores = {}

        scores["SURVIVE"] = (
            self.goals["SURVIVE"].priority
            +
            self_model.failures * 0.01
        )

        scores["EXPLORE"] = (
            self.goals["EXPLORE"].priority
            +
            curiosity * 0.5
        )

        if world.orbs:

            scores["COLLECT"] = (
                self.goals["COLLECT"].priority
                +
                0.8
            )

        else:

            scores["COLLECT"] = 0.1

        if not world.bridge_active:

            scores["CONTROL"] = (
                self.goals["CONTROL"].priority
                +
                0.7
            )

        else:

            scores["CONTROL"] = 0.1

        name = max(
            scores,
            key=scores.get,
        )

        return self.goals[name]

    def update(
        self,
        goal_name,
        reward,
        event_name,
    ):

        goal = self.goals.get(
            goal_name
        )

        if goal is None:
            return

        success = (
            event_name == "SUCCESS"
            or
            (
                goal_name == "CONTROL"
                and
                event_name == "SWITCH"
            )
        )

        failure = (
            event_name in
            ("DANGER", "FALL")
        )

        progress = reward / 5.0

        goal.update(
            progress,
            success,
            failure,
        )

    def consolidate(self):

        for goal in self.goals.values():

            goal.energy *= 0.997


# ============================================================
# 16. SKILL SYSTEM
# ============================================================

class Skill:

    def __init__(
        self,
        skill_id,
        name,
        actions,
    ):

        self.id = skill_id
        self.name = name

        self.actions = list(actions)

        self.visits = 0
        self.successes = 0
        self.failures = 0

        self.value = 0.0
        self.energy = 1.0

        self.owner = None

    @property
    def success_rate(self):

        return (
            self.successes
            /
            max(1, self.visits)
        )

    def observe(
        self,
        reward,
        success,
        failure,
    ):

        self.visits += 1

        if success:
            self.successes += 1

        if failure:
            self.failures += 1

        self.value = (
            0.9 * self.value
            +
            0.1 * reward
        )

        self.energy = min(
            3.0,
            self.energy + 0.03,
        )

    def score(self):

        return (
            self.value
            +
            self.success_rate * 2.0
            +
            1.0 /
            math.sqrt(
                self.visits + 1
            )
        )


class SkillSystem:

    def __init__(self):

        self.skills = []

        self.next_skill_id = 0

        self.recent_actions = defaultdict(
            lambda: deque(
                maxlen=8
            )
        )

        self.create_builtin_skills()

    def create_skill(
        self,
        name,
        actions,
    ):

        skill = Skill(
            self.next_skill_id,
            name,
            actions,
        )

        self.next_skill_id += 1

        self.skills.append(skill)

        return skill

    def create_builtin_skills(self):

        self.create_skill(
            "MOVE_RIGHT",
            [ACTION_RIGHT],
        )

        self.create_skill(
            "MOVE_LEFT",
            [ACTION_LEFT],
        )

        self.create_skill(
            "JUMP_FORWARD",
            [
                ACTION_RIGHT,
                ACTION_JUMP,
                ACTION_RIGHT,
            ],
        )

        self.create_skill(
            "DASH_FORWARD",
            [
                ACTION_RIGHT,
                ACTION_DASH,
            ],
        )

        self.create_skill(
            "BRAKE_CONTROL",
            [
                ACTION_BRAKE,
                ACTION_RIGHT,
            ],
        )

        self.create_skill(
            "WAIT_RECOVER",
            [
                ACTION_WAIT,
                ACTION_WAIT,
            ],
        )

    def observe(
        self,
        agent_id,
        action,
        reward,
        event_name,
    ):

        history = self.recent_actions[
            agent_id
        ]

        history.append(action)

        success = event_name == "SUCCESS"

        failure = event_name in (
            "DANGER",
            "FALL",
        )

        for skill in self.skills:

            if len(history) < len(skill.actions):
                continue

            recent = list(history)[
                -len(skill.actions):
            ]

            if recent == skill.actions:

                skill.observe(
                    reward,
                    success,
                    failure,
                )

    def available(
        self,
        agent_id,
    ):

        result = []

        history = list(
            self.recent_actions[
                agent_id
            ]
        )

        for skill in self.skills:

            if not history:
                result.append(skill)
                continue

            prefix = history[
                -min(
                    len(history),
                    len(skill.actions)
                ):
            ]

            if skill.actions[
                :len(prefix)
            ] == prefix:

                result.append(skill)

        return result

    def best_skill(self):

        if not self.skills:
            return None

        return max(
            self.skills,
            key=lambda s: s.score(),
        )

    def consolidate(self):

        for skill in self.skills:

            skill.energy *= 0.997


# ============================================================
# 17. COUNTERFACTUAL DREAM
# ============================================================

class CounterfactualDream:

    def __init__(
        self,
        hippocampus,
        causal_model,
    ):

        self.hippo = hippocampus
        self.causal = causal_model

        self.last_counterfactuals = []

    def evaluate(
        self,
        state,
        action,
        model,
        alternatives=None,
    ):

        if alternatives is None:

            alternatives = list(
                range(NUM_ACTIONS)
            )

        results = []

        for candidate_action in alternatives:

            prediction = model.predict(
                state,
                candidate_action,
            )

            transition = prediction.get(
                "transition"
            )

            if transition is None:

                expected_reward = 0.0
                uncertainty = 1.0
                risk = 0.0

            else:

                expected_reward = (
                    prediction["reward"]
                )

                uncertainty = (
                    prediction["uncertainty"]
                )

                risk = transition.risk()

            causal = (
                self.causal.causal_strength(
                    candidate_action
                )
            )

            score = (
                expected_reward
                +
                causal
                +
                uncertainty * 0.2
                -
                risk * 2.0
            )

            results.append({
                "action": candidate_action,
                "reward": expected_reward,
                "uncertainty": uncertainty,
                "risk": risk,
                "causal": causal,
                "score": score,
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        self.last_counterfactuals = results

        return results


# ============================================================
# 18. REFLECTION SYSTEM
# ============================================================

class ReflectionSystem:

    def __init__(self):

        self.history = deque(
            maxlen=100
        )

        self.success_count = 0
        self.failure_count = 0

        self.lessons = []

        self.last_reflection = (
            "No reflection yet."
        )

    def reflect(
        self,
        agent,
        goal,
        counterfactuals,
    ):

        reward = agent.episode_reward

        if agent.failures > agent.successes:

            lesson = (
                "Failure dominated. "
                "Prefer safer transitions and "
                "increase prediction confidence "
                "before committing."
            )

            self.failure_count += 1

        elif reward > 10:

            lesson = (
                "Positive trajectory detected. "
                "Consolidate the actions and "
                "transitions that produced reward."
            )

            self.success_count += 1

        else:

            lesson = (
                "Mixed outcome. "
                "Increase exploration around "
                "uncertain transitions."
            )

        if counterfactuals:

            best = counterfactuals[0]

            lesson += (
                f" Counterfactual best action="
                f"{ACTION_NAMES[best['action']]}"
                f" score={best['score']:+.2f}."
            )

        self.last_reflection = lesson

        self.history.append({
            "agent": agent.id,
            "goal": goal.name,
            "reward": reward,
            "lesson": lesson,
        })

        if lesson not in self.lessons:

            self.lessons.append(lesson)

        return lesson

    def consolidate(self):

        if len(self.lessons) > 50:

            self.lessons = self.lessons[-50:]


# ============================================================
# 19. MULTI-AGENT SPECIALIZATION
# ============================================================

class MultiAgentSpecialization:

    def __init__(self):

        self.roles = {}

        self.role_scores = defaultdict(
            lambda: defaultdict(float)
        )

        self.default_roles = [
            "EXPLORER",
            "COLLECTOR",
            "SURVIVOR",
        ]

    def initialize(self, agent_id):

        self.roles[agent_id] = (
            self.default_roles[
                agent_id
                %
                len(self.default_roles)
            ]
        )

    def update(
        self,
        agent,
        curiosity,
        reward,
    ):

        aid = agent.id

        self.role_scores[
            aid
        ]["EXPLORER"] += (
            curiosity * 0.05
        )

        self.role_scores[
            aid
        ]["COLLECTOR"] += (
            max(0.0, reward) * 0.02
        )

        self.role_scores[
            aid
        ]["SURVIVOR"] += (
            max(
                0.0,
                -reward
            )
            *
            0.03
        )

        self.role_scores[
            aid
        ]["CONTROLLER"] += (
            0.01
            if agent.world.bridge_active
            else 0.0
        )

        self.roles[aid] = max(
            self.role_scores[aid],
            key=self.role_scores[aid].get,
        )

    def role_bonus(
        self,
        agent_id,
        action,
    ):

        role = self.roles.get(
            agent_id,
            "EXPLORER",
        )

        bonus = 0.0

        if (
            role == "EXPLORER"
            and
            action in (
                ACTION_JUMP,
                ACTION_DASH,
            )
        ):

            bonus += 0.5

        if (
            role == "COLLECTOR"
            and
            action == ACTION_RIGHT
        ):

            bonus += 0.3

        if (
            role == "SURVIVOR"
            and
            action in (
                ACTION_BRAKE,
                ACTION_WAIT,
            )
        ):

            bonus += 0.4

        if (
            role == "CONTROLLER"
            and
            action == ACTION_RIGHT
        ):

            bonus += 0.25

        return bonus


# ============================================================
# 20. WORLD MODEL
# ============================================================

class WorldModel:

    def __init__(
        self,
        hippocampus,
        dream,
        causal_model,
        curiosity,
        goal_system,
        skill_system,
        counterfactual,
        reflection,
        specialization,
    ):

        self.hippo = hippocampus
        self.dream = dream

        self.causal_model = causal_model
        self.curiosity = curiosity
        self.goal_system = goal_system
        self.skill_system = skill_system

        self.counterfactual = counterfactual
        self.reflection = reflection

        self.specialization = specialization

        self.replay = ReplayMemory()

        self.last_event = {}

        self.error_history = []

        self.last_dream = []
        self.last_dream_score = 0.0

        self.last_counterfactuals = []

        self.learning_rate = 0.15

        self.exploration = 1.0

        self.episode_count = 0

        self.self_models = {
            i: SelfModel(i)
            for i in range(NUM_AGENTS)
        }

        # v5.1: Action -> Physics Event -> Consequence concept memory.
        self.physics_concepts = {}
        self.next_physics_concept_id = 0
        self.physics_concept_history = deque(maxlen=600)
        self.physics_chains = {}
        self.next_physics_chain_id = 0
        self.physics_chain_history = deque(maxlen=CHAIN_MEMORY)

    # --------------------------------------------------------
    # PHYSICS CHAIN ABSTRACTION
    # --------------------------------------------------------

    def _chain_consequence(self, events, state, next_state):
        ev = set(events or [])
        s = np.asarray(state, dtype=float)
        n = np.asarray(next_state, dtype=float)
        dx = (n[0] - s[0]) * 500.0
        dy = (n[1] - s[1]) * 250.0
        dvx = (n[2] - s[2]) * MAX_SPEED
        dvy = (n[3] - s[3]) * 18.0

        if 'GRAPPLE_ATTACH' in ev and 'PENDULUM_COLLISION' in ev:
            return 'SWING_TRANSFER'
        if 'GRAPPLE_RELEASE' in ev and 'LAUNCH_RAMP' in ev:
            return 'MOMENTUM_TO_FLIGHT'
        if 'LAUNCH_RAMP' in ev and 'WIND' in ev:
            return 'WIND_ADJUSTED_FLIGHT'
        if 'WIND' in ev and ('TRAMPOLINE_BOUNCE' in ev or 'UPDRAFT' in ev):
            return 'AERIAL_LIFT_CORRECTION'
        if 'CRUMBLE_PRESSURE' in ev and 'CRUMBLE_BREAK' in ev:
            return 'SUPPORT_FAILURE'
        if 'PLATFORM_CARRY' in ev and ('LAUNCH_RAMP' in ev or 'TRAMPOLINE_BOUNCE' in ev):
            return 'MOVING_PLATFORM_MOMENTUM'
        if 'PENDULUM_COLLISION' in ev and abs(dvx) + abs(dvy) > 3.0:
            return 'PENDULUM_IMPULSE'
        if abs(dvx) > 2.5 and abs(dvy) > 2.5:
            return 'COMPOUND_VELOCITY_CHANGE'
        if abs(dx) + abs(dy) > 25.0:
            return 'COMPOUND_DISPLACEMENT'
        return 'MULTI_EVENT_STATE_CHANGE'

    def observe_physics_chain(self, action, events, state, next_state, error):
        action_name = ACTION_NAMES.get(int(action), str(action))
        raw = [str(e) for e in (events or []) if e not in ('IDLE', 'NORMAL')]
        if not raw:
            return None
        seq = list(dict.fromkeys(raw))[:CHAIN_MAX]
        # A chain should represent an interaction, not merely repeated movement.
        if len(seq) < 2 and seq[0] in {'MOVE_LEFT','MOVE_RIGHT','WAIT','BRAKE','JUMP'}:
            return None

        consequence = self._chain_consequence(seq, state, next_state)
        key = (action_name, tuple(seq), consequence)
        concept = self.physics_chains.get(key)
        if concept is None:
            concept = PhysicsChainConcept(
                self.next_physics_chain_id, action_name, seq, consequence
            )
            self.next_physics_chain_id += 1
            self.physics_chains[key] = concept
        concept.observe(error)
        self.physics_chain_history.append({
            'concept_id': concept.id,
            'action': action_name,
            'events': list(seq),
            'consequence': consequence,
            'error': float(error),
        })
        return concept

    def physics_chain_novelty(self, action):
        action_name = ACTION_NAMES.get(int(action), str(action))
        visits = sum(c.visits for c in self.physics_chains.values() if c.action == action_name)
        return 1.0 / math.sqrt(visits + 1.0)

    # --------------------------------------------------------
    # PHYSICS CONCEPT ABSTRACTION
    # --------------------------------------------------------

    def _quantize(self, value, small=0.15, large=0.8):
        if value > large:
            return "HIGH_POS"
        if value > small:
            return "POS"
        if value < -large:
            return "HIGH_NEG"
        if value < -small:
            return "NEG"
        return "STABLE"

    def _consequence(self, action, events, state, next_state):
        s = np.asarray(state, dtype=float)
        n = np.asarray(next_state, dtype=float)
        dx = (n[0] - s[0]) * 500.0
        dy = (n[1] - s[1]) * 250.0
        dvx = (n[2] - s[2]) * MAX_SPEED
        dvy = (n[3] - s[3]) * 18.0

        ev = set(events or [])
        if "TRAMPOLINE_BOUNCE" in ev:
            return "VERTICAL_BOUNCE"
        if "PENDULUM_COLLISION" in ev:
            return "EXTERNAL_IMPULSE"
        if "WALL_COLLISION" in ev or "BOUNDARY_WALL" in ev:
            return "HORIZONTAL_REFLECTION"
        if "PLATFORM_CARRY" in ev:
            return "CARRIED_BY_SURFACE"
        if "WIND" in ev:
            if abs(dx) > 1.0:
                return "AIR_DRIFT" + ("_RIGHT" if dx > 0 else "_LEFT")
            return "AIR_DRIFT"
        if "SLOPE_FORCE" in ev:
            return "SLOPE_ACCELERATION"
        if "CONVEYOR" in ev:
            return "SURFACE_TRANSLATION"
        if "UPDRAFT" in ev:
            return "VERTICAL_LIFT"
        if "ROTATING_BEAM_HIT" in ev:
            return "ANGULAR_IMPULSE"
        if "LAUNCH_RAMP" in ev:
            return "LAUNCH_ACCELERATION"
        if "CRUMBLE_BREAK" in ev:
            return "SUPPORT_REMOVAL"
        if "WALL_JUMP" in ev:
            return "WALL_PROPULSION"
        if "GRAPPLE_PULL" in ev or "GRAPPLE_ATTACH" in ev:
            return "TOWARD_ANCHOR"
        if "SLIP" in ev:
            return "LOW_FRICTION"
        if "FRICTION" in ev:
            return "HIGH_FRICTION"
        if "FALL" in ev:
            return "LOSS_OF_SUPPORT"
        if "LAND" in ev:
            return "VERTICAL_STABILIZATION"
        if abs(dvy) > 2.0:
            return "VERTICAL_ACCELERATION"
        if abs(dvx) > 1.0:
            return "HORIZONTAL_ACCELERATION"
        if abs(dx) > 1.0 or abs(dy) > 1.0:
            return "BODY_DISPLACEMENT"
        return "STATE_PERSISTENCE"

    def observe_physics_consequence(self, action, events, state, next_state, error):
        action_name = ACTION_NAMES.get(int(action), str(action))
        event_name = (events[-1] if events else "NORMAL")
        consequence = self._consequence(action, events, state, next_state)
        key = (action_name, event_name, consequence)

        concept = self.physics_concepts.get(key)
        if concept is None:
            concept = PhysicsConcept(
                self.next_physics_concept_id,
                action_name,
                event_name,
                consequence,
            )
            self.next_physics_concept_id += 1
            self.physics_concepts[key] = concept

        s = np.asarray(state, dtype=float)
        n = np.asarray(next_state, dtype=float)
        dx = (n[0] - s[0]) * 500.0
        dy = (n[1] - s[1]) * 250.0
        dvx = (n[2] - s[2]) * MAX_SPEED
        dvy = (n[3] - s[3]) * 18.0
        concept.observe(dx, dy, dvx, dvy, error)

        self.physics_concept_history.append({
            "concept_id": concept.id,
            "action": action_name,
            "events": list(events or []),
            "consequence": consequence,
            "error": float(error),
        })
        return concept

    def physics_novelty(self, action, state):
        action_name = ACTION_NAMES.get(int(action), str(action))
        visits = sum(c.visits for c in self.physics_concepts.values() if c.action == action_name)
        return 1.0 / math.sqrt(visits + 1.0)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(
        self,
        state,
        action,
    ):
        # v3.1: state_t + action_t -> predicted state_t+1
        best_event = None
        best_score = 0.0

        for event in self.hippo.events:
            score = event.similarity(state)
            if score > best_score:
                best_score = score
                best_event = event

        if best_event is None or best_score < EVENT_SIM_THRESHOLD:
            return {
                "next_state": None,
                "reward": 0.0,
                "uncertainty": 1.0,
                "event": best_event,
                "transition": None,
                "similarity": best_score,
            }

        transition = self.hippo.get_transition(
            best_event.id, action
        )

        if transition is None:
            return {
                "next_state": None,
                "reward": 0.0,
                "uncertainty": 1.0,
                "event": best_event,
                "transition": None,
                "similarity": best_score,
            }

        uncertainty = (
            transition.error_mean
            / (1.0 + transition.visits)
        )

        return {
            "next_state": transition.predicted_next_state.copy(),
            "reward": transition.reward_mean,
            "uncertainty": uncertainty,
            "event": best_event,
            "transition": transition,
            "similarity": best_score,
        }

    # --------------------------------------------------------
    # LEARN
    # --------------------------------------------------------

    def learn(
        self,
        agent_id,
        state,
        action,
        next_state,
        reward,
        event_name="NORMAL",
        events=None,
    ):
        # v3.1: 1経験 = state_t --action_t--> next_state_t+1
        # 現在状態はlast_eventで保持し、予測は必ず現在stateから行う。

        source_event = self.last_event.get(agent_id)

        # 初回、または現在状態と記憶状態が離れた場合は再エンコード。
        if (
            source_event is None
            or source_event.similarity(state) < EVENT_SIM_THRESHOLD
        ):
            source_event, source_created = self.hippo.encode_event(
                state,
                ACTION_NONE,
                0.0,
                1.0,
                agent_id,
                "NORMAL",
            )
        else:
            source_created = False

        predicted = self.predict(state, action)

        if predicted["next_state"] is None:
            error = float(np.linalg.norm(
                np.asarray(next_state, dtype=float)
            ))
        else:
            error = float(np.linalg.norm(
                predicted["next_state"]
                - np.asarray(next_state, dtype=float)
            ))

        # 実際に到達したstateを次の状態ノードとして記憶する。
        target_event, target_created = self.hippo.encode_event(
            next_state,
            ACTION_NONE,
            reward,
            error,
            agent_id,
            event_name,
        )

        transition, transition_created = self.hippo.encode_transition(
            source_event,
            action,
            target_event,
            next_state,
            reward,
            error,
        )

        self.hippo.link_concepts(source_event, target_event)

        self.replay.add(
            source_event.id,
            target_event.id,
            action,
            reward,
            error,
            agent_id,
        )

        self.error_history.append(error)

        # New abstraction layer: action -> observed physics events -> consequence.
        observed_events = list(events or [event_name])
        physics_concept = self.observe_physics_consequence(
            action,
            observed_events,
            state,
            next_state,
            error,
        )
        physics_chain = self.observe_physics_chain(
            action,
            observed_events,
            state,
            next_state,
            error,
        )

        self.causal_model.observe(
            source_event,
            target_event,
            action,
            reward,
            error,
        )

        source_place, source_concept = self.hippo.organize(source_event)
        target_place, target_concept = self.hippo.organize(target_event)

        self.last_event[agent_id] = target_event

        if len(self.error_history) > MAX_ERROR_HISTORY:
            self.error_history.pop(0)

        return {
            "event": target_event,
            "previous_event": source_event,
            "next_event": target_event,
            "place": target_place,
            "concept": target_concept,
            "previous_place": source_place,
            "previous_concept": source_concept,
            "created": source_created or target_created,
            "source_created": source_created,
            "target_created": target_created,
            "transition_created": transition_created,
            "physics_concept": physics_concept,
            "physics_chain": physics_chain,
            "error": error,
        }

    # --------------------------------------------------------
    # ACTION SELECTION
    # --------------------------------------------------------

    def select_action(
        self,
        agent_id,
        state,
        world,
    ):

        self_model = self.self_models[
            agent_id
        ]

        scores = []

        # v5: goals do not drive action selection.
        # Exploration is driven by novelty / uncertainty / prediction error.
        goal = None

        for action in range(NUM_ACTIONS):

            prediction = self.predict(
                state,
                action,
            )

            transition = prediction[
                "transition"
            ]

            if transition is None:

                # Unknown transitions are selected because they are unknown,
                # not because a hand-coded action is considered "interesting".
                score = (
                    2.5
                    *
                    self.exploration
                )

            else:

                reward = prediction["reward"]

                uncertainty = (
                    prediction["uncertainty"]
                )

                confidence = (
                    transition.confidence
                )

                risk = transition.risk()

                novelty = (
                    1.0 /
                    math.sqrt(
                        transition.visits + 1
                    )
                )

                score = (
                    uncertainty * 1.25
                    +
                    novelty * 1.05
                    +
                    confidence * 0.25
                    -
                    risk * 1.2
                )

            # Prefer actions whose physical consequences are still poorly known.
            score += self.physics_novelty(action, state) * 1.15
            score += self.physics_chain_novelty(action) * 0.95

            curiosity_bonus = (
                self.curiosity.action_bonus(
                    self,
                    state,
                    action,
                    self_model,
                )
            )

            score += (
                curiosity_bonus
                *
                0.55
            )

            planning_bonus = (
                self.evaluate_action_sequence(
                    state,
                    action,
                )
            )

            score += (
                planning_bonus
                *
                0.65
            )

            # v5: specialization remains available as a learned statistic,
            # but it does not prescribe actions.

            score += (
                random.random()
                *
                0.35
                *
                self.exploration
            )

            score += (
                self_model.score_action(
                    action
                )
                *
                0.15
            )

            scores.append(score)

        return int(
            np.argmax(scores)
        )

    # --------------------------------------------------------
    # PLANNING
    # --------------------------------------------------------

    def evaluate_action_sequence(
        self,
        state,
        first_action,
    ):

        prediction = self.predict(
            state,
            first_action,
        )

        transition = prediction[
            "transition"
        ]

        if transition is None:
            return 0.0

        score = prediction["reward"]

        current_id = (
            transition.best_target()
        )

        current = (
            self.hippo.get_event(
                current_id
            )
        )

        if current is None:
            return score

        for depth in range(
            PLANNING_DEPTH - 1
        ):

            candidates = []

            for t in self.hippo.transitions:

                if t.source_id == current.id:

                    candidates.append(t)

            if not candidates:
                break

            best = max(
                candidates,
                key=lambda t:
                    t.reward_mean
                    -
                    t.risk() * 2.0
                    +
                    t.confidence * 0.5
                    +
                    1.0 /
                    math.sqrt(
                        t.visits + 1
                    ),
            )

            score += (
                best.reward_mean
                *
                (0.8 ** depth)
            )

            target_id = (
                best.best_target()
            )

            current = (
                self.hippo.get_event(
                    target_id
                )
            )

            if current is None:
                break

        return score

    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    def sleep(self):

        if not self.hippo.events:

            self.last_dream = []

            return

        self.hippo.replay(
            self.replay,
            REPLAY_COUNT,
        )

        self.hippo.replay_concepts(
            CONCEPT_REPLAY_COUNT,
        )

        self.skill_system.consolidate()

        self.causal_model.consolidate()

        self.goal_system.consolidate()

        self.reflection.consolidate()

        # ----------------------------------------------------
        # DREAM
        # ----------------------------------------------------

        ranked = sorted(
            self.hippo.events,
            key=lambda e:
                e.energy
                +
                e.visits * 0.25
                +
                e.error_mean * 0.15
                +
                abs(e.reward_mean) * 0.25,
            reverse=True,
        )

        best_path = []
        best_score = -999999.0

        for start in ranked[
            :DREAM_SOURCES
        ]:

            path, score = (
                self.dream.best_dream(
                    start,
                    DREAM_SAMPLES,
                    DREAM_STEPS,
                )
            )

            if (
                path
                and
                score > best_score
            ):

                best_path = path
                best_score = score

        self.last_dream = best_path
        self.last_dream_score = best_score

        # ----------------------------------------------------
        # DREAM REINFORCEMENT
        # ----------------------------------------------------

        for i in range(
            len(best_path) - 1
        ):

            current = best_path[i]

            transition = (
                self.hippo.get_transition(
                    current.id,
                    current.action,
                )
            )

            if transition is not None:

                transition.energy = min(
                    3.0,
                    transition.energy + 0.05,
                )

                transition.confidence = min(
                    1.0,
                    transition.confidence + 0.01,
                )

        # ----------------------------------------------------
        # SELF CONSOLIDATION
        # ----------------------------------------------------

        for self_model in self.self_models.values():

            self_model.consolidate()

        # ----------------------------------------------------
        # METABOLISM
        # ----------------------------------------------------

        self.hippo.metabolize()

        for chain in self.physics_chains.values():
            chain.decay()

        self.exploration *= 0.92

        self.exploration = max(
            0.18,
            self.exploration,
        )

        self.last_event.clear()

        self.episode_count += 1


# ============================================================
# 21. TURTLE WORLD v5
# ============================================================
#
# v5 DESIGN:
#   - one large experimental arena
#   - no predefined reward
#   - no goal-dependent world mechanics
#   - terrain + dynamic objects + forces coexist
#   - events describe physics, not value judgements
#
# The brain learns from:
#   state -> action -> physical consequence -> prediction error
#

class TurtleWorld:

    def __init__(self):

        self.left = -520
        self.right = -20
        self.bottom = -280
        self.top = 270

        # --------------------------------------------------------
        # STATIC TERRAIN
        # --------------------------------------------------------
        self.platforms = [
            [-515, -420, -190],
            [-405, -315, -125],
            [-300, -235, -55],
            [-205, -115, 40],
            [-85, -25, 115],
        ]

        # Slopes are line segments: x1, x2, y1, y2
        self.slopes = [
            [-420, -315, -190, -125],
            [-315, -235, -125, -55],
            [-115, -25, 40, 115],
            [-205, -115, 40, -10],
        ]

        # Holes remove the lower floor in selected areas.
        self.holes = [
            [-350, -315],
            [-235, -205],
            [-115, -85],
            [-25, 15],
        ]

        # Vertical walls: x, y1, y2, restitution
        self.walls = [
            [-365, -185, -95, 0.45],
            [-232, -95, 15, 0.35],
            [-118, -5, 105, 0.45],
            [-52, 90, 175, 0.35],
        ]

        # --------------------------------------------------------
        # DYNAMIC OBJECTS
        # --------------------------------------------------------
        self.moving_platforms = [
            {
                "base_x": -260.0,
                "base_y": 145.0,
                "axis": "x",
                "amplitude": 75.0,
                "speed": 0.065,
                "width": 72.0,
            },
            {
                "base_x": -150.0,
                "base_y": 210.0,
                "axis": "y",
                "amplitude": 42.0,
                "speed": 0.045,
                "width": 68.0,
            },
        ]

        self.trampolines = [
            {"x1": -430, "x2": -390, "y": -90, "power": 16.0},
            {"x1": -180, "x2": -140, "y": 20, "power": 18.0},
            {"x1": -80, "x2": -35, "y": 145, "power": 20.0},
        ]

        self.pendulums = [
            {
                "px": -275.0,
                "py": 215.0,
                "length": 85.0,
                "angle0": 0.55,
                "omega": 0.085,
                "radius": 14.0,
            },
            {
                "px": -75.0,
                "py": 220.0,
                "length": 62.0,
                "angle0": -0.45,
                "omega": 0.11,
                "radius": 12.0,
            },
        ]

        # Wind zones: x1, x2, y1, y2, vx, vy
        self.wind_zones = [
            {"x1": -340, "x2": -245, "y1": -55, "y2": 90, "vx": 0.22, "vy": 0.00},
            {"x1": -190, "x2": -100, "y1": 60, "y2": 175, "vx": -0.18, "vy": 0.05},
            {"x1": -95, "x2": -25, "y1": 120, "y2": 250, "vx": 0.12, "vy": 0.02},
        ]

        # --------------------------------------------------------
        # EXTRA PARKOUR MACHINES
        # --------------------------------------------------------
        self.conveyors = [
            {"x1": -500, "x2": -440, "y": -190, "speed": 1.25},
            {"x1": -305, "x2": -250, "y": -55, "speed": -1.40},
            {"x1": -210, "x2": -155, "y": 40, "speed": 1.65},
            {"x1": -80, "x2": -25, "y": 115, "speed": -1.20},
        ]

        self.updraft_zones = [
            {"x1": -470, "x2": -410, "y1": -120, "y2": 60, "vy": 0.55},
            {"x1": -270, "x2": -220, "y1": -20, "y2": 160, "vy": 0.70},
            {"x1": -120, "x2": -70, "y1": 70, "y2": 245, "vy": 0.80},
        ]

        self.rotating_beams = [
            {"px": -430.0, "py": 5.0, "length": 72.0, "speed": 0.075, "phase": 0.2, "thickness": 8.0},
            {"px": -170.0, "py": 135.0, "length": 78.0, "speed": -0.09, "phase": 1.4, "thickness": 8.0},
            {"px": -45.0, "py": 215.0, "length": 62.0, "speed": 0.11, "phase": -0.8, "thickness": 7.0},
        ]

        self.seesaws = [
            {"px": -365.0, "py": -55.0, "length": 74.0, "angle0": -0.20},
            {"px": -205.0, "py": 100.0, "length": 70.0, "angle0": 0.18},
        ]

        self.launch_ramps = [
            {"x1": -455, "x2": -420, "y": -120, "boost_x": 7.0, "boost_y": 8.5},
            {"x1": -280, "x2": -245, "y": -5, "boost_x": 8.5, "boost_y": 9.0},
            {"x1": -135, "x2": -95, "y": 100, "boost_x": 7.0, "boost_y": 10.0},
        ]

        self.crumble_platforms = [
            {"x1": -385, "x2": -350, "y": -35, "stability": 0.0, "broken": False, "respawn": 0},
            {"x1": -245, "x2": -205, "y": 80, "stability": 0.0, "broken": False, "respawn": 0},
            {"x1": -55, "x2": -15, "y": 185, "stability": 0.0, "broken": False, "respawn": 0},
        ]

        self.grapple_points = [
            (-470.0, 100.0), (-335.0, 150.0), (-185.0, 220.0), (-65.0, 255.0)
        ]

        # v5: coupled machinery. These are not goals. They are interacting
        # physical components whose effects propagate into one another.
        self.air_cannons = [
            {"x": -300.0, "y": 40.0, "vx": 2.8, "vy": 1.0, "radius": 55.0},
            {"x": -95.0, "y": 170.0, "vx": -2.4, "vy": 0.5, "radius": 48.0},
        ]
        self.spring_bridges = [
            {"x1": -330.0, "x2": -280.0, "y": 10.0, "energy": 0.0, "cooldown": 0},
            {"x1": -145.0, "x2": -105.0, "y": 155.0, "energy": 0.0, "cooldown": 0},
        ]
        self.chain_context = deque(maxlen=8)
        self.chain_event_counts = defaultdict(int)
        self.last_chain = []

        # Compatibility fields kept for the existing brain/UI.
        self.orbs = []
        self.switch_x = -9999.0
        self.switch_y = -9999.0
        self.bridge_active = False
        self.moving_platform = self.moving_platforms[0]
        self.hazards = []
        self.checkpoint = (-490.0, -200.0)

        self.time = 0.0
        self.last_events = []
        self.event_counts = defaultdict(int)
        self.last_event_details = []
        self.last_surface_kind = "GROUND"

        # Additional physical regions: smooth slope, slippery strip, rough strip.
        self.slippery_zones = [(-405.0, -360.0, -205.0, -120.0), (-160.0, -120.0, 20.0, 100.0)]
        self.rough_zones = [(-300.0, -265.0, -125.0, -50.0), (-90.0, -55.0, 100.0, 160.0)]

        self.drawer = turtle.Turtle()
        self.drawer.hideturtle()
        self.drawer.penup()
        self.drawer.speed(0)

    # --------------------------------------------------------
    # RESET / DYNAMICS
    # --------------------------------------------------------

    def reset(self):
        self.time = 0.0
        self.last_events = []
        self.event_counts = defaultdict(int)
        self.last_event_details = []
        self.last_surface_kind = "GROUND"
        self.chain_context.clear()
        self.chain_event_counts = defaultdict(int)
        self.last_chain = []
        for bridge in self.spring_bridges:
            bridge["energy"] = 0.0
            bridge["cooldown"] = 0
        self.bridge_active = False
        self.orbs = []
        for cp in self.crumble_platforms:
            cp["stability"] = 0.0
            cp["broken"] = False
            cp["respawn"] = 0
        self.draw()

    def update(self):
        self.time += 1.0

    def moving_platform_state(self, index):
        mp = self.moving_platforms[index]
        phase = self.time * mp["speed"]
        if mp["axis"] == "x":
            return {
                "x": mp["base_x"] + mp["amplitude"] * math.sin(phase),
                "y": mp["base_y"],
                "width": mp["width"],
            }
        return {
            "x": mp["base_x"],
            "y": mp["base_y"] + mp["amplitude"] * math.sin(phase),
            "width": mp["width"],
        }

    def moving_x(self):
        return self.moving_platform_state(0)["x"]

    def moving_y(self):
        return self.moving_platform_state(1)["y"]

    def pendulum_state(self, index):
        p = self.pendulums[index]
        angle = p["angle0"] * math.cos(self.time * p["omega"])
        bx = p["px"] + p["length"] * math.sin(angle)
        by = p["py"] - p["length"] * math.cos(angle)
        return {
            "px": p["px"],
            "py": p["py"],
            "bx": bx,
            "by": by,
            "radius": p["radius"],
        }

    def rotating_beam_state(self, index):
        b = self.rotating_beams[index]
        angle = b["phase"] + self.time * b["speed"]
        dx = math.cos(angle) * b["length"]
        dy = math.sin(angle) * b["length"]
        return {"px": b["px"], "py": b["py"], "x2": b["px"] + dx, "y2": b["py"] + dy, "angle": angle, "thickness": b["thickness"]}

    def seesaw_state(self, index):
        s = self.seesaws[index]
        # Small quasi-static response that slowly depends on world time.
        angle = s["angle0"] + 0.12 * math.sin(self.time * 0.07 + index)
        dx = math.cos(angle) * s["length"]
        dy = math.sin(angle) * s["length"]
        return {"px": s["px"], "py": s["py"], "x1": s["px"] - dx, "y1": s["py"] - dy, "x2": s["px"] + dx, "y2": s["py"] + dy, "angle": angle}

    # --------------------------------------------------------
    # TERRAIN QUERIES
    # --------------------------------------------------------

    def in_hole(self, x):
        return any(x1 <= x <= x2 for x1, x2 in self.holes)

    def ground_height(self, x):
        if self.in_hole(x):
            return None
        return -220.0

    def slope_height(self, x):
        for x1, x2, y1, y2 in self.slopes:
            if x1 <= x <= x2:
                t = (x - x1) / max(1e-9, x2 - x1)
                return y1 + t * (y2 - y1)
        return None

    def slope_gradient(self, x):
        for x1, x2, y1, y2 in self.slopes:
            if x1 <= x <= x2:
                return (y2 - y1) / max(1.0, x2 - x1)
        return 0.0

    def surface_kind(self, x, y):
        for x1, x2, y1, y2 in self.slippery_zones:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return "SLIPPERY"
        for x1, x2, y1, y2 in self.rough_zones:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return "ROUGH"
        if self.slope_height(x) is not None:
            return "SLOPE"
        if self.in_hole(x):
            return "HOLE"
        return "GROUND"

    def wall_hit(self, x, y, radius=10.0):
        for wx, y1, y2, restitution in self.walls:
            if y1 - radius <= y <= y2 + radius and abs(x - wx) <= radius:
                return wx, restitution
        return None

    def platform_at(self, x, y):
        # Horizontal platforms.
        candidates = []
        for x1, x2, py in self.platforms:
            if x1 <= x <= x2 and abs(y - py) < 24:
                candidates.append(py)

        # Slopes.
        sy = self.slope_height(x)
        if sy is not None and abs(y - sy) < 26:
            candidates.append(sy)

        # Dynamic platforms.
        for i in range(len(self.moving_platforms)):
            state = self.moving_platform_state(i)
            if abs(x - state["x"]) <= state["width"] / 2 and abs(y - state["y"]) < 24:
                candidates.append(state["y"])

        # Choose highest support below / near the agent.
        if not candidates:
            return None
        return min(candidates, key=lambda py: abs(y - py))

    def surface_height(self, x, previous_y=None):
        candidates = []
        ground = self.ground_height(x)
        if ground is not None:
            candidates.append(ground)
        sy = self.slope_height(x)
        if sy is not None:
            candidates.append(sy)
        for i in range(len(self.moving_platforms)):
            state = self.moving_platform_state(i)
            if abs(x - state["x"]) <= state["width"] / 2:
                candidates.append(state["y"])
        for cp in self.crumble_platforms:
            if (not cp["broken"]) and cp["x1"] <= x <= cp["x2"]:
                candidates.append(cp["y"])
        if not candidates:
            return None
        if previous_y is None:
            return max(candidates)
        below = [v for v in candidates if v <= previous_y + 22]
        return max(below) if below else max(candidates)

    def near_switch(self, x, y):
        return False

    def collect_orb(self, x, y):
        return 0

    def is_hazard(self, x, y):
        # Compatibility only. Hazards are physical events in v5.
        return False

    # --------------------------------------------------------
    # FORCES / OBJECTS
    # --------------------------------------------------------

    def wind_at(self, x, y):
        wx = 0.0
        wy = 0.0
        active = []
        for i, zone in enumerate(self.wind_zones):
            if zone["x1"] <= x <= zone["x2"] and zone["y1"] <= y <= zone["y2"]:
                wx += zone["vx"]
                wy += zone["vy"]
                active.append(i)
        return wx, wy, active

    def trampoline_at(self, x, y):
        for i, pad in enumerate(self.trampolines):
            if pad["x1"] <= x <= pad["x2"] and abs(y - pad["y"]) < 25:
                return i, pad
        return None

    def pendulum_collision(self, x, y, radius=12.0):
        for i in range(len(self.pendulums)):
            p = self.pendulum_state(i)
            d = math.hypot(x - p["bx"], y - p["by"])
            if d <= p["radius"] + radius:
                return i, p
        return None


    def rotating_beam_collision(self, x, y, radius=11.0):
        for i in range(len(self.rotating_beams)):
            b = self.rotating_beam_state(i)
            # Segment distance.
            ax, ay = b["px"], b["py"]
            bx, by = b["x2"], b["y2"]
            vx, vy = bx - ax, by - ay
            wx, wy = x - ax, y - ay
            denom = vx * vx + vy * vy
            t = 0.0 if denom <= 1e-9 else clamp((wx * vx + wy * vy) / denom, 0.0, 1.0)
            cx, cy = ax + t * vx, ay + t * vy
            if math.hypot(x - cx, y - cy) <= radius + b["thickness"]:
                return i, b, cx, cy
        return None

    def conveyor_at(self, x, y):
        for i, c in enumerate(self.conveyors):
            if c["x1"] <= x <= c["x2"] and abs(y - c["y"]) < 24:
                return i, c
        return None

    def updraft_at(self, x, y):
        ay = 0.0
        active = []
        for i, zone in enumerate(self.updraft_zones):
            if zone["x1"] <= x <= zone["x2"] and zone["y1"] <= y <= zone["y2"]:
                ay += zone["vy"]
                active.append(i)
        return ay, active

    def grapple_target(self, x, y):
        best = None
        best_d = GRAPPLE_RANGE
        for i, (gx, gy) in enumerate(self.grapple_points):
            d = math.hypot(x - gx, y - gy)
            if d < best_d:
                best_d = d
                best = (i, gx, gy, d)
        return best

    # --------------------------------------------------------
    # COUPLED PHYSICS
    # --------------------------------------------------------

    def pendulum_velocity(self, index):
        p = self.pendulums[index]
        prev_t = max(0.0, self.time - 1.0)
        a0 = p["angle0"] * math.cos(prev_t * p["omega"])
        a1 = p["angle0"] * math.cos(self.time * p["omega"])
        p0x = p["px"] + p["length"] * math.sin(a0)
        p0y = p["py"] - p["length"] * math.cos(a0)
        p1x = p["px"] + p["length"] * math.sin(a1)
        p1y = p["py"] - p["length"] * math.cos(a1)
        return p1x - p0x, p1y - p0y

    def dynamic_grapple_target(self, x, y):
        target = self.grapple_target(x, y)
        best = None
        best_d = GRAPPLE_RANGE if target is None else target[3]
        if target is not None:
            idx, gx, gy, dist = target
            best = ("ANCHOR", idx, gx, gy, dist)
        for i in range(len(self.pendulums)):
            p = self.pendulum_state(i)
            d = math.hypot(x - p["bx"], y - p["by"])
            if d < best_d:
                best_d = d
                best = ("PENDULUM", i, p["bx"], p["by"], d)
        for i in range(len(self.moving_platforms)):
            st = self.moving_platform_state(i)
            d = math.hypot(x - st["x"], y - st["y"])
            if d < best_d and d < GRAPPLE_RANGE * 0.85:
                best_d = d
                best = ("PLATFORM", i, st["x"], st["y"], d)
        return best

    def surface_velocity(self, x, y):
        cv = self.conveyor_at(x, y)
        if cv is not None:
            return cv[1]["speed"], 0.0
        for i in range(len(self.moving_platforms)):
            st = self.moving_platform_state(i)
            if abs(x - st["x"]) <= st["width"] / 2 and abs(y - st["y"]) < 24:
                mp = self.moving_platforms[i]
                phase = self.time * mp["speed"]
                vx = mp["amplitude"] * mp["speed"] * math.cos(phase) if mp["axis"] == "x" else 0.0
                vy = mp["amplitude"] * mp["speed"] * math.cos(phase) if mp["axis"] == "y" else 0.0
                return vx, vy
        return 0.0, 0.0

    def spring_bridge_at(self, x, y):
        for i, b in enumerate(self.spring_bridges):
            if b["x1"] <= x <= b["x2"] and abs(y - b["y"]) < 24:
                return i, b
        return None

    # --------------------------------------------------------
    # PHYSICS STEP
    # --------------------------------------------------------

    def step(self, agent, action):

        x = float(agent.x)
        y = float(agent.y)
        vx = float(agent.vx)
        vy = float(agent.vy)
        grounded = bool(agent.grounded)
        jumps = int(agent.jumps)
        dash_timer = int(agent.dash_timer)

        reward = 0.0
        events = []

        # ----------------------------------------------------
        # ACTION LAYER
        # ----------------------------------------------------
        if action == ACTION_LEFT:
            vx -= GROUND_ACCEL if grounded else AIR_ACCEL
            events.append('MOVE_LEFT')
        elif action == ACTION_RIGHT:
            vx += GROUND_ACCEL if grounded else AIR_ACCEL
            events.append('MOVE_RIGHT')
        elif action == ACTION_JUMP:
            if grounded:
                vy = JUMP_POWER
                grounded = False
                jumps = 1
                events.append('JUMP')
            elif jumps < MAX_JUMPS:
                vy = DOUBLE_JUMP_POWER
                jumps += 1
                events.append('DOUBLE_JUMP')
            else:
                events.append('JUMP_NO_EFFECT')
        elif action == ACTION_DASH:
            if dash_timer <= 0:
                direction = 1 if vx >= 0 else -1
                if abs(vx) > 0.5:
                    direction = 1 if vx > 0 else -1
                vx = direction * DASH_SPEED
                dash_timer = DASH_DURATION
                events.append('DASH')
        elif action == ACTION_BRAKE:
            vx *= 0.30
            events.append('BRAKE')
        elif action == ACTION_WAIT:
            vx *= 0.90
            events.append('WAIT')
        elif action == ACTION_SLIDE:
            vx *= SLIDE_FRICTION
            agent.crouching = True
            events.append('SLIDE')
        elif action == ACTION_WALL_JUMP:
            wall = self.wall_hit(x, y, 14.0)
            if (wall is not None or abs(x - self.left) < 18 or abs(x - self.right) < 18) and not grounded:
                if wall is not None:
                    wx, _ = wall
                    push = -1.0 if x > wx else 1.0
                else:
                    push = 1.0 if x <= self.left + 20 else -1.0
                vx = push * 7.0
                vy = WALL_JUMP_POWER
                jumps = 1
                events.append('WALL_JUMP')
            else:
                events.append('WALL_JUMP_NO_EFFECT')
        elif action == ACTION_GRAPPLE:
            target = self.dynamic_grapple_target(x, y)
            if target is not None:
                kind, idx, gx, gy, dist = target
                dx, dy = gx - x, gy - y
                norm = max(1e-6, math.hypot(dx, dy))
                vx += (dx / norm) * GRAPPLE_PULL
                vy += (dy / norm) * GRAPPLE_PULL
                agent.grapple_target = (kind, idx)
                agent.grapple_timer = 24
                agent.grapple_length = max(25.0, dist)
                grounded = False
                events.append('GRAPPLE_ATTACH')
                events.append('GRAPPLE_TO_' + kind)
            else:
                events.append('GRAPPLE_MISS')
        elif action == ACTION_RELEASE:
            if agent.grapple_timer > 0:
                events.append('GRAPPLE_RELEASE')
            agent.grapple_timer = 0
            agent.grapple_target = None
        else:
            events.append('IDLE')

        # ----------------------------------------------------
        # GRAPPLE / SWING CONSTRAINT
        # ----------------------------------------------------
        if agent.grapple_timer > 0 and agent.grapple_target is not None:
            kind, idx = agent.grapple_target
            if kind == 'ANCHOR':
                gx, gy = self.grapple_points[idx]
                target_vx, target_vy = 0.0, 0.0
            elif kind == 'PENDULUM':
                p = self.pendulum_state(idx)
                gx, gy = p['bx'], p['by']
                target_vx, target_vy = self.pendulum_velocity(idx)
            else:
                st = self.moving_platform_state(idx)
                gx, gy = st['x'], st['y']
                target_vx, target_vy = self.surface_velocity(gx, gy)

            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)
            if dist > agent.grapple_length:
                nx, ny = dx / max(dist, 1e-6), dy / max(dist, 1e-6)
                radial_v = vx * nx + vy * ny
                correction = min(5.0, (dist - agent.grapple_length) * 0.12)
                vx += nx * correction - nx * radial_v * 0.45
                vy += ny * correction - ny * radial_v * 0.45
                vx += target_vx * SWING_INHERIT
                vy += target_vy * SWING_INHERIT
                events.append('GRAPPLE_TENSION')
            else:
                vx = vx * GRAPPLE_DAMP + target_vx * (1.0 - GRAPPLE_DAMP) * SWING_INHERIT
                vy = vy * GRAPPLE_DAMP + target_vy * (1.0 - GRAPPLE_DAMP) * SWING_INHERIT
                events.append('GRAPPLE_SWING')
            agent.grapple_timer -= 1
            if agent.grapple_timer <= 0:
                agent.grapple_timer = 0
                agent.grapple_target = None
                events.append('GRAPPLE_AUTO_RELEASE')

        # ----------------------------------------------------
        # WORLD FORCES
        # ----------------------------------------------------
        wind_x, wind_y, active_wind = self.wind_at(x, y)
        if active_wind:
            scale = AIR_WIND_SCALE if not grounded else 1.0
            # Wind gets slightly stronger near the moving machinery, so the
            # same jump can have different outcomes as the world evolves.
            machinery_proximity = 0.0
            for i in range(len(self.moving_platforms)):
                st = self.moving_platform_state(i)
                d = math.hypot(x - st['x'], y - st['y'])
                machinery_proximity = max(machinery_proximity, max(0.0, 1.0 - d / 160.0))
            wind_gain = 1.0 + WIND_FEEDBACK_SCALE * machinery_proximity
            vx += wind_x * scale * wind_gain
            vy += wind_y * scale * wind_gain
            events.append('WIND')
            if machinery_proximity > 0.25:
                events.append('WIND_MACHINE_COUPLING')

        slope_grad = self.slope_gradient(x)
        if grounded and abs(slope_grad) > 1e-4:
            vx += slope_grad * SLOPE_ACCELERATION * 10.0
            events.append('SLOPE_FORCE')

        # Updrafts and air cannons operate even without a direct action.
        updraft, active_updraft = self.updraft_at(x, y)
        if active_updraft and not grounded:
            vy += updraft
            events.append('UPDRAFT')

        for cannon in self.air_cannons:
            d = math.hypot(x - cannon['x'], y - cannon['y'])
            if d < cannon['radius'] and not grounded:
                factor = 1.0 - d / cannon['radius']
                vx += cannon['vx'] * factor
                vy += cannon['vy'] * factor
                events.append('AIR_CANNON')

        # ----------------------------------------------------
        # SURFACE DYNAMICS
        # ----------------------------------------------------
        surface_kind = self.surface_kind(x, y)
        self.last_surface_kind = surface_kind
        vx = float(np.clip(vx, -MAX_SPEED * 1.35, MAX_SPEED * 1.35))

        if grounded:
            if surface_kind == 'SLIPPERY':
                vx *= SLIPPERY_FRICTION
                events.append('SLIP')
            elif surface_kind == 'ROUGH':
                vx *= ROUGH_FRICTION
                events.append('FRICTION')
            else:
                vx *= GROUND_FRICTION
        else:
            vx *= AIR_FRICTION
            vy -= GRAVITY

        previous_y = y
        x += vx
        y += vy

        # ----------------------------------------------------
        # CONTACT / COLLISIONS
        # ----------------------------------------------------
        if x < self.left + 8:
            x = self.left + 8
            vx *= -0.4
            events.append('BOUNDARY_WALL')
        if x > self.right - 8:
            x = self.right - 8
            vx *= -0.4
            events.append('BOUNDARY_WALL')

        wall = self.wall_hit(x, y)
        if wall is not None:
            wx, restitution = wall
            x = wx + (10.0 if x <= wx else -10.0)
            vx *= -restitution
            vy += abs(vx) * 0.12
            events.append('WALL_COLLISION')

        pendulum_hit = self.pendulum_collision(x, y, 12.0)
        if pendulum_hit is not None:
            idx, p = pendulum_hit
            pvx, pvy = self.pendulum_velocity(idx)
            dx, dy = x - p['bx'], y - p['by']
            norm = max(1e-6, math.hypot(dx, dy))
            nx, ny = dx / norm, dy / norm
            tangent_x, tangent_y = -ny, nx
            tangential_speed = pvx * tangent_x + pvy * tangent_y
            vx += tangent_x * (PENDULUM_TANGENT_IMPULSE + tangential_speed * 0.8)
            vy += tangent_y * (PENDULUM_TANGENT_IMPULSE + tangential_speed * 0.8)
            grounded = False
            events.append('PENDULUM_COLLISION')
            if agent.grapple_timer > 0:
                events.append('PENDULUM_SWING_TRANSFER')

        beam_hit = self.rotating_beam_collision(x, y, 11.0)
        if beam_hit is not None:
            _, b, cx, cy = beam_hit
            dx, dy = x - cx, y - cy
            norm = max(1e-6, math.hypot(dx, dy))
            vx += (dx / norm) * 3.8
            vy += (dy / norm) * 3.8
            grounded = False
            events.append('ROTATING_BEAM_HIT')

        # Ramp converts current momentum, not a fixed reward.
        for ramp in self.launch_ramps:
            if ramp['x1'] <= x <= ramp['x2'] and abs(y - ramp['y']) < 30 and vy <= 0:
                momentum = abs(vx)
                vx += ramp['boost_x'] * (1.0 if vx >= 0 else -1.0) + momentum * RAMP_MOMENTUM_SCALE
                vy = max(vy, ramp['boost_y'] + momentum * 0.25)
                grounded = False
                jumps = 1
                events.append('LAUNCH_RAMP')
                if momentum > 3.0:
                    events.append('MOMENTUM_LAUNCH')
                break

        tramp = self.trampoline_at(x, y)
        if tramp is not None and vy <= 0:
            _, pad = tramp
            y = pad['y'] + 20
            incoming = abs(vy) + abs(vx) * 0.25
            vy = pad['power'] + incoming * 0.35
            grounded = False
            jumps = 1
            events.append('TRAMPOLINE_BOUNCE')
            if incoming > 3.0:
                events.append('ENERGY_TRANSFER')

        # Spring bridge accumulates kinetic energy and releases it later.
        spring = self.spring_bridge_at(x, y)
        if spring is not None and grounded:
            idx, bridge = spring
            bridge['energy'] += abs(vx) * 0.18 + 0.04
            events.append('SPRING_LOAD')
            if bridge['energy'] > 2.4 and bridge['cooldown'] <= 0:
                vy = 8.0 + bridge['energy'] * 1.8
                vx += 2.0 if vx >= 0 else -2.0
                grounded = False
                jumps = 1
                bridge['cooldown'] = 18
                bridge['energy'] = 0.0
                events.append('SPRING_RELEASE')
        for bridge in self.spring_bridges:
            bridge['cooldown'] = max(0, bridge['cooldown'] - 1)

        # ----------------------------------------------------
        # LANDING / MOVING SURFACES
        # ----------------------------------------------------
        support = self.surface_height(x, previous_y)
        landing_speed = abs(vy)
        if support is not None and vy <= 0 and y <= support + 20:
            if not grounded and y < support + 18:
                events.append('LAND')
            y = support + 20
            vy = 0.0
            grounded = True
            jumps = 0

            svx, svy = self.surface_velocity(x, y)
            if abs(svx) + abs(svy) > 0.05:
                vx += svx * PLATFORM_CARRY_SCALE
                events.append('PLATFORM_CARRY')
                if abs(svx) > 1.0 or abs(svy) > 1.0:
                    events.append('FAST_SURFACE_TRANSFER')

            cv = self.conveyor_at(x, y)
            if cv is not None:
                vx += cv[1]['speed'] * 0.55
                events.append('CONVEYOR')

            # A hard landing loads a crumble platform.
            for cp in self.crumble_platforms:
                if (not cp['broken'] and cp['x1'] <= x <= cp['x2'] and abs(y - cp['y']) < 24):
                    load = landing_speed * CRUMBLE_LOAD_SCALE + abs(vx) * 0.01
                    cp['stability'] += max(0.03, load)
                    events.append('CRUMBLE_PRESSURE')
                    if cp['stability'] > 1.0:
                        cp['broken'] = True
                        cp['respawn'] = 0
                        grounded = False
                        events.append('CRUMBLE_BREAK')
                        # Structural failure produces a small air kick at the edge.
                        vy = max(vy, 2.8)
                        events.append('SUPPORT_EJECTION')

        else:
            grounded = False

        # ----------------------------------------------------
        # CRUMBLE RECOVERY / MOVING MACHINES
        # ----------------------------------------------------
        for cp in self.crumble_platforms:
            if cp['broken']:
                cp['respawn'] += 1
                if cp['respawn'] > 90:
                    cp['broken'] = False
                    cp['respawn'] = 0
                    cp['stability'] = 0.0

        for i in range(len(self.moving_platforms)):
            st = self.moving_platform_state(i)
            if abs(x - st['x']) < st['width'] / 2 + 12 and abs(y - st['y']) < 34:
                events.append(f'MOVING_PLATFORM_{i}')

        # A machine can be the immediate precursor to another machine.
        if 'TRAMPOLINE_BOUNCE' in events and active_wind:
            events.append('TRAMPOLINE_TO_WIND')
        if 'PENDULUM_COLLISION' in events and any('LAUNCH_RAMP' in e for e in events):
            events.append('PENDULUM_TO_RAMP')
        if 'GRAPPLE_TENSION' in events and 'PENDULUM_COLLISION' in events:
            events.append('GRAPPLE_TO_PENDULUM')
        if 'LAUNCH_RAMP' in events and active_wind:
            events.append('RAMP_TO_WIND')
        if 'WIND' in events and any(e.startswith('MOVING_PLATFORM_') for e in events):
            events.append('WIND_TO_PLATFORM')

        # Fall is an observed physical transition, not a penalty.
        if y < self.bottom - 25:
            events.append('FALL')
            x, y = self.checkpoint
            vx = 0.0
            vy = 0.0
            grounded = True
            jumps = 0

        # ----------------------------------------------------
        # CHAIN MEMORY
        # ----------------------------------------------------
        unique_events = list(dict.fromkeys(events))
        self.last_events = unique_events
        for e in unique_events:
            self.event_counts[e] += 1
            self.chain_event_counts[e] += 1

        # Keep only physical interaction events for cross-step chains.
        physical = [
            e for e in unique_events
            if e not in {'IDLE','MOVE_LEFT','MOVE_RIGHT','WAIT','BRAKE'}
        ]
        if physical:
            self.chain_context.extend(physical[-3:])
        recent = list(self.chain_context)[-CHAIN_MAX:]
        self.last_chain = list(dict.fromkeys(recent))

        event = unique_events[-1] if unique_events else 'NORMAL'

        agent.x = x
        agent.y = y
        agent.vx = vx
        agent.vy = vy
        agent.grounded = grounded
        agent.jumps = jumps
        agent.dash_timer = max(0, dash_timer - 1)
        agent.crouching = False if action != ACTION_SLIDE else agent.crouching

        agent.turtle.goto(x, y)

        return {
            'reward': 0.0,
            'event': event,
            'events': unique_events,
            'chain': list(self.last_chain),
            'triggered': len(unique_events) > 1 or event not in ('IDLE', 'MOVE_LEFT', 'MOVE_RIGHT'),
        }

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    def draw(self):

        self.drawer.clear()

        # Frame.
        self.drawer.color("#30303a")
        self.drawer.goto(self.left, self.bottom)
        self.drawer.pendown()
        self.drawer.goto(self.right, self.bottom)
        self.drawer.goto(self.right, self.top)
        self.drawer.goto(self.left, self.top)
        self.drawer.goto(self.left, self.bottom)
        self.drawer.penup()

        # Static platforms.
        self.drawer.color("#00dddd")
        for x1, x2, y in self.platforms:
            self.drawer.goto(x1, y)
            self.drawer.pendown()
            self.drawer.goto(x2, y)
            self.drawer.penup()

        # Slopes.
        self.drawer.color("#66bbff")
        for x1, x2, y1, y2 in self.slopes:
            self.drawer.goto(x1, y1)
            self.drawer.pendown()
            self.drawer.goto(x2, y2)
            self.drawer.penup()

        # Holes.
        self.drawer.color("#101018")
        for x1, x2 in self.holes:
            self.drawer.goto(x1, self.bottom + 3)
            self.drawer.pendown()
            self.drawer.goto(x2, self.bottom + 3)
            self.drawer.penup()

        # Walls.
        self.drawer.color("#bbbbcc")
        for wx, y1, y2, _ in self.walls:
            self.drawer.goto(wx, y1)
            self.drawer.pendown()
            self.drawer.goto(wx, y2)
            self.drawer.penup()

        # Slippery / rough surfaces.
        self.drawer.color("#5577ff")
        for x1, x2, y1, y2 in self.slippery_zones:
            self.drawer.goto(x1, y1)
            self.drawer.pendown()
            self.drawer.goto(x2, y1)
            self.drawer.penup()
        self.drawer.color("#aa7744")
        for x1, x2, y1, y2 in self.rough_zones:
            self.drawer.goto(x1, y1)
            self.drawer.pendown()
            self.drawer.goto(x2, y1)
            self.drawer.penup()

        # Moving platforms.
        self.drawer.color("#00ff88")
        for i in range(len(self.moving_platforms)):
            st = self.moving_platform_state(i)
            self.drawer.goto(st["x"] - st["width"] / 2, st["y"])
            self.drawer.pendown()
            self.drawer.goto(st["x"] + st["width"] / 2, st["y"])
            self.drawer.penup()

        # Trampolines.
        self.drawer.color("#ff55ff")
        for pad in self.trampolines:
            self.drawer.goto(pad["x1"], pad["y"])
            self.drawer.pendown()
            self.drawer.goto(pad["x2"], pad["y"])
            self.drawer.penup()

        # Wind zones.
        self.drawer.color("#88aaff")
        for zone in self.wind_zones:
            self.drawer.goto(zone["x1"], zone["y1"])
            self.drawer.pendown()
            self.drawer.goto(zone["x2"], zone["y1"])
            self.drawer.goto(zone["x2"], zone["y2"])
            self.drawer.goto(zone["x1"], zone["y2"])
            self.drawer.goto(zone["x1"], zone["y1"])
            self.drawer.penup()

        # Conveyors.
        self.drawer.color("#88ddff")
        for c in self.conveyors:
            self.drawer.goto(c["x1"], c["y"])
            self.drawer.pendown()
            self.drawer.goto(c["x2"], c["y"])
            self.drawer.penup()
            # tiny arrows
            step = 12 if c["speed"] > 0 else -12
            for ax in np.arange(c["x1"] + (8 if c["speed"] > 0 else -8), c["x2"], step):
                self.drawer.goto(ax, c["y"] + 4)
                self.drawer.dot(4)

        # Updraft zones.
        self.drawer.color("#6688ff")
        for z in self.updraft_zones:
            self.drawer.goto(z["x1"], z["y1"])
            self.drawer.pendown()
            self.drawer.goto(z["x1"], z["y2"])
            self.drawer.penup()

        # Launch ramps.
        self.drawer.color("#ffd166")
        for r in self.launch_ramps:
            self.drawer.goto(r["x1"], r["y"])
            self.drawer.pendown()
            self.drawer.goto(r["x2"], r["y"] + 18)
            self.drawer.penup()

        # Crumble platforms.
        self.drawer.color("#d9a066")
        for cp in self.crumble_platforms:
            if not cp["broken"]:
                self.drawer.goto(cp["x1"], cp["y"])
                self.drawer.pendown()
                self.drawer.goto(cp["x2"], cp["y"])
                self.drawer.penup()

        # Seesaws.
        self.drawer.color("#c080ff")
        for i in range(len(self.seesaws)):
            ss = self.seesaw_state(i)
            self.drawer.goto(ss["x1"], ss["y1"])
            self.drawer.pendown()
            self.drawer.goto(ss["x2"], ss["y2"])
            self.drawer.penup()
            self.drawer.goto(ss["px"], ss["py"])
            self.drawer.dot(7)

        # Rotating beams.
        self.drawer.color("#ff9955")
        for i in range(len(self.rotating_beams)):
            b = self.rotating_beam_state(i)
            self.drawer.goto(b["px"], b["py"])
            self.drawer.pendown()
            self.drawer.goto(b["x2"], b["y2"])
            self.drawer.penup()

        # Grapple anchors.
        self.drawer.color("#ffee66")
        for gx, gy in self.grapple_points:
            self.drawer.goto(gx, gy)
            self.drawer.dot(9)

        # Pendulums.
        self.drawer.color("#ff6644")
        for i in range(len(self.pendulums)):
            p = self.pendulum_state(i)
            self.drawer.goto(p["px"], p["py"])
            self.drawer.pendown()
            self.drawer.goto(p["bx"], p["by"])
            self.drawer.penup()
            self.drawer.goto(p["bx"], p["by"])
            self.drawer.dot(18)


# ============================================================
# 22. SENSORS v5
# ============================================================

def get_sensors(world, agent):

    direction = 1.0 if agent.vx >= 0 else -1.0
    front_distance = 40.0
    front_x = agent.x + front_distance * direction

    front_surface = world.surface_height(front_x, agent.y)
    current_surface = world.surface_height(agent.x, agent.y)

    wind_x, wind_y, active_wind = world.wind_at(agent.x, agent.y)
    pendulum_distances = []
    for i in range(len(world.pendulums)):
        p = world.pendulum_state(i)
        pendulum_distances.append(
            math.hypot(agent.x - p["bx"], agent.y - p["by"])
        )
    nearest_pendulum = min(pendulum_distances) if pendulum_distances else 400.0

    moving_distances = []
    for i in range(len(world.moving_platforms)):
        mp = world.moving_platform_state(i)
        moving_distances.append(
            math.hypot(agent.x - mp["x"], agent.y - mp["y"])
        )
    nearest_moving = min(moving_distances) if moving_distances else 400.0

    trampoline_near = 0.0
    for pad in world.trampolines:
        if pad["x1"] - 15 <= agent.x <= pad["x2"] + 15 and abs(agent.y - pad["y"]) < 45:
            trampoline_near = 1.0
            break

    hole_near = 0.0
    for x1, x2 in world.holes:
        if x1 - 25 <= agent.x <= x2 + 25:
            hole_near = 1.0
            break

    wall_near = 0.0
    for wx, y1, y2, _ in world.walls:
        if abs(agent.x - wx) < 35 and y1 - 20 <= agent.y <= y2 + 20:
            wall_near = 1.0
            break

    conveyor_near = 0.0
    conveyor_speed = 0.0
    cinfo = world.conveyor_at(agent.x, agent.y)
    if cinfo is not None:
        conveyor_near = 1.0
        conveyor_speed = cinfo[1]["speed"] / 2.0

    updraft, active_updraft = world.updraft_at(agent.x, agent.y)
    grapple_target = world.grapple_target(agent.x, agent.y)
    grapple_dist = 2.0 if grapple_target is None else min(grapple_target[3] / GRAPPLE_RANGE, 2.0)

    beam_near = 0.0
    if world.rotating_beam_collision(agent.x, agent.y, 32.0) is not None:
        beam_near = 1.0

    seesaw_near = 0.0
    for i in range(len(world.seesaws)):
        ss = world.seesaw_state(i)
        if math.hypot(agent.x - ss["px"], agent.y - ss["py"]) < 90:
            seesaw_near = 1.0
            break

    crumble_near = 0.0
    for cp in world.crumble_platforms:
        if (not cp["broken"]) and cp["x1"] - 15 <= agent.x <= cp["x2"] + 15 and abs(agent.y - cp["y"]) < 40:
            crumble_near = cp["stability"]
            break

    return np.array([
        agent.x / 500.0,
        agent.y / 250.0,
        agent.vx / MAX_SPEED,
        agent.vy / 18.0,
        float(agent.grounded),
        float(agent.jumps / MAX_JUMPS),
        float(agent.dash_timer > 0),
        float(front_surface is not None),
        0.0 if current_surface is None else current_surface / 250.0,
        0.0 if front_surface is None else front_surface / 250.0,
        float(bool(active_wind)),
        wind_x / 0.5,
        wind_y / 0.5,
        min(nearest_pendulum / 250.0, 2.0),
        min(nearest_moving / 300.0, 2.0),
        trampoline_near,
        hole_near,
        wall_near,
        conveyor_near,
        conveyor_speed,
        updraft / 1.0,
        grapple_dist,
        float(agent.grapple_timer > 0),
        beam_near,
        seesaw_near,
        crumble_near,
        float(world.time % 1000) / 1000.0,
        min(abs(agent.vx * agent.vy) / 120.0, 3.0),
        float(len(world.last_chain)) / CHAIN_MAX,
        min(sum(world.chain_event_counts.values()) / 500.0, 3.0),
    ], dtype=float)


# ============================================================
# 23. AGENT
# ============================================================

class Agent:

    def __init__(
        self,
        agent_id,
        color,
        world,
        model,
    ):

        self.id = agent_id

        self.color = color

        self.world = world

        self.model = model

        self.turtle = turtle.Turtle()

        self.turtle.shape(
            "turtle"
        )

        self.turtle.color(
            color
        )

        self.turtle.penup()

        self.turtle.speed(0)

        self.self_model = (
            model.self_models[
                agent_id
            ]
        )

        self.reset()

    def reset(self):

        self.x = (
            -490.0
            +
            self.id * 18
        )

        self.y = -200.0

        self.vx = 0.0
        self.vy = 0.0

        self.grounded = True

        self.jumps = 0

        self.dash_timer = 0
        self.grapple_timer = 0
        self.grapple_target = None
        self.crouching = False

        self.steps = 0

        self.episode_reward = 0.0

        self.total_reward = 0.0

        self.successes = 0
        self.failures = 0

        self.last_action = ACTION_NONE

        self.last_event = "NONE"

        self.last_error = 0.0
        self.last_physics_concept = None
        self.last_physics_chain = None

        self.turtle.goto(
            self.x,
            self.y,
        )

    def step(self):

        state = get_sensors(
            self.world,
            self,
        )

        # ----------------------------------------------------
        # CURIOSITY PREVIEW
        # ----------------------------------------------------

        curiosity = 0.0

        for action in range(
            NUM_ACTIONS
        ):

            curiosity += (
                self.model.curiosity.action_bonus(
                    self.model,
                    state,
                    action,
                    self.self_model,
                )
            )

        curiosity /= NUM_ACTIONS

        # ----------------------------------------------------
        # ACTION
        # ----------------------------------------------------

        action = self.model.select_action(
            self.id,
            state,
            self.world,
        )

        # ----------------------------------------------------
        # ACT
        # ----------------------------------------------------

        result = self.world.step(
            self,
            action,
        )

        # ----------------------------------------------------
        # OBSERVE
        # ----------------------------------------------------

        next_state = get_sensors(
            self.world,
            self,
        )

        # ----------------------------------------------------
        # LEARN
        # ----------------------------------------------------

        learned = self.model.learn(
            self.id,
            state,
            action,
            next_state,
            result["reward"],
            result["event"],
            result.get("events", [result["event"]]),
        )

        error = learned["error"]
        self.last_physics_concept = learned.get("physics_concept")
        self.last_physics_chain = learned.get("physics_chain")

        # ----------------------------------------------------
        # SELF MODEL
        # ----------------------------------------------------

        self.self_model.observe(
            action,
            result["reward"],
            error,
            result["event"],
        )

        # ----------------------------------------------------
        # SKILL
        # ----------------------------------------------------

        self.model.skill_system.observe(
            self.id,
            action,
            result["reward"],
            result["event"],
        )

        # ----------------------------------------------------
        # GOAL
        # ----------------------------------------------------

        # v5: GoalSystem is retained for architectural compatibility,
        # but it is not a source of reward or action directives.
        goal = None

        # ----------------------------------------------------
        # SPECIALIZATION
        # ----------------------------------------------------

        self.model.specialization.update(
            self,
            curiosity,
            result["reward"],
        )

        # ----------------------------------------------------
        # STATS
        # ----------------------------------------------------

        self.steps += 1

        self.last_action = action

        self.last_event = result["event"]

        self.last_error = error

        self.episode_reward += (
            result["reward"]
        )

        self.total_reward += (
            result["reward"]
        )

        if result["event"] == "SUCCESS":

            self.successes += 1

        if result["event"] in (
            "DANGER",
            "FALL",
        ):

            self.failures += 1

        # ----------------------------------------------------
        # IDENTITY
        # ----------------------------------------------------

        self.self_model.update_identity(
            exploration=curiosity,
            collection=(
                1.0
                if result["event"] == "SUCCESS"
                else 0.0
            ),
            survival=(
                1.0
                if result["event"]
                not in ("DANGER", "FALL")
                else 0.0
            ),
            control=(
                1.0
                if result["event"] == "SWITCH"
                else 0.0
            ),
        )


# ============================================================
# 24. SHARED BRAIN
# ============================================================

shared_hippocampus = Hippocampus()

shared_dream = DreamSimulator(
    shared_hippocampus
)

shared_causal_model = CausalModel()

shared_curiosity = CuriosityEngine()

shared_goal_system = GoalSystem()

shared_skill_system = SkillSystem()

shared_counterfactual = CounterfactualDream(
    shared_hippocampus,
    shared_causal_model,
)

shared_reflection = ReflectionSystem()

shared_specialization = (
    MultiAgentSpecialization()
)

shared_model = WorldModel(
    shared_hippocampus,
    shared_dream,
    shared_causal_model,
    shared_curiosity,
    shared_goal_system,
    shared_skill_system,
    shared_counterfactual,
    shared_reflection,
    shared_specialization,
)

for i in range(NUM_AGENTS):

    shared_specialization.initialize(i)


# ============================================================
# 25. SCREEN
# ============================================================

screen = turtle.Screen()

screen.setup(
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
)

screen.bgcolor(
    "#0b0b12"
)

screen.title(
    "Embodied Self-Organizing World Model v5"
)

screen.tracer(False)


# ============================================================
# 26. WORLD
# ============================================================

world = TurtleWorld()


# ============================================================
# 27. AGENTS
# ============================================================

agent_colors = [
    "#00ffff",
    "#00ff7f",
    "#ffa500",
]

agents = []

for i, color in enumerate(
    agent_colors
):

    agents.append(
        Agent(
            i,
            color,
            world,
            shared_model,
        )
    )


# ============================================================
# 28. DRAWERS
# ============================================================

model_drawer = turtle.Turtle()

model_drawer.hideturtle()
model_drawer.penup()
model_drawer.speed(0)


text_drawer = turtle.Turtle()

text_drawer.hideturtle()
text_drawer.penup()
text_drawer.speed(0)


dream_drawer = turtle.Turtle()

dream_drawer.hideturtle()
dream_drawer.penup()
dream_drawer.speed(0)


# ============================================================
# 29. TEXT
# ============================================================

def write_text(
    x,
    y,
    text,
    size=10,
    color="#dddddd",
):

    text_drawer.goto(
        x,
        y,
    )

    text_drawer.color(
        color
    )

    text_drawer.write(
        text,
        font=(
            "Arial",
            size,
            "normal",
        ),
    )


# ============================================================
# 30. DRAW WORLD MODEL
# ============================================================

def draw_world_model():

    model_drawer.clear()
    text_drawer.clear()
    dream_drawer.clear()

    stats = (
        shared_hippocampus.statistics()
    )

    causal_stats = (
        shared_causal_model.statistics()
    )

    world_events = sorted(
        world.event_counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:8]

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    write_text(
        -510,
        315,
        "EXTERNAL WORLD",
        15,
        "#ffffff",
    )

    write_text(
        -510,
        295,
        "REWARD = 0 / GOAL = NONE / PHYSICS = OPEN",
        9,
        "#77ddaa",
    )

    write_text(
        50,
        315,
        "EVOLVED SHARED WORLD MODEL v5.0",
        15,
        "#ffffff",
    )

    # --------------------------------------------------------
    # BASIC STATS
    # --------------------------------------------------------

    write_text(
        50,
        290,
        f"Events       : {stats['events']}",
        10,
    )

    write_text(
        50,
        273,
        f"Transitions  : {stats['transitions']}",
        10,
        "#66ccff",
    )

    write_text(
        50,
        256,
        f"Places       : {stats['places']}",
        10,
    )

    write_text(
        50,
        239,
        f"Concepts     : {stats['concepts']}",
        10,
    )

    write_text(
        50,
        222,
        f"Replay       : {len(shared_model.replay)}",
        10,
    )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if shared_model.error_history:

        error = np.mean(
            shared_model.error_history[-50:]
        )

    else:

        error = 0.0

    write_text(
        50,
        205,
        f"Prediction Error : {error:.3f}",
        10,
        "#ffaa00",
    )

    write_text(
        50,
        188,
        f"Exploration      : {shared_model.exploration:.3f}",
        10,
        "#aa88ff",
    )

    write_text(
        50,
        171,
        f"Causal Relations : {causal_stats['relations']}",
        10,
        "#ff66aa",
    )

    write_text(
        50,
        154,
        f"Strong Causes    : {causal_stats['strong']}",
        10,
        "#ff66aa",
    )

    write_text(
        50,
        137,
        f"Skills           : {len(shared_skill_system.skills)}",
        10,
        "#00ff88",
    )

    write_text(
        50,
        120,
        f"Physics Concepts : {len(shared_model.physics_concepts)}",
        10,
        "#ffaa66",
    )

    write_text(
        50,
        103,
        f"Physics Chains   : {len(shared_model.physics_chains)}",
        10,
        "#ffcc88",
    )

    write_text(
        50,
        86,
        f"Parkour Systems : {len(world.moving_platforms)+len(world.trampolines)+len(world.pendulums)+len(world.rotating_beams)+len(world.seesaws)+len(world.conveyors)+len(world.updraft_zones)+len(world.air_cannons)+len(world.spring_bridges)}",
        9,
        "#66ddff",
    )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    important_events = sorted(
        shared_hippocampus.events,
        key=lambda e:
            e.energy
            +
            e.visits * 0.5
            +
            abs(e.reward_mean)
            +
            e.error_mean * 0.15,
        reverse=True,
    )[:70]

    positions = {}

    for i, event in enumerate(
        important_events
    ):

        x = (
            90
            +
            (i % 10) * 42
        )

        y = (
            110
            -
            (i // 10) * 35
        )

        positions[event.id] = (
            x,
            y,
        )

    # --------------------------------------------------------
    # TRANSITIONS
    # --------------------------------------------------------

    for transition in (
        shared_hippocampus.transitions
    ):

        if transition.source_id not in positions:
            continue

        target_id = transition.best_target()

        if target_id not in positions:
            continue

        sx, sy = positions[
            transition.source_id
        ]

        tx, ty = positions[
            target_id
        ]

        if transition.reward_mean > 2:

            color = "#00ff7f"

        elif transition.reward_mean < -3:

            color = "#ff3344"

        elif transition.confidence > 0.7:

            color = "#668899"

        else:

            color = "#354454"

        width = int(
            clamp(
                1
                +
                transition.confidence * 3,
                1,
                4,
            )
        )

        model_drawer.color(
            color
        )

        model_drawer.pensize(
            width
        )

        model_drawer.goto(
            sx,
            sy,
        )

        model_drawer.pendown()

        model_drawer.goto(
            tx,
            ty,
        )

        model_drawer.penup()

    # --------------------------------------------------------
    # NODE COLORS
    # --------------------------------------------------------

    concept_colors = {

        "SUCCESS":
            "#00ff7f",

        "FAILURE":
            "#ff3344",

        "SURPRISE":
            "#ff9900",

        "JUMP":
            "#bb88ff",

        "DASH":
            "#ffff00",

        "BRAKE":
            "#ff66aa",

        "TURN_LEFT":
            "#66ccff",

        "TURN_RIGHT":
            "#66ccff",

        "WAIT":
            "#999999",

        "MOVEMENT":
            "#66ccff",
    }

    for event in important_events:

        x, y = positions[
            event.id
        ]

        if event.concepts:

            concept = (
                shared_hippocampus.get_concept(
                    event.concepts[0]
                )
            )

            if concept:

                color = concept_colors.get(
                    concept.name,
                    "#66ccff",
                )

            else:

                color = "#66ccff"

        else:

            color = "#66ccff"

        size = int(
            clamp(
                5 + event.visits,
                5,
                18,
            )
        )

        model_drawer.goto(
            x,
            y,
        )

        model_drawer.dot(
            size,
            color,
        )

    # --------------------------------------------------------
    # AGENTS
    # --------------------------------------------------------

    y = 175

    for agent in agents:

        role = shared_specialization.roles.get(
            agent.id,
            "UNKNOWN",
        )

        self_model = agent.self_model

        write_text(
            -510,
            y,
            (
                f"Agent {agent.id} "
                f"R={agent.episode_reward:+.1f} "
                f"vx={agent.vx:+.1f} "
                f"vy={agent.vy:+.1f} "
                f"{ACTION_NAMES.get(agent.last_action)} "
                f"{agent.last_event} "
                f"PE={agent.last_error:.2f}"
            ),
            9,
            agent.color,
        )

        physics_label = "NONE"
        if agent.last_physics_concept is not None:
            pc = agent.last_physics_concept
            physics_label = f"{pc.action}->{pc.event}->{pc.consequence}"
            if len(physics_label) > 42:
                physics_label = physics_label[:39] + "..."

        write_text(
            -510,
            y - 14,
            (
                f"  ROLE={role} "
                f"SELF={self_model.dominant_role()} "
                f"CONF={self_model.confidence:.2f}"
            ),
            8,
            agent.color,
        )

        write_text(
            -510,
            y - 26,
            f"  PHYSICS={physics_label}",
            7,
            "#ddddaa",
        )

        y -= 30

    # --------------------------------------------------------
    # CONCEPTS
    # --------------------------------------------------------

    y = -105

    write_text(
        50,
        y,
        "CONCEPTS",
        11,
        "#ffffff",
    )

    y -= 20

    concepts = sorted(
        shared_hippocampus.concepts,
        key=lambda c:
            c.visits,
        reverse=True,
    )[:8]

    for concept in concepts:

        color = concept_colors.get(
            concept.name,
            "#cccccc",
        )

        write_text(
            50,
            y,
            (
                f"{concept.name:<13} "
                f"n={concept.visits:<4} "
                f"V={concept.value:+.2f}"
            ),
            9,
            color,
        )

        y -= 16

    # --------------------------------------------------------
    # PHYSICS CONCEPTS
    # --------------------------------------------------------

    write_text(
        235,
        -160,
        "PHYSICS CONCEPTS",
        10,
        "#ffffff",
    )

    physics_sorted = sorted(
        shared_model.physics_concepts.values(),
        key=lambda c: c.visits * 0.7 + c.strength + c.energy * 0.2,
        reverse=True,
    )[:6]

    py = -178
    for pc in physics_sorted:
        label = f"{pc.action}->{pc.event}->{pc.consequence}"
        if len(label) > 38:
            label = label[:35] + "..."
        write_text(
            235,
            py,
            f"{label} n={pc.visits}",
            7,
            "#ffcc88",
        )
        py -= 13

    write_text(
        460,
        -160,
        "PHYSICS CHAINS",
        10,
        "#ffffff",
    )

    chain_sorted = sorted(
        shared_model.physics_chains.values(),
        key=lambda c: c.visits * 0.8 + c.strength + c.energy * 0.2,
        reverse=True,
    )[:5]

    cy = -178
    for chain in chain_sorted:
        label = f"{chain.action}->" + "+".join(chain.events) + f"=>{chain.consequence}"
        if len(label) > 47:
            label = label[:44] + "..."
        write_text(
            460,
            cy,
            f"{label} n={chain.visits}",
            7,
            "#ff9966",
        )
        cy -= 13

    # --------------------------------------------------------
    # GOALS (ARCHITECTURE RETAINED, NOT DRIVING ACTION)
    # --------------------------------------------------------

    write_text(
        235,
        -105,
        "GOALS: INACTIVE IN v5.0",
        10,
        "#ffffff",
    )

    write_text(
        235,
        -125,
        "world has no predefined objective",
        8,
        "#77ddaa",
    )

    write_text(
        235,
        -140,
        "action = novelty + uncertainty + physics novelty",
        8,
        "#77ddaa",
    )

    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    write_text(
        410,
        -105,
        "SKILLS",
        11,
        "#ffffff",
    )

    y = -125

    strongest_skills = sorted(
        shared_skill_system.skills,
        key=lambda s:
            s.score(),
        reverse=True,
    )[:7]

    for skill in strongest_skills:

        write_text(
            410,
            y,
            (
                f"{skill.name:<15} "
                f"V={skill.value:+.2f} "
                f"S={skill.success_rate:.2f}"
            ),
            8,
            "#00ff88",
        )

        y -= 15

    # --------------------------------------------------------
    # STRONG TRANSITIONS
    # --------------------------------------------------------

    write_text(
        610,
        -105,
        "STRONG TRANSITIONS",
        11,
        "#ffffff",
    )

    strongest = sorted(
        shared_hippocampus.transitions,
        key=lambda t:
            t.confidence
            +
            t.visits * 0.02
            +
            abs(t.reward_mean) * 0.1,
        reverse=True,
    )[:7]

    y = -125

    for t in strongest:

        write_text(
            610,
            y,
            (
                f"E{t.source_id} "
                f"--{ACTION_NAMES[t.action]}--> "
                f"E{t.best_target()} "
                f"R={t.reward_mean:+.2f} "
                f"C={t.confidence:.2f}"
            ),
            8,
            "#88aacc",
        )

        y -= 15

    # --------------------------------------------------------
    # WORLD EVENTS
    # --------------------------------------------------------

    write_text(
        610,
        130,
        "PHYSICAL EVENTS",
        11,
        "#ffffff",
    )

    ey = 110
    for name, count in world_events:
        write_text(
            610,
            ey,
            f"{name}: {count}",
            8,
            "#66ccff",
        )
        ey -= 14

    # --------------------------------------------------------
    # DREAM
    # --------------------------------------------------------

    if shared_model.last_dream:

        write_text(
            50,
            -285,
            "LAST DREAM",
            11,
            "#aa88ff",
        )

        x = 140

        for event in (
            shared_model.last_dream[:16]
        ):

            if event.reward_mean >= 3:

                color = "#00ff7f"

            elif event.reward_mean <= -5:

                color = "#ff3344"

            elif event.error_mean >= 6:

                color = "#ff9900"

            else:

                color = "#aa88ff"

            dream_drawer.goto(
                x,
                -285,
            )

            dream_drawer.dot(
                10,
                color,
            )

            x += 22

        write_text(
            50,
            -312,
            (
                f"dream score = "
                f"{shared_model.last_dream_score:+.2f}"
            ),
            9,
            "#9999bb",
        )

    # --------------------------------------------------------
    # REFLECTION
    # --------------------------------------------------------

    write_text(
        610,
        -235,
        "REFLECTION",
        11,
        "#ffffff",
    )

    reflection = (
        shared_reflection.last_reflection
    )

    if len(reflection) > 80:

        reflection = (
            reflection[:77]
            +
            "..."
        )

    write_text(
        610,
        -255,
        reflection,
        8,
        "#dd88ff",
    )


# ============================================================
# 31. EPISODE CONTROL
# ============================================================

episode = 0
current_step = 0

phase = "DAY"

finished = False


def reset_episode():

    world.reset()

    shared_model.last_event.clear()

    for agent in agents:

        agent.reset()


# ============================================================
# 32. SLEEP
# ============================================================

def run_sleep():

    global episode
    global current_step
    global phase
    global finished

    if finished:
        return

    phase = "SLEEP"

    # --------------------------------------------------------
    # COUNTERFACTUAL DREAM
    # --------------------------------------------------------

    if agents:

        agent = agents[0]

        state = get_sensors(
            world,
            agent,
        )

        shared_model.last_counterfactuals = (
            shared_counterfactual.evaluate(
                state,
                agent.last_action,
                shared_model,
            )
        )

    # --------------------------------------------------------
    # REFLECTION
    # --------------------------------------------------------

    if agents:

        agent = max(
            agents,
            key=lambda a:
                a.episode_reward
                -
                a.failures * 2.0
        )

        state = get_sensors(
            world,
            agent,
        )

        counterfactuals = (
            shared_counterfactual.last_counterfactuals
        )

        goal = Goal(
            "SELF_DISCOVERY",
            0.0,
            "understand_physics",
        )

        shared_reflection.reflect(
            agent,
            goal,
            counterfactuals,
        )

    # --------------------------------------------------------
    # CONSOLIDATION
    # --------------------------------------------------------

    shared_model.sleep()

    world.draw()

    draw_world_model()

    write_text(
        -510,
        285,
        "SLEEP / REPLAY / COUNTERFACTUAL DREAM / REFLECTION / CONSOLIDATION",
        11,
        "#aa88ff",
    )

    screen.update()

    episode += 1

    if episode >= MAX_EPISODES:

        finished = True

        write_text(
            -200,
            -340,
            "SIMULATION FINISHED",
            18,
            "#ffffff",
        )

        screen.update()

        return

    current_step = 0

    screen.ontimer(
        start_day,
        SLEEP_DELAY,
    )


# ============================================================
# 33. DAY
# ============================================================

def run_day():

    global current_step
    global phase

    if finished:
        return

    phase = "DAY"

    world.update()

    for agent in agents:

        agent.step()

    world.draw()

    draw_world_model()

    write_text(
        -510,
        285,
        (
            f"DAY "
            f"Episode {episode + 1}/{MAX_EPISODES} "
            f"Step {current_step}/{STEPS_PER_EPISODE}"
        ),
        11,
        "#00ff7f",
    )

    screen.update()

    current_step += 1

    if (
        current_step
        <
        STEPS_PER_EPISODE
    ):

        screen.ontimer(
            run_day,
            DAY_DELAY,
        )

    else:

        screen.ontimer(
            run_sleep,
            300,
        )


# ============================================================
# 34. START DAY
# ============================================================

def start_day():

    if finished:
        return

    reset_episode()

    run_day()


# ============================================================
# 35. MAIN
# ============================================================

def main():

    reset_episode()

    world.draw()

    draw_world_model()

    write_text(
        -510,
        285,
        "INITIALIZING EMBODIED WORLD MODEL v5",
        11,
        "#ffffff",
    )

    screen.update()

    screen.ontimer(
        start_day,
        500,
    )

    turtle.done()


# ============================================================
# 36. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
