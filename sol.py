# ============================================================
# EVOLVED EMBODIED SELF-ORGANIZING WORLD MODEL v3
# ============================================================
#
# 既存アーキテクチャを維持した拡張版
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

NUM_ACTIONS = 7

ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3
ACTION_DASH = 4
ACTION_BRAKE = 5
ACTION_WAIT = 6

ACTION_NAMES = {
    ACTION_NONE: "NONE",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_JUMP: "JUMP",
    ACTION_DASH: "DASH",
    ACTION_BRAKE: "BRAKE",
    ACTION_WAIT: "WAIT",
}


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

    def __init__(
        self,
        event_id,
        state,
        action,
        agent_id,
    ):
        self.id = event_id

        self.state = np.array(
            state,
            dtype=float,
        )

        self.action = int(action)
        self.agent_id = agent_id

        self.visits = 1

        self.energy = 1.0
        self.activation = 1.0

        self.reward_mean = 0.0
        self.error_mean = 1.0

        self.place_id = None
        self.concepts = []

    def similarity(self, state, action):

        if self.action != action:
            return 0.0

        state = np.asarray(
            state,
            dtype=float,
        )

        distance = np.linalg.norm(
            self.state - state
        )

        return math.exp(-distance * 1.8)

    def reinforce(
        self,
        state,
        reward,
        error,
    ):

        self.visits += 1

        self.energy = min(
            3.0,
            self.energy + 0.05,
        )

        self.activation = min(
            2.0,
            self.activation + 0.10,
        )

        state = np.asarray(
            state,
            dtype=float,
        )

        self.state = (
            0.90 * self.state
            +
            0.10 * state
        )

        self.reward_mean = (
            0.90 * self.reward_mean
            +
            0.10 * reward
        )

        self.error_mean = (
            0.90 * self.error_mean
            +
            0.10 * error
        )

    def decay(self):

        self.energy *= 0.997
        self.activation *= 0.985


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
# 10. HIPPOCAMPUS
# ============================================================

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
        action,
        reward,
        error,
        agent_id,
    ):

        self.total_encodes += 1

        best = None
        best_score = 0.0

        for event in self.events:

            score = event.similarity(
                state,
                action,
            )

            if score > best_score:

                best_score = score
                best = event

        if (
            best is not None
            and
            best_score >= EVENT_SIM_THRESHOLD
        ):

            best.reinforce(
                state,
                reward,
                error,
            )

            return best, False

        event = EventCell(
            self.next_event_id,
            state,
            action,
            agent_id,
        )

        event.reward_mean = reward
        event.error_mean = error

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

        self.confidence = clamp(
            self.confidence
            +
            0.01
            if self.prediction_error < 1.0
            else self.confidence - 0.005,
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

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(
        self,
        state,
        action,
    ):

        best_event = None
        best_score = 0.0

        for event in self.hippo.events:

            score = event.similarity(
                state,
                action,
            )

            if score > best_score:

                best_score = score
                best_event = event

        if best_event is None:

            return {
                "next_state": None,
                "reward": 0.0,
                "uncertainty": 1.0,
                "event": None,
                "transition": None,
            }

        transition = self.hippo.get_transition(
            best_event.id,
            action,
        )

        if transition is None:

            return {
                "next_state": None,
                "reward": 0.0,
                "uncertainty": 1.0,
                "event": best_event,
                "transition": None,
            }

        uncertainty = (
            transition.error_mean
            /
            (
                1.0
                +
                transition.visits
            )
        )

        return {
            "next_state":
                transition.predicted_next_state.copy(),

            "reward":
                transition.reward_mean,

            "uncertainty":
                uncertainty,

            "event":
                best_event,

            "transition":
                transition,
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
    ):

        previous_event = self.last_event.get(
            agent_id
        )

        event, created = (
            self.hippo.encode_event(
                state,
                action,
                reward,
                0.0,
                agent_id,
            )
        )

        error = 1.0

        if previous_event is not None:

            predicted = self.predict(
                previous_event.state,
                action,
            )

            if predicted["next_state"] is None:

                error = 1.0

            else:

                error = float(
                    np.linalg.norm(
                        predicted["next_state"]
                        -
                        next_state
                    )
                )

            transition, _ = (
                self.hippo.encode_transition(
                    previous_event,
                    action,
                    event,
                    next_state,
                    reward,
                    error,
                )
            )

            self.hippo.link_concepts(
                previous_event,
                event,
            )

            self.replay.add(
                previous_event.id,
                event.id,
                action,
                reward,
                error,
                agent_id,
            )

            self.error_history.append(
                error
            )

            self.causal_model.observe(
                previous_event,
                event,
                action,
                reward,
                error,
            )

        place, concept = (
            self.hippo.organize(event)
        )

        self.last_event[agent_id] = event

        if len(self.error_history) > MAX_ERROR_HISTORY:

            self.error_history.pop(0)

        return {
            "event": event,
            "place": place,
            "concept": concept,
            "created": created,
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

        goal = self.goal_system.generate(
            state,
            world,
            self_model,
            0.0,
        )

        for action in range(NUM_ACTIONS):

            prediction = self.predict(
                state,
                action,
            )

            transition = prediction[
                "transition"
            ]

            if transition is None:

                score = (
                    2.5
                    *
                    self.exploration
                )

                if action == ACTION_JUMP:
                    score += 0.4

                if action == ACTION_DASH:
                    score += 0.3

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
                    reward * 1.5
                    +
                    uncertainty * 1.0
                    +
                    novelty * 0.9
                    +
                    confidence * 0.5
                    -
                    risk * 3.0
                )

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

            score += (
                self.specialization.role_bonus(
                    agent_id,
                    action,
                )
            )

            if goal.name == "SURVIVE":

                if action in (
                    ACTION_BRAKE,
                    ACTION_WAIT,
                ):

                    score += 0.15

            if goal.name == "EXPLORE":

                if action in (
                    ACTION_JUMP,
                    ACTION_DASH,
                ):

                    score += 0.25

            if goal.name == "COLLECT":

                if action == ACTION_RIGHT:
                    score += 0.25

            if goal.name == "CONTROL":

                if action in (
                    ACTION_RIGHT,
                    ACTION_JUMP,
                ):

                    score += 0.20

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

        self.exploration *= 0.92

        self.exploration = max(
            0.18,
            self.exploration,
        )

        self.last_event.clear()

        self.episode_count += 1


# ============================================================
# 21. TURTLE WORLD
# ============================================================

class TurtleWorld:

    def __init__(self):

        self.left = -520
        self.right = -20

        self.bottom = -280
        self.top = 270

        self.platforms = [
            [-520, -460, -130],
            [-445, -350, -70],
            [-335, -240, -10],
            [-225, -120, -90],
            [-100, -25, 30],
        ]

        self.hazards = [
            [-400, -355, -130],
            [-300, -250, -10],
            [-190, -145, -90],
        ]

        self.orbs = []

        self.switch_x = -300
        self.switch_y = -95

        self.bridge_active = False

        self.moving_platform = {
            "base_x": -250.0,
            "y": 80.0,
            "width": 70.0,
            "amplitude": 90.0,
        }

        self.time = 0.0

        self.checkpoint = (
            -490.0,
            -110.0,
        )

        self.drawer = turtle.Turtle()

        self.drawer.hideturtle()
        self.drawer.penup()
        self.drawer.speed(0)

    def reset(self):

        self.time = 0.0

        self.bridge_active = False

        self.orbs = [
            np.array(
                [-125.0, 55.0]
            ),
            np.array(
                [-40.0, 100.0]
            ),
        ]

        self.draw()

    def moving_x(self):

        return (
            self.moving_platform["base_x"]
            +
            self.moving_platform["amplitude"]
            *
            math.sin(
                self.time * 0.05
            )
        )

    def update(self):

        self.time += 1.0

    def platform_at(
        self,
        x,
        y,
    ):

        for x1, x2, py in self.platforms:

            if (
                x1 <= x <= x2
                and
                abs(y - py) < 20
            ):

                return py

        if self.bridge_active:

            if (
                -350 <= x <= -120
                and
                abs(y + 75) < 20
            ):

                return -75

        mx = self.moving_x()

        mp = self.moving_platform

        if (
            mx - mp["width"] / 2
            <= x
            <=
            mx + mp["width"] / 2
            and
            abs(y - mp["y"]) < 20
        ):

            return mp["y"]

        return None

    def is_hazard(
        self,
        x,
        y,
    ):

        for x1, x2, hy in self.hazards:

            if (
                x1 <= x <= x2
                and
                y <= hy + 18
            ):

                return True

        return False

    def near_switch(
        self,
        x,
        y,
    ):

        return (
            math.hypot(
                x - self.switch_x,
                y - self.switch_y,
            )
            <
            24
        )

    def collect_orb(
        self,
        x,
        y,
    ):

        collected = 0

        remaining = []

        for orb in self.orbs:

            distance = np.linalg.norm(
                np.array([x, y])
                -
                orb
            )

            if distance < 22:

                collected += 1

            else:

                remaining.append(orb)

        self.orbs = remaining

        return collected

    def step(
        self,
        agent,
        action,
    ):

        x = float(agent.x)
        y = float(agent.y)

        vx = float(agent.vx)
        vy = float(agent.vy)

        grounded = bool(agent.grounded)

        jumps = int(agent.jumps)

        dash_timer = int(
            agent.dash_timer
        )

        reward = -0.015

        event = "NORMAL"

        triggered = False

        if action == ACTION_LEFT:

            if grounded:
                vx -= GROUND_ACCEL
            else:
                vx -= AIR_ACCEL

            event = "MOVE"

        elif action == ACTION_RIGHT:

            if grounded:
                vx += GROUND_ACCEL
            else:
                vx += AIR_ACCEL

            event = "MOVE"

        elif action == ACTION_JUMP:

            if grounded:

                vy = JUMP_POWER

                grounded = False

                jumps = 1

                event = "JUMP"

            elif jumps < MAX_JUMPS:

                vy = DOUBLE_JUMP_POWER

                jumps += 1

                event = "DOUBLE_JUMP"

        elif action == ACTION_DASH:

            if dash_timer <= 0:

                direction = 1

                if abs(vx) > 0.5:

                    direction = (
                        1
                        if vx > 0
                        else -1
                    )

                vx = (
                    direction
                    *
                    DASH_SPEED
                )

                dash_timer = DASH_DURATION

                event = "DASH"

        elif action == ACTION_BRAKE:

            vx *= 0.30

            event = "BRAKE"

        elif action == ACTION_WAIT:

            vx *= 0.90

            event = "WAIT"

        vx = np.clip(
            vx,
            -MAX_SPEED,
            MAX_SPEED,
        )

        if grounded:

            vx *= GROUND_FRICTION

        else:

            vx *= AIR_FRICTION

        if not grounded:

            vy -= GRAVITY

            y += vy

        x += vx

        if x < self.left + 10:

            x = self.left + 10

            vx *= -0.4

            event = "WALL"

        if x > self.right - 10:

            x = self.right - 10

            vx *= -0.4

            event = "WALL"

        if self.near_switch(x, y):

            if not self.bridge_active:

                self.bridge_active = True

                reward += 2.0

                event = "SWITCH"

                triggered = True

        if self.is_hazard(x, y):

            reward -= 8.0

            event = "DANGER"

            triggered = True

            x, y = self.checkpoint

            vx = 0.0
            vy = 0.0

            grounded = True
            jumps = 0

        py = self.platform_at(
            x,
            y,
        )

        if (
            py is not None
            and
            vy <= 0
            and
            y <= py + 20
        ):

            y = py + 20

            vy = 0.0

            grounded = True

            jumps = 0

        elif y < -245:

            reward -= 10.0

            event = "FALL"

            triggered = True

            x, y = self.checkpoint

            vx = 0.0
            vy = 0.0

            grounded = True
            jumps = 0

        else:

            grounded = False

        collected = self.collect_orb(
            x,
            y,
        )

        if collected > 0:

            reward += 5.0 * collected

            event = "SUCCESS"

            triggered = True

        agent.x = x
        agent.y = y

        agent.vx = vx
        agent.vy = vy

        agent.grounded = grounded
        agent.jumps = jumps

        agent.dash_timer = max(
            0,
            dash_timer - 1,
        )

        agent.turtle.goto(
            x,
            y,
        )

        return {
            "reward": float(reward),
            "event": event,
            "triggered": triggered,
        }

    def draw(self):

        self.drawer.clear()

        self.drawer.color(
            "#444455"
        )

        self.drawer.goto(
            self.left,
            self.bottom,
        )

        self.drawer.pendown()

        self.drawer.goto(
            self.right,
            self.bottom,
        )

        self.drawer.goto(
            self.right,
            self.top,
        )

        self.drawer.goto(
            self.left,
            self.top,
        )

        self.drawer.goto(
            self.left,
            self.bottom,
        )

        self.drawer.penup()

        self.drawer.color(
            "#00dddd"
        )

        for x1, x2, y in self.platforms:

            self.drawer.goto(x1, y)

            self.drawer.pendown()

            self.drawer.goto(x2, y)

            self.drawer.penup()

        if self.bridge_active:

            self.drawer.color(
                "#ffaa00"
            )

            self.drawer.goto(
                -350,
                -75,
            )

            self.drawer.pendown()

            self.drawer.goto(
                -120,
                -75,
            )

            self.drawer.penup()

        mx = self.moving_x()

        mp = self.moving_platform

        self.drawer.color(
            "#00ff88"
        )

        self.drawer.goto(
            mx - mp["width"] / 2,
            mp["y"],
        )

        self.drawer.pendown()

        self.drawer.goto(
            mx + mp["width"] / 2,
            mp["y"],
        )

        self.drawer.penup()

        self.drawer.color(
            "#ff3344"
        )

        for x1, x2, y in self.hazards:

            self.drawer.goto(
                x1,
                y,
            )

            self.drawer.pendown()

            self.drawer.goto(
                x2,
                y,
            )

            self.drawer.penup()

        self.drawer.goto(
            self.switch_x,
            self.switch_y,
        )

        self.drawer.dot(
            15,
            "#ff00ff"
            if self.bridge_active
            else "#555555",
        )

        self.drawer.color(
            "#ffff00"
        )

        for orb in self.orbs:

            self.drawer.goto(
                orb[0],
                orb[1],
            )

            self.drawer.dot(
                11
            )


# ============================================================
# 22. SENSORS
# ============================================================

def get_sensors(
    world,
    agent,
):

    front_distance = 35.0

    direction = (
        1.0
        if agent.vx >= 0
        else -1.0
    )

    front_x = (
        agent.x
        +
        front_distance * direction
    )

    front_platform = (
        world.platform_at(
            front_x,
            agent.y,
        )
        is not None
    )

    hazard = world.is_hazard(
        agent.x,
        agent.y,
    )

    near_switch = world.near_switch(
        agent.x,
        agent.y,
    )

    moving_x = world.moving_x()

    moving_near = (
        abs(agent.x - moving_x)
        <
        50
    )

    nearest_orb = 1.0

    if world.orbs:

        distances = [

            np.linalg.norm(
                np.array([
                    agent.x,
                    agent.y,
                ])
                -
                orb
            )

            for orb in world.orbs
        ]

        nearest_orb = (
            min(distances)
            /
            400.0
        )

    return np.array([

        agent.x / 500.0,
        agent.y / 200.0,

        agent.vx / MAX_SPEED,
        agent.vy / 15.0,

        float(agent.grounded),

        float(
            agent.jumps / MAX_JUMPS
        ),

        float(
            agent.dash_timer > 0
        ),

        float(front_platform),

        float(hazard),

        float(near_switch),

        float(world.bridge_active),

        float(moving_near),

        nearest_orb,

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

        self.y = -110.0

        self.vx = 0.0
        self.vy = 0.0

        self.grounded = True

        self.jumps = 0

        self.dash_timer = 0

        self.steps = 0

        self.episode_reward = 0.0

        self.total_reward = 0.0

        self.successes = 0
        self.failures = 0

        self.last_action = ACTION_NONE

        self.last_event = "NONE"

        self.last_error = 0.0

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
        )

        error = learned["error"]

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

        goal = self.model.goal_system.generate(
            state,
            self.world,
            self.self_model,
            curiosity,
        )

        self.model.goal_system.update(
            goal.name,
            result["reward"],
            result["event"],
        )

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
    "Evolved Embodied Self-Organizing World Model v3"
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
        50,
        315,
        "EVOLVED SHARED WORLD MODEL v3",
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
    # GOALS
    # --------------------------------------------------------

    write_text(
        235,
        -105,
        "GOALS",
        11,
        "#ffffff",
    )

    y = -125

    for goal in shared_goal_system.goals.values():

        write_text(
            235,
            y,
            (
                f"{goal.name:<10} "
                f"P={goal.priority:.2f} "
                f"V={goal.progress:+.2f}"
            ),
            8,
            "#ffaa66",
        )

        y -= 15

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

        goal = shared_goal_system.generate(
            state,
            world,
            agent.self_model,
            0.5,
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
        "INITIALIZING EVOLVED WORLD MODEL v3",
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
