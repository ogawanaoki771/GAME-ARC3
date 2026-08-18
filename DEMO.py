import turtle
import random
import math
import numpy as np
# ============================================================
# Embodied Self-Organizing World Model
# MOVEMENT EVOLUTION VERSION
#
# 進化した身体:
#
#   acceleration
#   friction
#   air control
#   jump
#   double jump
#   dash
#   wall jump
#   moving platform
#   bounce
#   hazard
#   checkpoint
#
# AIは「位置」だけでなく、
#
#   x
#   y
#   vx
#   vy
#   grounded
#   wall
#   jumps
#   dash
#
# を身体状態として経験する。
# ============================================================
# ============================================================
# 1. Configuration
# ============================================================
SCREEN_W = 1100
SCREEN_H = 760
NUM_AGENTS = 3
STEPS_PER_EPISODE = 220
MAX_EPISODES = 20
DAY_DELAY = 30
SLEEP_DELAY = 600
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
# Physics
# ============================================================
MAX_SPEED = 8.0
GROUND_ACCEL = 0.9
AIR_ACCEL = 0.45
FRICTION = 0.82
AIR_FRICTION = 0.96
GRAVITY = 0.65
JUMP_POWER = 11.0
DOUBLE_JUMP_POWER = 9.0
DASH_SPEED = 18.0
DASH_TIME = 3
WALL_JUMP_X = 9.0
WALL_JUMP_Y = 10.0
MAX_JUMPS = 2
BODY_WIDTH = 12
BODY_HEIGHT = 20
# ============================================================
# Memory
# ============================================================
EVENT_SIM_THRESHOLD = 0.84
PLACE_RADIUS = 0.65
MAX_EVENTS = 600
# ============================================================
# 2. EventCell
# ============================================================
class EventCell:
    def __init__(
        self,
        event_id,
        state,
        action,
        expected,
        actual,
        surprise,
        reward,
        agent_id
    ):
        self.id = event_id
        self.state = np.array(
            state,
            dtype=float
        )
        self.action = int(action)
        self.expected = np.array(
            expected,
            dtype=float
        )
        self.actual = np.array(
            actual,
            dtype=float
        )
        self.surprise = float(surprise)
        self.reward = float(reward)
        self.agent_id = agent_id
        self.visits = 1
        self.energy = 1.0
        self.activation = 1.0
        self.place_id = None
        self.concepts = []
        # action -> target -> strength
        self.links = {}
    def similarity(
        self,
        state,
        action
    ):
        if self.action != action:
            return 0.0
        state = np.asarray(
            state,
            dtype=float
        )
        d = np.linalg.norm(
            self.state - state
        )
        return math.exp(
            -d * 1.5
        )
    def reinforce(
        self,
        actual,
        surprise,
        reward
    ):
        self.visits += 1
        self.energy = min(
            2.5,
            self.energy + 0.06
        )
        self.activation = min(
            2.0,
            self.activation + 0.15
        )
        actual = np.asarray(
            actual,
            dtype=float
        )
        self.actual = (
            0.75 * self.actual
            +
            0.25 * actual
        )
        self.surprise = (
            0.8 * self.surprise
            +
            0.2 * surprise
        )
        self.reward = (
            0.8 * self.reward
            +
            0.2 * reward
        )
    def strengthen(
        self,
        action,
        target_id,
        amount=0.15
    ):
        if action not in self.links:
            self.links[action] = {}
        old = self.links[action].get(
            target_id,
            0.0
        )
        self.links[action][target_id] = min(
            5.0,
            old + amount
        )
    def decay(self):
        self.activation *= 0.98
        self.energy *= 0.997
# ============================================================
# 3. PlaceCell
# ============================================================
class PlaceCell:
    def __init__(
        self,
        place_id
    ):
        self.id = place_id
        self.center = None
        self.events = []
        self.visits = 0
        self.mean_reward = 0.0
        self.mean_surprise = 0.0
        self.energy = 1.0
    def add(
        self,
        event
    ):
        if event.id not in self.events:
            self.events.append(
                event.id
            )
        if self.center is None:
            self.center = event.state.copy()
        else:
            self.center = (
                0.9 * self.center
                +
                0.1 * event.state
            )
        self.visits += 1
        self.mean_reward = (
            0.9 * self.mean_reward
            +
            0.1 * event.reward
        )
        self.mean_surprise = (
            0.9 * self.mean_surprise
            +
            0.1 * event.surprise
        )
        self.energy = min(
            2.0,
            self.energy + 0.02
        )
        event.place_id = self.id
    def distance(
        self,
        state
    ):
        if self.center is None:
            return 999999
        return float(
            np.linalg.norm(
                self.center
                -
                np.asarray(
                    state,
                    dtype=float
                )
            )
        )
# ============================================================
# 4. ConceptCell
# ============================================================
class ConceptCell:
    def __init__(
        self,
        concept_id,
        name
    ):
        self.id = concept_id
        self.name = name
        self.events = []
        self.places = []
        self.visits = 0
        self.value = 0.0
        self.activation = 0.0
        self.energy = 1.0
        self.links = {}
    def absorb(
        self,
        event
    ):
        if event.id not in self.events:
            self.events.append(
                event.id
            )
        if self.id not in event.concepts:
            event.concepts.append(
                self.id
            )
        self.visits += 1
        self.activation = (
            0.8 * self.activation
            +
            0.2
        )
        self.value = (
            0.9 * self.value
            +
            0.1 * event.reward
        )
        self.energy = min(
            2.0,
            self.energy + 0.02
        )
    def link(
        self,
        target
    ):
        self.links[target] = min(
            5.0,
            self.links.get(
                target,
                0.0
            ) + 0.05
        )
# ============================================================
# 5. Hippocampus
# ============================================================
class Hippocampus:
    def __init__(self):
        self.events = []
        self.places = []
        self.concepts = []
        self.next_event_id = 0
        self.next_place_id = 0
        self.next_concept_id = 0
        self.total_encodes = 0
    def get_event(
        self,
        event_id
    ):
        for event in self.events:
            if event.id == event_id:
                return event
        return None
    def get_concept(
        self,
        concept_id
    ):
        for concept in self.concepts:
            if concept.id == concept_id:
                return concept
        return None
    def encode(
        self,
        state,
        action,
        expected,
        actual,
        surprise,
        reward,
        agent_id
    ):
        self.total_encodes += 1
        best = None
        best_score = 0.0
        for event in self.events:
            score = event.similarity(
                state,
                action
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
                actual,
                surprise,
                reward
            )
            return best, False
        event = EventCell(
            self.next_event_id,
            state,
            action,
            expected,
            actual,
            surprise,
            reward,
            agent_id
        )
        self.next_event_id += 1
        self.events.append(
            event
        )
        return event, True
    def assign_place(
        self,
        event
    ):
        best = None
        best_distance = 999999
        for place in self.places:
            d = place.distance(
                event.state
            )
            if d < best_distance:
                best_distance = d
                best = place
        if (
            best is not None
            and
            best_distance < PLACE_RADIUS
        ):
            best.add(event)
            return best
        place = PlaceCell(
            self.next_place_id
        )
        self.next_place_id += 1
        place.add(event)
        self.places.append(
            place
        )
        return place
    def classify(
        self,
        event
    ):
        if event.reward >= 4:
            return "SUCCESS"
        if event.reward <= -5:
            return "FAILURE"
        if event.surprise >= 8:
            return "DANGER"
        if event.action == ACTION_JUMP:
            return "JUMP"
        if event.action == ACTION_DASH:
            return "DASH"
        if event.action == ACTION_LEFT:
            return "TURN_LEFT"
        if event.action == ACTION_RIGHT:
            return "TURN_RIGHT"
        if event.action == ACTION_BRAKE:
            return "BRAKE"
        return "MOVEMENT"
    def assign_concept(
        self,
        event,
        place
    ):
        name = self.classify(
            event
        )
        concept = None
        for c in self.concepts:
            if c.name == name:
                concept = c
                break
        if concept is None:
            concept = ConceptCell(
                self.next_concept_id,
                name
            )
            self.next_concept_id += 1
            self.concepts.append(
                concept
            )
        concept.absorb(
            event
        )
        if place.id not in concept.places:
            concept.places.append(
                place.id
            )
        return concept
    def organize(
        self,
        event
    ):
        place = self.assign_place(
            event
        )
        concept = self.assign_concept(
            event,
            place
        )
        return place, concept
    def link_concepts(
        self,
        previous,
        current
    ):
        if previous is None:
            return
        if current is None:
            return
        for p in previous.concepts:
            pc = self.get_concept(p)
            if pc is None:
                continue
            for c in current.concepts:
                if p != c:
                    pc.link(c)
    def replay(
        self,
        count=400
    ):
        if not self.events:
            return
        for _ in range(
            count
        ):
            event = random.choice(
                self.events
            )
            event.energy = min(
                2.5,
                event.energy + 0.025
            )
            if event.links:
                action = random.choice(
                    list(
                        event.links.keys()
                    )
                )
                targets = event.links[action]
                if targets:
                    target = random.choice(
                        list(
                            targets.keys()
                        )
                    )
                    targets[target] = min(
                        5.0,
                        targets[target] * 1.015
                    )
    def replay_concepts(
        self,
        count=200
    ):
        if not self.concepts:
            return
        for _ in range(
            count
        ):
            concept = random.choice(
                self.concepts
            )
            concept.energy = min(
                2.0,
                concept.energy + 0.02
            )
            if concept.links:
                target = random.choice(
                    list(
                        concept.links.keys()
                    )
                )
                concept.links[target] = min(
                    5.0,
                    concept.links[target] * 1.01
                )
    def metabolize(self):
        for event in self.events:
            event.decay()
        survivors = []
        for event in self.events:
            important = (
                event.visits >= 4
                or
                event.reward > 3
                or
                event.surprise > 8
            )
            if (
                important
                or
                event.energy > 0.15
            ):
                survivors.append(
                    event
                )
        self.events = survivors
        if len(self.events) > MAX_EVENTS:
            self.events.sort(
                key=lambda e: (
                    e.visits
                    +
                    e.energy
                    +
                    abs(e.reward)
                    +
                    e.surprise * 0.2
                ),
                reverse=True
            )
            self.events = self.events[
                :MAX_EVENTS
            ]
    def statistics(self):
        return {
            "events": len(
                self.events
            ),
            "places": len(
                self.places
            ),
            "concepts": len(
                self.concepts
            ),
        }
# ============================================================
# 6. Dream Simulator
# ============================================================
class DreamSimulator:
    def __init__(
        self,
        hippo
    ):
        self.hippo = hippo
    def rollout(
        self,
        start,
        steps=8
    ):
        path = []
        current = start
        for _ in range(
            steps
        ):
            path.append(
                current
            )
            candidates = []
            for action, targets in (
                current.links.items()
            ):
                for target_id, weight in (
                    targets.items()
                ):
                    target = self.hippo.get_event(
                        target_id
                    )
                    if target is not None:
                        candidates.append(
                            (
                                target,
                                weight
                            )
                        )
            if not candidates:
                break
            weights = np.array(
                [
                    max(
                        0.001,
                        x[1]
                    )
                    for x in candidates
                ],
                dtype=float
            )
            weights /= weights.sum()
            index = random.choices(
                range(
                    len(candidates)
                ),
                weights=weights
            )[0]
            current = candidates[
                index
            ][0]
        return path
    def evaluate(
        self,
        path
    ):
        score = 0.0
        for event in path:
            score += event.reward
            score += (
                event.energy
                * 0.3
            )
            score -= (
                event.surprise
                * 0.1
            )
        return score
    def best_dream(
        self,
        start,
        samples=10,
        steps=8
    ):
        best_path = []
        best_score = -999999
        for _ in range(
            samples
        ):
            path = self.rollout(
                start,
                steps
            )
            score = self.evaluate(
                path
            )
            if score > best_score:
                best_score = score
                best_path = path
        return best_path, best_score
# ============================================================
# 7. World Model
# ============================================================
class WorldModel:
    def __init__(
        self,
        hippo,
        dream
    ):
        self.hippo = hippo
        self.dream = dream
        self.last_event = {}
        self.error_history = []
        self.last_dream = []
        self.last_dream_score = 0.0
    def predict(
        self,
        state,
        action
    ):
        best = None
        best_score = 0.0
        for event in self.hippo.events:
            score = event.similarity(
                state,
                action
            )
            if score > best_score:
                best_score = score
                best = event
        if best is None:
            return None, None, None
        return (
            best.expected.copy(),
            best.surprise,
            best
        )
    def learn(
        self,
        agent_id,
        state,
        action,
        expected,
        actual,
        surprise,
        reward
    ):
        event, created = (
            self.hippo.encode(
                state,
                action,
                expected,
                actual,
                surprise,
                reward,
                agent_id
            )
        )
        place, concept = (
            self.hippo.organize(
                event
            )
        )
        previous = self.last_event.get(
            agent_id
        )
        if previous is not None:
            previous.strengthen(
                action,
                event.id
            )
            self.hippo.link_concepts(
                previous,
                event
            )
        self.last_event[
            agent_id
        ] = event
        self.error_history.append(
            surprise
        )
        if len(
            self.error_history
        ) > 1000:
            self.error_history.pop(
                0
            )
        return {
            "event": event,
            "place": place,
            "concept": concept,
            "created": created
        }
    def select_action(
        self,
        state
    ):
        scores = []
        for action in range(
            NUM_ACTIONS
        ):
            predicted, surprise, event = (
                self.predict(
                    state,
                    action
                )
            )
            if event is None:
                score = 3.0
            else:
                curiosity = (
                    surprise
                )
                novelty = (
                    1.5
                    if event.visits <= 2
                    else 0.0
                )
                reward = (
                    event.reward
                )
                danger = (
                    2.0
                    if event.surprise > 8
                    else 0.0
                )
                score = (
                    1.1 * curiosity
                    +
                    0.8 * novelty
                    +
                    0.7 * reward
                    -
                    danger
                )
            score += (
                random.random()
                * 0.6
            )
            scores.append(
                score
            )
        return int(
            np.argmax(scores)
        )
    def sleep_dream(
        self
    ):
        if not self.hippo.events:
            return
        ranked = sorted(
            self.hippo.events,
            key=lambda e: (
                e.energy
                +
                e.visits * 0.3
                +
                e.surprise * 0.15
                +
                abs(e.reward) * 0.2
            ),
            reverse=True
        )
        best_path = []
        best_score = -999999
        for start in ranked[:10]:
            path, score = (
                self.dream.best_dream(
                    start,
                    samples=10,
                    steps=8
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
# ============================================================
# 8. Advanced Physical World
# ============================================================
class TurtleWorld:
    def __init__(self):
        self.left = -510
        self.right = -20
        self.bottom = -280
        self.top = 270
        # Static platforms
        self.platforms = [
            [-510, -450, -130],
            [-430, -340, -60],
            [-320, -220, -10],
            [-200, -100, -90],
            [-70, -20, 40],
        ]
        # Hazards
        self.hazards = [
            [-385, -345, -130],
            [-285, -235, -10],
            [-165, -125, -90],
        ]
        # Moving platform
        self.moving_platform = {
            "x": -250.0,
            "y": 70.0,
            "width": 70.0,
            "amplitude": 80.0,
            "speed": 0.03
        }
        # Bounce platform
        self.bounce = [
            -460,
            -420,
            -60
        ]
        # Goal orb
        self.orbs = []
        # Switch
        self.switch_x = -250
        self.switch_y = -75
        self.bridge = False
        self.checkpoint = [
            -470.0,
            -110.0
        ]
        self.time = 0.0
        self.drawer = turtle.Turtle()
        self.drawer.hideturtle()
        self.drawer.penup()
        self.drawer.speed(0)
    def reset(self):
        self.time = 0.0
        self.bridge = False
        self.orbs = [
            np.array(
                [-125.0, 55.0]
            ),
            np.array(
                [-35.0, 100.0]
            )
        ]
        self.draw()
    def moving_x(self):
        p = self.moving_platform
        return (
            p["x"]
            +
            p["amplitude"]
            *
            math.sin(
                self.time
                *
                p["speed"]
                *
                20
            )
        )
    def update(self):
        self.time += 1.0
    def platform_at(
        self,
        x,
        y
    ):
        for x1, x2, py in self.platforms:
            if (
                x1 <= x <= x2
                and
                abs(y - py) < 20
            ):
                return py
        # Bridge
        if self.bridge:
            if (
                -350 <= x <= -120
                and
                abs(y + 75) < 20
            ):
                return -75
        # Moving platform
        mx = self.moving_x()
        p = self.moving_platform
        if (
            mx - p["width"] / 2
            <= x
            <=
            mx + p["width"] / 2
            and
            abs(
                y - p["y"]
            ) < 20
        ):
            return p["y"]
        # Bounce
        bx1, bx2, by = self.bounce
        if (
            bx1 <= x <= bx2
            and
            abs(y - by) < 20
        ):
            return by
        return None
    def wall_near(
        self,
        x
    ):
        left_wall = (
            x <= self.left + 14
        )
        right_wall = (
            x >= self.right - 14
        )
        return (
            left_wall,
            right_wall
        )
    def is_hazard(
        self,
        x,
        y
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
        y
    ):
        return (
            math.hypot(
                x - self.switch_x,
                y - self.switch_y
            )
            <
            24
        )
    def collect_orb(
        self,
        x,
        y
    ):
        collected = 0
        remaining = []
        for orb in self.orbs:
            d = np.linalg.norm(
                np.array(
                    [x, y]
                )
                -
                orb
            )
            if d < 22:
                collected += 1
            else:
                remaining.append(
                    orb
                )
        self.orbs = remaining
        return collected
    def step(
        self,
        agent,
        action
    ):
        x = agent.x
        y = agent.y
        vx = agent.vx
        vy = agent.vy
        grounded = agent.grounded
        jumps = agent.jumps
        dash_timer = agent.dash_timer
        reward = -0.015
        event = "NORMAL"
        triggered = False
        # ====================================================
        # ACTION
        # ====================================================
        if action == ACTION_LEFT:
            if grounded:
                vx -= GROUND_ACCEL
            else:
                vx -= AIR_ACCEL
        elif action == ACTION_RIGHT:
            if grounded:
                vx += GROUND_ACCEL
            else:
                vx += AIR_ACCEL
        elif action == ACTION_JUMP:
            # Normal jump
            if grounded:
                vy = JUMP_POWER
                grounded = False
                jumps = 1
                event = "JUMP"
            # Double jump
            elif jumps < MAX_JUMPS:
                vy = DOUBLE_JUMP_POWER
                jumps += 1
                event = "DOUBLE_JUMP"
        elif action == ACTION_DASH:
            if dash_timer <= 0:
                direction = (
                    1
                    if agent.heading >= 0
                    else
                    -1
                )
                if (
                    abs(vx) > 1
                ):
                    direction = (
                        1
                        if vx > 0
                        else
                        -1
                    )
                vx = (
                    direction
                    *
                    DASH_SPEED
                )
                dash_timer = DASH_TIME
                event = "DASH"
        elif action == ACTION_BRAKE:
            vx *= 0.35
            event = "BRAKE"
        elif action == ACTION_WAIT:
            vx *= 0.90
        # ====================================================
        # Clamp velocity
        # ====================================================
        vx = np.clip(
            vx,
            -MAX_SPEED,
            MAX_SPEED
        )
        # ====================================================
        # Horizontal friction
        # ====================================================
        if grounded:
            vx *= FRICTION
        else:
            vx *= AIR_FRICTION
        # ====================================================
        # Gravity
        # ====================================================
        if not grounded:
            vy -= GRAVITY
            y += vy
        # ====================================================
        # Horizontal movement
        # ====================================================
        x += vx
        # ====================================================
        # Walls
        # ====================================================
        left_wall, right_wall = (
            self.wall_near(x)
        )
        if left_wall:
            x = self.left + 14
            vx *= -0.35
            event = "WALL"
        if right_wall:
            x = self.right - 14
            vx *= -0.35
            event = "WALL"
        # ====================================================
        # Wall jump
        # ====================================================
        if (
            action == ACTION_JUMP
            and
            not grounded
            and
            (left_wall or right_wall)
        ):
            vy = WALL_JUMP_Y
            if left_wall:
                vx = WALL_JUMP_X
            else:
                vx = -WALL_JUMP_X
            event = "WALL_JUMP"
        # ====================================================
        # Hazard
        # ====================================================
        if self.is_hazard(
            x,
            y
        ):
            reward -= 8
            event = "DANGER"
            triggered = True
            x, y = self.checkpoint
            vx = 0
            vy = 0
            grounded = True
            jumps = 0
        # ====================================================
        # Switch
        # ====================================================
        if self.near_switch(
            x,
            y
        ):
            self.bridge = (
                not self.bridge
            )
            reward += 2.0
            event = "SWITCH"
            triggered = True
        # ====================================================
        # Platform collision
        # ====================================================
        platform_y = self.platform_at(
            x,
            y
        )
        if (
            platform_y is not None
            and
            vy <= 0
            and
            y <= platform_y + 20
        ):
            y = platform_y + 20
            # Bounce platform
            if (
                self.bounce[0]
                <= x
                <= self.bounce[1]
            ):
                vy = 15
                grounded = False
                jumps = 1
                event = "BOUNCE"
                reward += 0.5
            else:
                vy = 0
                grounded = True
                jumps = 0
        elif y < -245:
            reward -= 10
            event = "FALL"
            triggered = True
            x, y = self.checkpoint
            vx = 0
            vy = 0
            grounded = True
            jumps = 0
        else:
            grounded = False
        # ====================================================
        # Orb
        # ====================================================
        collected = self.collect_orb(
            x,
            y
        )
        if collected > 0:
            reward += (
                5
                *
                collected
            )
            event = "SUCCESS"
            triggered = True
        # ====================================================
        # Store
        # ====================================================
        agent.x = x
        agent.y = y
        agent.vx = vx
        agent.vy = vy
        agent.grounded = grounded
        agent.jumps = jumps
        agent.dash_timer = max(
            0,
            dash_timer - 1
        )
        agent.turtle.goto(
            x,
            y
        )
        return {
            "reward": float(
                reward
            ),
            "event": event,
            "triggered": triggered
        }
    def draw(self):
        self.drawer.clear()
        # Border
        self.drawer.color(
            "#444455"
        )
        self.drawer.goto(
            self.left,
            self.bottom
        )
        self.drawer.pendown()
        self.drawer.goto(
            self.right,
            self.bottom
        )
        self.drawer.goto(
            self.right,
            self.top
        )
        self.drawer.goto(
            self.left,
            self.top
        )
        self.drawer.goto(
            self.left,
            self.bottom
        )
        self.drawer.penup()
        # Platforms
        self.drawer.color(
            "#00dddd"
        )
        for x1, x2, y in self.platforms:
            self.drawer.goto(
                x1,
                y
            )
            self.drawer.pendown()
            self.drawer.goto(
                x2,
                y
            )
            self.drawer.penup()
        # Bridge
        if self.bridge:
            self.drawer.color(
                "#ffaa00"
            )
            self.drawer.goto(
                -350,
                -75
            )
            self.drawer.pendown()
            self.drawer.goto(
                -120,
                -75
            )
            self.drawer.penup()
        # Moving platform
        mx = self.moving_x()
        p = self.moving_platform
        self.drawer.color(
            "#00ff88"
        )
        self.drawer.goto(
            mx - p["width"] / 2,
            p["y"]
        )
        self.drawer.pendown()
        self.drawer.goto(
            mx + p["width"] / 2,
            p["y"]
        )
        self.drawer.penup()
        # Hazards
        self.drawer.color(
            "#ff3344"
        )
        for x1, x2, y in self.hazards:
            self.drawer.goto(
                x1,
                y
            )
            self.drawer.pendown()
            self.drawer.goto(
                x2,
                y
            )
            self.drawer.penup()
        # Bounce
        self.drawer.color(
            "#bb88ff"
        )
        self.drawer.goto(
            self.bounce[0],
            self.bounce[2]
        )
        self.drawer.pendown()
        self.drawer.goto(
            self.bounce[1],
            self.bounce[2]
        )
        self.drawer.penup()
        # Switch
        self.drawer.goto(
            self.switch_x,
            self.switch_y
        )
        self.drawer.dot(
            15,
            "#ff00ff"
            if self.bridge
            else
            "#555555"
        )
        # Orbs
        self.drawer.color(
            "#ffff00"
        )
        for orb in self.orbs:
            self.drawer.goto(
                orb[0],
                orb[1]
            )
            self.drawer.dot(
                11
            )
# ============================================================
# 9. Local Sensors
# ============================================================
def sense(
    world,
    agent
):
    front = agent.x + (
        35
        *
        (
            1
            if agent.vx >= 0
            else
            -1
        )
    )
    front_platform = (
        world.platform_at(
            front,
            agent.y
        )
        is not None
    )
    left_wall, right_wall = (
        world.wall_near(
            agent.x
        )
    )
    hazard = world.is_hazard(
        agent.x,
        agent.y
    )
    near_switch = (
        world.near_switch(
            agent.x,
            agent.y
        )
    )
    moving_x = (
        world.moving_x()
    )
    moving_near = (
        abs(
            agent.x - moving_x
        )
        <
        45
    )
    nearest_orb = 1.0
    if world.orbs:
        distances = [
            np.linalg.norm(
                np.array(
                    [
                        agent.x,
                        agent.y
                    ]
                )
                -
                orb
            )
            for orb in world.orbs
        ]
        nearest_orb = min(
            distances
        ) / 400
    # --------------------------------------------------------
    # Body/world state
    # --------------------------------------------------------
    return np.array(
        [
            agent.x / 500.0,
            agent.y / 200.0,
            agent.vx / MAX_SPEED,
            agent.vy / 15.0,
            float(agent.grounded),
            float(agent.jumps)
            /
            MAX_JUMPS,
            float(
                agent.dash_timer > 0
            ),
            float(left_wall),
            float(right_wall),
            float(front_platform),
            float(hazard),
            float(near_switch),
            float(world.bridge),
            float(moving_near),
            nearest_orb,
        ],
        dtype=float
    )
# ============================================================
# 10. Agent
# ============================================================
class Agent:
    def __init__(
        self,
        agent_id,
        color,
        world,
        model
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
        self.x = (
            -480
            +
            agent_id * 20
        )
        self.y = -110
        self.vx = 0.0
        self.vy = 0.0
        self.heading = 1
        self.grounded = True
        self.jumps = 0
        self.dash_timer = 0
        self.steps = 0
        self.reward = 0.0
        self.episode_reward = 0.0
        self.successes = 0
        self.failures = 0
        self.last_action = ACTION_NONE
        self.last_event = "NONE"
        self.turtle.goto(
            self.x,
            self.y
        )
    def reset(self):
        self.x = (
            -480
            +
            self.id * 20
        )
        self.y = -110
        self.vx = 0.0
        self.vy = 0.0
        self.grounded = True
        self.jumps = 0
        self.dash_timer = 0
        self.steps = 0
        self.episode_reward = 0.0
        self.successes = 0
        self.failures = 0
        self.last_action = ACTION_NONE
        self.last_event = "NONE"
        self.turtle.goto(
            self.x,
            self.y
        )
    def step(self):
        # ----------------------------------------------------
        # Sense
        # ----------------------------------------------------
        state = sense(
            self.world,
            self
        )
        # ----------------------------------------------------
        # Decide
        # ----------------------------------------------------
        action = (
            self.model.select_action(
                state
            )
        )
        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------
        predicted, _, _ = (
            self.model.predict(
                state,
                action
            )
        )
        if predicted is None:
            predicted = np.zeros(
                len(state)
            )
        # ----------------------------------------------------
        # Act
        # ----------------------------------------------------
        result = self.world.step(
            self,
            action
        )
        # ----------------------------------------------------
        # Observe
        # ----------------------------------------------------
        next_state = sense(
            self.world,
            self
        )
        actual = (
            next_state
            -
            state
        )
        # ----------------------------------------------------
        # Prediction error
        # ----------------------------------------------------
        surprise = float(
            np.linalg.norm(
                predicted
                -
                actual
            )
        )
        if result["triggered"]:
            surprise += 5
        # ----------------------------------------------------
        # Learn
        # ----------------------------------------------------
        self.model.learn(
            self.id,
            state,
            action,
            predicted,
            actual,
            surprise,
            result["reward"]
        )
        # ----------------------------------------------------
        # Stats
        # ----------------------------------------------------
        self.steps += 1
        self.last_action = action
        self.last_event = (
            result["event"]
        )
        self.reward += (
            result["reward"]
        )
        self.episode_reward += (
            result["reward"]
        )
        if result["event"] == "SUCCESS":
            self.successes += 1
        if result["event"] in (
            "DANGER",
            "FALL"
        ):
            self.failures += 1
# ============================================================
# 11. System
# ============================================================
screen = turtle.Screen()
screen.setup(
    SCREEN_W,
    SCREEN_H
)
screen.bgcolor(
    "#0b0b12"
)
screen.title(
    "Embodied World Model - Movement Evolution"
)
screen.tracer(False)
world = TurtleWorld()
hippocampus = Hippocampus()
dream = DreamSimulator(
    hippocampus
)
model = WorldModel(
    hippocampus,
    dream
)
# ============================================================
# 12. Agents
# ============================================================
colors = [
    "#00ffff",
    "#00ff7f",
    "#ffa500"
]
agents = []
for i in range(
    NUM_AGENTS
):
    agents.append(
        Agent(
            i,
            colors[i],
            world,
            model
        )
    )
# ============================================================
# 13. Visualization
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
def write_text(
    x,
    y,
    message,
    size=10,
    color="#dddddd"
):
    text_drawer.goto(
        x,
        y
    )
    text_drawer.color(
        color
    )
    text_drawer.write(
        message,
        font=(
            "Arial",
            size,
            "normal"
        )
    )
# ============================================================
# 14. Visualization
# ============================================================
def draw_model():
    model_drawer.clear()
    text_drawer.clear()
    dream_drawer.clear()
    stats = (
        hippocampus.statistics()
    )
    # --------------------------------------------------------
    # Titles
    # --------------------------------------------------------
    write_text(
        -515,
        310,
        "EMBODIED WORLD",
        15,
        "#ffffff"
    )
    write_text(
        70,
        310,
        "SHARED WORLD MODEL",
        15,
        "#ffffff"
    )
    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------
    write_text(
        70,
        285,
        f"Events   : {stats['events']}",
        10
    )
    write_text(
        70,
        268,
        f"Places   : {stats['places']}",
        10
    )
    write_text(
        70,
        251,
        f"Concepts : {stats['concepts']}",
        10
    )
    if model.error_history:
        error = np.mean(
            model.error_history[-50:]
        )
    else:
        error = 0.0
    write_text(
        70,
        234,
        f"Surprise : {error:.3f}",
        10,
        "#ffaa00"
    )
    # --------------------------------------------------------
    # Event graph
    # --------------------------------------------------------
    events = sorted(
        hippocampus.events,
        key=lambda e: (
            e.energy
            +
            e.visits
            +
            abs(e.reward)
        ),
        reverse=True
    )[:70]
    positions = {}
    for i, event in enumerate(
        events
    ):
        x = (
            120
            +
            (
                i % 10
            )
            *
            42
        )
        y = (
            180
            -
            (
                i // 10
            )
            *
            43
        )
        positions[
            event.id
        ] = (
            x,
            y
        )
    # --------------------------------------------------------
    # Links
    # --------------------------------------------------------
    model_drawer.color(
        "#334455"
    )
    for event in events:
        if event.id not in positions:
            continue
        sx, sy = positions[
            event.id
        ]
        count = 0
        for targets in (
            event.links.values()
        ):
            for target_id in targets:
                if target_id not in positions:
                    continue
                tx, ty = positions[
                    target_id
                ]
                model_drawer.goto(
                    sx,
                    sy
                )
                model_drawer.pendown()
                model_drawer.goto(
                    tx,
                    ty
                )
                model_drawer.penup()
                count += 1
                if count >= 4:
                    break
            if count >= 4:
                break
    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------
    for event in events:
        x, y = positions[
            event.id
        ]
        if event.reward >= 4:
            color = "#00ff7f"
        elif event.reward <= -5:
            color = "#ff3344"
        elif event.surprise >= 8:
            color = "#ff9900"
        elif event.action in (
            ACTION_JUMP,
            ACTION_DASH
        ):
            color = "#bb88ff"
        else:
            color = "#66ccff"
        size = min(
            18,
            max(
                5,
                int(
                    5
                    +
                    event.visits
                )
            )
        )
        model_drawer.goto(
            x,
            y
        )
        model_drawer.dot(
            size,
            color
        )
    # --------------------------------------------------------
    # Agent information
    # --------------------------------------------------------
    y = 180
    for agent in agents:
        write_text(
            -515,
            y,
            (
                f"A{agent.id} "
                f"reward={agent.episode_reward:+.1f} "
                f"vx={agent.vx:+.1f} "
                f"vy={agent.vy:+.1f} "
                f"{ACTION_NAMES.get(agent.last_action)} "
                f"{agent.last_event}"
            ),
            9,
            agent.color
        )
        y -= 20
    # --------------------------------------------------------
    # Concepts
    # --------------------------------------------------------
    y = -120
    write_text(
        70,
        y,
        "CONCEPTS",
        11,
        "#ffffff"
    )
    y -= 20
    concept_colors = {
        "SUCCESS": "#00ff7f",
        "FAILURE": "#ff3344",
        "DANGER": "#ff9900",
        "JUMP": "#bb88ff",
        "DOUBLE_JUMP": "#bb88ff",
        "DASH": "#ffff00",
        "BRAKE": "#ff66aa",
        "MOVEMENT": "#66ccff",
        "TURN_LEFT": "#66ccff",
        "TURN_RIGHT": "#66ccff"
    }
    concepts = sorted(
        hippocampus.concepts,
        key=lambda c: c.visits,
        reverse=True
    )[:10]
    for concept in concepts:
        color = concept_colors.get(
            concept.name,
            "#cccccc"
        )
        write_text(
            70,
            y,
            (
                f"{concept.name:<14}"
                f" n={concept.visits:<4}"
                f" V={concept.value:+.2f}"
            ),
            9,
            color
        )
        y -= 17
    # --------------------------------------------------------
    # Dream
    # --------------------------------------------------------
    if model.last_dream:
        write_text(
            70,
            -285,
            "LAST DREAM",
            11,
            "#aa88ff"
        )
        x = 160
        for event in model.last_dream[:14]:
            if event.reward >= 4:
                color = "#00ff7f"
            elif event.reward <= -5:
                color = "#ff3344"
            elif event.surprise >= 8:
                color = "#ff9900"
            else:
                color = "#aa88ff"
            dream_drawer.goto(
                x,
                -285
            )
            dream_drawer.dot(
                10,
                color
            )
            x += 24
        write_text(
            70,
            -310,
            (
                f"dream score="
                f"{model.last_dream_score:+.2f}"
            ),
            9,
            "#9999bb"
        )
# ============================================================
# 15. Lifecycle
# ============================================================
episode = 0
step_count = 0
finished = False
def reset_episode():
    world.reset()
    model.last_event.clear()
    for agent in agents:
        agent.reset()
def sleep_phase():
    global episode
    global step_count
    global finished
    # --------------------------------------------------------
    # Replay
    # --------------------------------------------------------
    hippocampus.replay(
        400
    )
    hippocampus.replay_concepts(
        200
    )
    # --------------------------------------------------------
    # Dream
    # --------------------------------------------------------
    model.sleep_dream()
    # --------------------------------------------------------
    # Metabolism
    # --------------------------------------------------------
    hippocampus.metabolize()
    # --------------------------------------------------------
    # Reset temporal pointers
    # --------------------------------------------------------
    model.last_event.clear()
    # --------------------------------------------------------
    # Update screen
    # --------------------------------------------------------
    draw_model()
    screen.update()
    episode += 1
    if episode >= MAX_EPISODES:
        finished = True
        write_text(
            -180,
            -340,
            "SIMULATION FINISHED",
            18,
            "#ffffff"
        )
        screen.update()
        return
    step_count = 0
    screen.ontimer(
        start_day,
        SLEEP_DELAY
    )
def run_day():
    global step_count
    if finished:
        return
    # --------------------------------------------------------
    # World update
    # --------------------------------------------------------
    world.update()
    # --------------------------------------------------------
    # Agents
    # --------------------------------------------------------
    for agent in agents:
        agent.step()
    # --------------------------------------------------------
    # Drawing
    # --------------------------------------------------------
    world.draw()
    draw_model()
    write_text(
        -515,
        285,
        (
            f"DAY "
            f"{episode + 1}/{MAX_EPISODES} "
            f"STEP "
            f"{step_count}/{STEPS_PER_EPISODE}"
        ),
        11,
        "#00ff7f"
    )
    screen.update()
    # --------------------------------------------------------
    # Continue
    # --------------------------------------------------------
    step_count += 1
    if step_count < STEPS_PER_EPISODE:
        screen.ontimer(
            run_day,
            DAY_DELAY
        )
    else:
        screen.ontimer(
            sleep_phase,
            200
        )
def start_day():
    if finished:
        return
    reset_episode()
    run_day()
# ============================================================
# 16. Start
# ============================================================
reset_episode()
world.draw()
draw_model()
screen.update()
screen.ontimer(
    start_day,
    500
)
turtle.done()

    

 
