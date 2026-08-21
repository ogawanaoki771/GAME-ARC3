# ============================================================
# VISIBLE LAW WORLD
#
# VERSION 2
#
# ------------------------------------------------------------
# CORE:
#
#     World
#       ↓
#     ΔWorld
#       ↓
#     Transformation
#       ↓
#     TransformationGraph
#       ↓
#     WorldModel
#       ↓
#     Action
#
# ------------------------------------------------------------
# DESIGN PRINCIPLE
#
#   「法則は隠された設定ではなく、
#     世界の変化として現れ、
#     観察によって可視化され、
#     行動によって検証される。」
#
# ------------------------------------------------------------
#
# NO REWARD
# NO SCORE
# NO FIXED GOAL
#
# The world itself is the experiment.
# ============================================================

import turtle
import math
import random
from dataclasses import dataclass, field
from collections import defaultdict, deque


# ============================================================
# CONFIG
# ============================================================

SCREEN_W = 1200
SCREEN_H = 760

WORLD_LEFT = -560
WORLD_RIGHT = 260
WORLD_BOTTOM = -270
WORLD_TOP = 270

PANEL_LEFT = 290

FRAME_MS = 35

PLAYER_SPEED = 4.5

OBSERVATION_DISTANCE = 150

TRANSFORMATION_MEMORY_LIMIT = 500
GRAPH_EDGE_LIMIT = 1000

RANDOM_SEED = 2026

random.seed(RANDOM_SEED)


# ============================================================
# SCREEN
# ============================================================

screen = turtle.Screen()

screen.setup(
    SCREEN_W,
    SCREEN_H
)

screen.title(
    "VISIBLE LAW WORLD : Transformation World Model"
)

screen.bgcolor(
    "#0b1016"
)

screen.tracer(
    0,
    0
)


# ============================================================
# TURTLE FACTORY
# ============================================================

def make_turtle():

    t = turtle.Turtle()

    t.hideturtle()
    t.penup()
    t.speed(0)

    return t


world_pen = make_turtle()
ui_pen = make_turtle()
law_pen = make_turtle()


# ============================================================
# UTILITY
# ============================================================

def distance(x1, y1, x2, y2):

    dx = x2 - x1
    dy = y2 - y1

    return math.sqrt(
        dx * dx + dy * dy
    )


def clamp(value, low, high):

    return max(
        low,
        min(
            high,
            value
        )
    )


# ============================================================
# WORLD OBJECT
# ============================================================

@dataclass
class WorldObject:

    object_id: str

    kind: str

    x: float
    y: float

    vx: float = 0.0
    vy: float = 0.0

    state: int = 0

    timer: int = 0

    active: bool = True

    radius: float = 18

    color: str = "white"

    phase: float = 0.0

    def snapshot(self):

        return {

            "x": round(self.x, 2),

            "y": round(self.y, 2),

            "vx": round(self.vx, 2),

            "vy": round(self.vy, 2),

            "state": self.state,

            "active": self.active

        }


# ============================================================
# PLAYER
# ============================================================

@dataclass
class Player:

    x: float = -350

    y: float = -100

    vx: float = 0.0

    vy: float = 0.0

    radius: float = 10

    last_action: str = "WAIT"

    def snapshot(self):

        return {

            "x": round(self.x, 2),

            "y": round(self.y, 2),

            "vx": round(self.vx, 2),

            "vy": round(self.vy, 2)

        }

    def move(self):

        self.x += self.vx
        self.y += self.vy

        self.vx *= 0.82
        self.vy *= 0.82

        self.x = clamp(
            self.x,
            WORLD_LEFT + self.radius,
            WORLD_RIGHT - self.radius
        )

        self.y = clamp(
            self.y,
            WORLD_BOTTOM + self.radius,
            WORLD_TOP - self.radius
        )


# ============================================================
# WORLD STATE
# ============================================================

@dataclass
class WorldState:

    time: int

    player: dict

    objects: dict

    nearby_objects: tuple = field(
        default_factory=tuple
    )

    def compact_signature(self):

        player_sig = (

            round(self.player["x"] / 20),

            round(self.player["y"] / 20),

            round(self.player["vx"]),

            round(self.player["vy"])

        )

        object_sig = []

        for oid in sorted(
            self.objects.keys()
        ):

            obj = self.objects[oid]

            object_sig.append(

                (

                    oid,

                    round(obj["x"] / 20),

                    round(obj["y"] / 20),

                    round(obj["vx"]),

                    round(obj["vy"]),

                    obj["state"],

                    obj["active"]

                )

            )

        return (

            player_sig,

            tuple(object_sig)

        )


# ============================================================
# ΔWORLD
# ============================================================

@dataclass
class WorldDelta:

    changed_objects: dict = field(
        default_factory=dict
    )

    player_changed: bool = False

    relation_changes: list = field(
        default_factory=list
    )

    def is_empty(self):

        return (

            not self.changed_objects

            and not self.player_changed

            and not self.relation_changes

        )


# ============================================================
# TRANSFORMATION
# ============================================================

@dataclass
class Transformation:

    transformation_id: int

    time: int

    action: str

    source_signature: tuple

    target_signature: tuple

    delta: WorldDelta

    context: tuple

    frequency: int = 1

    confidence: float = 0.0

    def key(self):

        changed = tuple(
            sorted(
                self.delta.changed_objects.keys()
            )
        )

        relations = tuple(
            self.delta.relation_changes
        )

        return (

            self.action,

            changed,

            relations

        )


# ============================================================
# TRANSFORMATION GRAPH
# ============================================================

class TransformationGraph:

    def __init__(self):

        self.nodes = {}

        self.edges = defaultdict(
            dict
        )

        self.edge_counts = defaultdict(
            int
        )

        self.transformations = []

        self.next_id = 0

    # --------------------------------------------------------
    # Node
    # --------------------------------------------------------

    def add_node(
        self,
        signature
    ):

        if signature not in self.nodes:

            self.nodes[signature] = {

                "visits": 0

            }

        self.nodes[
            signature
        ]["visits"] += 1

    # --------------------------------------------------------
    # Transformation
    # --------------------------------------------------------

    def add_transformation(
        self,
        transformation
    ):

        self.transformations.append(
            transformation
        )

        if len(
            self.transformations
        ) > GRAPH_EDGE_LIMIT:

            self.transformations.pop(0)

        key = transformation.key()

        self.edge_counts[key] += 1

        transformation.frequency = (
            self.edge_counts[key]
        )

        transformation.confidence = min(
            1.0,
            transformation.frequency / 8.0
        )

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    def frequent_transformations(
        self,
        minimum=2
    ):

        result = []

        for t in self.transformations:

            if t.frequency >= minimum:

                result.append(t)

        return result

    # --------------------------------------------------------
    # Recent
    # --------------------------------------------------------

    def recent(
        self,
        n=10
    ):

        return self.transformations[-n:]


# ============================================================
# LAW
# ============================================================

@dataclass
class Law:

    key: str

    description: str

    evidence: int = 0

    confidence: float = 0.0

    last_seen: int = 0

    examples: deque = field(
        default_factory=lambda:
        deque(maxlen=8)
    )

    def observe(
        self,
        time
    ):

        self.evidence += 1

        self.last_seen = time

        self.confidence = min(
            1.0,
            self.evidence / 8.0
        )


# ============================================================
# WORLD MODEL
# ============================================================

class WorldModel:

    def __init__(self):

        self.previous_state = None

        self.previous_action = "WAIT"

        self.graph = (
            TransformationGraph()
        )

        self.laws = {}

        self.memory = deque(
            maxlen=TRANSFORMATION_MEMORY_LIMIT
        )

        self.predictions = []

        self.prediction_errors = []

    # --------------------------------------------------------
    # Capture
    # --------------------------------------------------------

    def capture_world(
        self,
        world
    ):

        objects = {}

        for oid, obj in world.objects.items():

            objects[oid] = (
                obj.snapshot()
            )

        nearby = []

        for oid, obj in world.objects.items():

            d = distance(

                world.player.x,
                world.player.y,

                obj.x,
                obj.y

            )

            if d <= OBSERVATION_DISTANCE:

                nearby.append(
                    oid
                )

        return WorldState(

            time=world.time,

            player=world.player.snapshot(),

            objects=objects,

            nearby_objects=tuple(
                sorted(nearby)
            )

        )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    def compute_delta(
        self,
        before,
        after
    ):

        delta = WorldDelta()

        # ------------------------------------
        # Player
        # ------------------------------------

        if before.player != after.player:

            delta.player_changed = True

        # ------------------------------------
        # Objects
        # ------------------------------------

        for oid in after.objects:

            if oid not in before.objects:

                delta.changed_objects[
                    oid
                ] = (

                    None,

                    after.objects[oid]

                )

                continue

            old = before.objects[oid]

            new = after.objects[oid]

            if old != new:

                delta.changed_objects[
                    oid
                ] = (

                    old,

                    new

                )

                # velocity reversal

                if (

                    old["vx"] != 0

                    and new["vx"] != 0

                    and (
                        old["vx"] > 0
                        and new["vx"] < 0
                    )

                    or

                    (
                        old["vx"] < 0
                        and new["vx"] > 0
                    )

                ):

                    delta.relation_changes.append(

                        (
                            oid,
                            "VELOCITY_REVERSAL"
                        )

                    )

                # state change

                if (
                    old["state"]
                    != new["state"]
                ):

                    delta.relation_changes.append(

                        (
                            oid,
                            "STATE_CHANGE"
                        )

                    )

        # ------------------------------------
        # Nearby relation
        # ------------------------------------

        if (
            before.nearby_objects
            != after.nearby_objects
        ):

            delta.relation_changes.append(

                (
                    "WORLD",
                    "NEARBY_RELATION_CHANGE"
                )

            )

        return delta

    # --------------------------------------------------------
    # Learn
    # --------------------------------------------------------

    def observe_transition(
        self,
        before,
        after,
        action
    ):

        if before is None:

            self.previous_state = after

            return None

        delta = self.compute_delta(
            before,
            after
        )

        source = (
            before.compact_signature()
        )

        target = (
            after.compact_signature()
        )

        self.graph.add_node(
            source
        )

        self.graph.add_node(
            target
        )

        if delta.is_empty():

            self.previous_state = after

            return None

        context = (

            tuple(
                before.nearby_objects
            ),

            tuple(
                after.nearby_objects
            )

        )

        transformation = Transformation(

            transformation_id=
                self.graph.next_id,

            time=after.time,

            action=action,

            source_signature=source,

            target_signature=target,

            delta=delta,

            context=context

        )

        self.graph.next_id += 1

        self.graph.add_transformation(
            transformation
        )

        self.memory.append(
            transformation
        )

        self.detect_laws(
            transformation
        )

        self.previous_state = after

        return transformation

    # --------------------------------------------------------
    # Law discovery
    # --------------------------------------------------------

    def detect_laws(
        self,
        transformation
    ):

        delta = transformation.delta

        # ------------------------------------
        # Velocity reversal
        # ------------------------------------

        for relation in (
            delta.relation_changes
        ):

            if (
                relation[1]
                == "VELOCITY_REVERSAL"
            ):

                key = (
                    f"{relation[0]}:"
                    "VELOCITY_REVERSAL"
                )

                if key not in self.laws:

                    self.laws[key] = Law(

                        key,

                        "対象の速度方向が反転する"

                    )

                law = self.laws[key]

                law.observe(
                    transformation.time
                )

                law.examples.append(
                    transformation.transformation_id
                )

        # ------------------------------------
        # State change
        # ------------------------------------

        for relation in (
            delta.relation_changes
        ):

            if (
                relation[1]
                == "STATE_CHANGE"
            ):

                key = (
                    f"{relation[0]}:"
                    "STATE_CHANGE"
                )

                if key not in self.laws:

                    self.laws[key] = Law(

                        key,

                        "対象の内部状態が変化する"

                    )

                law = self.laws[key]

                law.observe(
                    transformation.time
                )

                law.examples.append(
                    transformation.transformation_id
                )

        # ------------------------------------
        # Nearby relation
        # ------------------------------------

        for relation in (
            delta.relation_changes
        ):

            if (
                relation[1]
                == "NEARBY_RELATION_CHANGE"
            ):

                key = (
                    "WORLD:"
                    "NEARBY_RELATION"
                )

                if key not in self.laws:

                    self.laws[key] = Law(

                        key,

                        "プレイヤーと対象の空間関係が変化する"

                    )

                law = self.laws[key]

                law.observe(
                    transformation.time
                )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    def predict_action(
        self,
        state,
        action
    ):

        candidates = []

        for t in self.memory:

            if (
                t.action == action
            ):

                candidates.append(t)

        if not candidates:

            return None

        return candidates[-1]

    # --------------------------------------------------------
    # Choose action
    # --------------------------------------------------------

    def choose_action(
        self,
        world
    ):

        state = self.capture_world(
            world
        )

        actions = [

            "LEFT",
            "RIGHT",
            "UP",
            "DOWN",
            "WAIT"

        ]

        # ------------------------------------------------
        # Exploration score
        #
        # No reward.
        #
        # Score means only:
        # "How informative could this action be?"
        # ------------------------------------------------

        best_action = "WAIT"

        best_score = -999999

        for action in actions:

            score = 0.0

            # --------------------------------------------
            # Novel action
            # --------------------------------------------

            count = 0

            for t in self.memory:

                if t.action == action:

                    count += 1

            score += (
                1.0
                / (1.0 + count)
            )

            # --------------------------------------------
            # Transformation diversity
            # --------------------------------------------

            related = [

                t

                for t in self.memory

                if t.action == action

            ]

            if related:

                unique = len(

                    set(
                        t.key()
                        for t in related
                    )

                )

                score += (
                    unique * 0.15
                )

            # --------------------------------------------
            # Recent uncertainty
            # --------------------------------------------

            if action == "WAIT":

                score += 0.05

            if score > best_score:

                best_score = score

                best_action = action

        return best_action


# ============================================================
# WORLD
# ============================================================

class World:

    def __init__(self):

        self.time = 0

        self.player = Player()

        self.objects = {}

        self.create_stage()

    # --------------------------------------------------------
    # Stage
    # --------------------------------------------------------

    def create_stage(self):

        self.objects.clear()

        self.objects["RED"] = WorldObject(

            object_id="RED",

            kind="REVERSAL",

            x=-80,

            y=120,

            vx=2.0,

            color="#ff6262"

        )

        self.objects["BLUE"] = WorldObject(

            object_id="BLUE",

            kind="MOVING_WALL",

            x=90,

            y=20,

            vx=-1.5,

            color="#5da9ff"

        )

        self.objects["PURPLE"] = WorldObject(

            object_id="PURPLE",

            kind="STATE",

            x=-120,

            y=190,

            vy=-1.0,

            color="#b16cff"

        )

        self.objects["GREEN"] = WorldObject(

            object_id="GREEN",

            kind="PULSE",

            x=100,

            y=-150,

            vx=1.0,

            color="#59d69a"

        )

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update(self):

        self.time += 1

        self.player.move()

        for obj in self.objects.values():

            self.update_object(
                obj
            )

    # --------------------------------------------------------
    # Object laws
    # --------------------------------------------------------

    def update_object(
        self,
        obj
    ):

        obj.timer += 1

        # ================================================
        # REVERSAL
        # ================================================

        if obj.kind == "REVERSAL":

            obj.x += obj.vx

            if (
                obj.timer % 100 == 0
            ):

                obj.vx *= -1

        # ================================================
        # MOVING WALL
        # ================================================

        elif obj.kind == "MOVING_WALL":

            obj.x += obj.vx

            obj.y = (
                20
                + math.sin(
                    obj.timer * 0.045
                ) * 100
            )

            if (
                obj.x < -20
                or obj.x > 220
            ):

                obj.vx *= -1

        # ================================================
        # STATE
        # ================================================

        elif obj.kind == "STATE":

            obj.y += obj.vy

            if (
                obj.y < -180
                or obj.y > 220
            ):

                obj.vy *= -1

            if (
                obj.timer % 120 == 0
            ):

                obj.state = (
                    obj.state + 1
                ) % 3

        # ================================================
        # PULSE
        # ================================================

        elif obj.kind == "PULSE":

            obj.x += obj.vx

            obj.vx += (

                math.sin(
                    obj.timer * 0.10
                ) * 0.03

            )

        # ================================================
        # Boundary
        # ================================================

        obj.x = clamp(
            obj.x,
            WORLD_LEFT + obj.radius,
            WORLD_RIGHT - obj.radius
        )

        obj.y = clamp(
            obj.y,
            WORLD_BOTTOM + obj.radius,
            WORLD_TOP - obj.radius
        )


# ============================================================
# GLOBAL WORLD / MODEL
# ============================================================

world = World()

model = WorldModel()

previous_state = None


# ============================================================
# INPUT STATE
# ============================================================

manual_action = "WAIT"


# ============================================================
# PLAYER CONTROL
# ============================================================

def move_left():

    player_action("LEFT")


def move_right():

    player_action("RIGHT")


def move_up():

    player_action("UP")


def move_down():

    player_action("DOWN")


def wait_action():

    player_action("WAIT")


def player_action(
    action
):

    global manual_action

    manual_action = action

    world.player.last_action = action

    if action == "LEFT":

        world.player.vx -= PLAYER_SPEED

    elif action == "RIGHT":

        world.player.vx += PLAYER_SPEED

    elif action == "UP":

        world.player.vy += PLAYER_SPEED

    elif action == "DOWN":

        world.player.vy -= PLAYER_SPEED

    elif action == "WAIT":

        pass


# ============================================================
# RESET
# ============================================================

def reset_world():

    global world
    global model
    global previous_state
    global manual_action

    world = World()

    model = WorldModel()

    previous_state = None

    manual_action = "WAIT"


# ============================================================
# DRAW HELPERS
# ============================================================

def draw_circle(
    pen,
    x,
    y,
    radius,
    color
):

    pen.goto(
        x + radius,
        y
    )

    pen.setheading(90)

    pen.pencolor(
        color
    )

    pen.pensize(2)

    pen.pendown()

    pen.circle(
        radius
    )

    pen.penup()


def draw_arrow(
    pen,
    x1,
    y1,
    x2,
    y2,
    color
):

    dx = x2 - x1

    dy = y2 - y1

    length = math.sqrt(
        dx * dx + dy * dy
    )

    if length < 0.1:

        return

    angle = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    pen.goto(
        x1,
        y1
    )

    pen.setheading(
        angle
    )

    pen.pencolor(
        color
    )

    pen.pensize(2)

    pen.pendown()

    pen.forward(
        length
    )

    pen.left(150)

    pen.forward(7)

    pen.backward(7)

    pen.right(300)

    pen.forward(7)

    pen.penup()


def write(
    pen,
    x,
    y,
    value,
    color="#ffffff",
    size=9,
    bold=False
):

    pen.goto(
        x,
        y
    )

    pen.color(
        color
    )

    pen.write(

        value,

        font=(

            "Arial",

            size,

            "bold"
            if bold
            else "normal"

        )

    )


# ============================================================
# DRAW WORLD
# ============================================================

def draw_world():

    world_pen.clear()

    # --------------------------------------------------------
    # Border
    # --------------------------------------------------------

    world_pen.goto(
        WORLD_LEFT,
        WORLD_BOTTOM
    )

    world_pen.setheading(0)

    world_pen.pencolor(
        "#33404d"
    )

    world_pen.pensize(2)

    world_pen.pendown()

    for _ in range(4):

        world_pen.forward(

            WORLD_RIGHT
            - WORLD_LEFT
            if _
            % 2 == 0
            else
            WORLD_TOP
            - WORLD_BOTTOM

        )

        world_pen.left(90)

    world_pen.penup()

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    world_pen.pencolor(
        "#17212b"
    )

    world_pen.pensize(1)

    x = WORLD_LEFT

    while x <= WORLD_RIGHT:

        world_pen.goto(
            x,
            WORLD_BOTTOM
        )

        world_pen.pendown()

        world_pen.goto(
            x,
            WORLD_TOP
        )

        world_pen.penup()

        x += 40

    y = WORLD_BOTTOM

    while y <= WORLD_TOP:

        world_pen.goto(
            WORLD_LEFT,
            y
        )

        world_pen.pendown()

        world_pen.goto(
            WORLD_RIGHT,
            y
        )

        world_pen.penup()

        y += 40

    # --------------------------------------------------------
    # Observation radius
    # --------------------------------------------------------

    world_pen.goto(
        world.player.x,
        world.player.y
    )

    world_pen.pencolor(
        "#263642"
    )

    world_pen.pensize(1)

    world_pen.pendown()

    world_pen.circle(
        OBSERVATION_DISTANCE
    )

    world_pen.penup()

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    for obj in world.objects.values():

        draw_circle(

            world_pen,

            obj.x,

            obj.y,

            obj.radius,

            obj.color

        )

        draw_arrow(

            world_pen,

            obj.x,

            obj.y,

            obj.x + obj.vx * 12,

            obj.y + obj.vy * 12,

            obj.color

        )

        write(

            world_pen,

            obj.x + 22,

            obj.y + 8,

            obj.object_id,

            obj.color,

            7,

            True

        )

        if obj.state != 0:

            write(

                world_pen,

                obj.x + 22,

                obj.y - 8,

                f"S{obj.state}",

                "#aab6c3",

                7

            )

    # --------------------------------------------------------
    # Player
    # --------------------------------------------------------

    world_pen.goto(
        world.player.x,
        world.player.y
    )

    world_pen.dot(
        22,
        "#ffffff"
    )

    draw_arrow(

        world_pen,

        world.player.x,

        world.player.y,

        world.player.x
        + world.player.vx * 8,

        world.player.y
        + world.player.vy * 8,

        "#ffffff"

    )


# ============================================================
# DRAW TRANSFORMATION
# ============================================================

def draw_transformation():

    law_pen.clear()

    recent = model.graph.recent(
        5
    )

    if not recent:

        return

    y = WORLD_TOP - 60

    write(

        law_pen,

        WORLD_LEFT + 10,

        y,

        "ΔWORLD / RECENT TRANSFORMATIONS",

        "#e5c76b",

        10,

        True

    )

    y -= 18

    for t in reversed(recent):

        if y < WORLD_BOTTOM + 30:

            break

        changes = []

        for oid in (
            t.delta.changed_objects.keys()
        ):

            changes.append(
                oid
            )

        relation_names = [

            r[1]

            for r
            in t.delta.relation_changes

        ]

        label = (

            f"t={t.time} "

            f"A={t.action} "

            f"OBJ={','.join(changes)} "

            f"{','.join(relation_names)}"

        )

        write(

            law_pen,

            WORLD_LEFT + 10,

            y,

            label[:72],

            "#aab6c3",

            7

        )

        y -= 14


# ============================================================
# DRAW LAWS
# ============================================================

def draw_laws():

    ui_pen.clear()

    x = PANEL_LEFT

    y = WORLD_TOP + 30

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    write(

        ui_pen,

        x,

        y,

        "WORLD MODEL",

        "#ffffff",

        15,

        True

    )

    y -= 25

    write(

        ui_pen,

        x,

        y,

        f"TIME : {world.time}",

        "#aab6c3",

        8

    )

    y -= 18

    write(

        ui_pen,

        x,

        y,

        f"TRANSFORMATIONS : "
        f"{len(model.memory)}",

        "#aab6c3",

        8

    )

    y -= 18

    write(

        ui_pen,

        x,

        y,

        f"GRAPH NODES : "
        f"{len(model.graph.nodes)}",

        "#aab6c3",

        8

    )

    y -= 30

    # --------------------------------------------------------
    # Laws
    # --------------------------------------------------------

    write(

        ui_pen,

        x,

        y,

        "DISCOVERED LAWS",

        "#69d391",

        12,

        True

    )

    y -= 22

    if not model.laws:

        write(

            ui_pen,

            x,

            y,

            "まだ法則は発見されていない",

            "#687583",

            8

        )

        y -= 25

    else:

        for key, law in model.laws.items():

            if y < WORLD_BOTTOM + 120:

                break

            write(

                ui_pen,

                x,

                y,

                key,

                "#69d391",

                8,

                True

            )

            y -= 14

            write(

                ui_pen,

                x,

                y,

                law.description,

                "#aab6c3",

                7

            )

            y -= 14

            write(

                ui_pen,

                x,

                y,

                (

                    f"evidence={law.evidence} "
                    f"confidence="
                    f"{int(law.confidence * 100)}%"

                ),

                "#718096",

                7

            )

            y -= 22

    # --------------------------------------------------------
    # Current action
    # --------------------------------------------------------

    y -= 10

    write(

        ui_pen,

        x,

        y,

        "CURRENT ACTION",

        "#70b7ff",

        10,

        True

    )

    y -= 18

    write(

        ui_pen,

        x,

        y,

        manual_action,

        "#ffffff",

        10,

        True

    )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    write(

        ui_pen,

        x,

        WORLD_BOTTOM + 80,

        "WASD / ARROWS : ACTION",

        "#718096",

        8

    )

    write(

        ui_pen,

        x,

        WORLD_BOTTOM + 62,

        "SPACE : WAIT / OBSERVE",

        "#718096",

        8

    )

    write(

        ui_pen,

        x,

        WORLD_BOTTOM + 44,

        "R : RESET WORLD MODEL",

        "#718096",

        8

    )

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    write(

        ui_pen,

        x,

        WORLD_BOTTOM + 5,

        "World",

        "#ffffff",

        8,

        True

    )

    write(

        ui_pen,

        x + 45,

        WORLD_BOTTOM + 5,

        "→ ΔWorld",

        "#e5c76b",

        8,

        True

    )

    write(

        ui_pen,

        x + 120,

        WORLD_BOTTOM + 5,

        "→ Transformation",

        "#c38cff",

        8,

        True

    )

    write(

        ui_pen,

        x + 240,

        WORLD_BOTTOM + 5,

        "→ Law",

        "#69d391",

        8,

        True

    )


# ============================================================
# OBSERVATION
# ============================================================

def observe():

    global previous_state

    state = model.capture_world(
        world
    )

    transformation = (
        model.observe_transition(

            previous_state,

            state,

            manual_action

        )
    )

    previous_state = state

    return transformation


# ============================================================
# MAIN LOOP
# ============================================================

running = True


def game_loop():

    if not running:

        return

    # --------------------------------------------------------
    # Capture BEFORE
    # --------------------------------------------------------

    before = model.capture_world(
        world
    )

    # --------------------------------------------------------
    # World update
    # --------------------------------------------------------

    world.update()

    # --------------------------------------------------------
    # Capture AFTER
    # --------------------------------------------------------

    after = model.capture_world(
        world
    )

    # --------------------------------------------------------
    # Learn ΔWorld
    # --------------------------------------------------------

    transformation = (
        model.observe_transition(

            before,

            after,

            manual_action

        )
    )

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    draw_world()

    draw_laws()

    draw_transformation()

    screen.update()

    # --------------------------------------------------------
    # Next frame
    # --------------------------------------------------------

    screen.ontimer(
        game_loop,
        FRAME_MS
    )


# ============================================================
# KEYBOARD
# ============================================================

screen.listen()

screen.onkeypress(
    move_left,
    "Left"
)

screen.onkeypress(
    move_right,
    "Right"
)

screen.onkeypress(
    move_up,
    "Up"
)

screen.onkeypress(
    move_down,
    "Down"
)

screen.onkeypress(
    move_left,
    "a"
)

screen.onkeypress(
    move_right,
    "d"
)

screen.onkeypress(
    move_up,
    "w"
)

screen.onkeypress(
    move_down,
    "s"
)

screen.onkeypress(
    wait_action,
    "space"
)

screen.onkeypress(
    reset_world,
    "r"
)


# ============================================================
# START
# ============================================================

draw_world()

draw_laws()

draw_transformation()

screen.update()

screen.ontimer(
    game_loop,
    FRAME_MS
)

turtle.done()

