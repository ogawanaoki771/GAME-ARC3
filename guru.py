# ============================================================
# TOKYO GHOUL : BOX OF CHAOS
# REWARD-FREE LIVING WORLD MODEL
#
# 「報酬されるから生きる」のではなく、
# 「理解できないから観測する」。
#
# 喰種 = 生存する予測器
# RC細胞 = 内部状態
# 赫眼 = 高覚醒状態
# 赫子 = 身体の境界から流れ出す生体組織
# 爪 = その先端が一時的に硬化した状態
# 形態 = 行動ではなく身体と世界の相互変換
# 箱 = 世界
# ぐちゃぐちゃ = 予測誤差
#
# 必要:
#   Python 3.x
#   numpy
#
# pip install numpy
# ============================================================
import math
import random
from collections import deque
import numpy as np
import turtle
# ============================================================
# CONFIG
# ============================================================
SCREEN_W = 1240
SCREEN_H = 820
WORLD_LEFT = -520
WORLD_RIGHT = 520
WORLD_BOTTOM = -285
WORLD_TOP = 275
NUM_GHOULS = 3
STEPS_PER_EPISODE = 900
MAX_EPISODES = 30
# 小さいセル
CELL_SIZE = 3
OBS_W = 80
OBS_H = 50
CHANNELS = 9
PATCH = 4
MEMORY_SIZE = 220
EXPERIENCE_SIZE = 3000
# ------------------------------------------------------------
# 物理
# ------------------------------------------------------------
GRAVITY = 0.0
GROUND_ACCEL = 0.72
AIR_ACCEL = 0.42
GROUND_FRICTION = 0.86
AIR_FRICTION = 0.985
JUMP_POWER = 9.0
MAX_SPEED = 7.5
# ------------------------------------------------------------
# 空間
# ------------------------------------------------------------
LEFT_ROOM_X1 = -495
LEFT_ROOM_X2 = -80
RIGHT_ROOM_X1 = 80
RIGHT_ROOM_X2 = 495
TUNNEL_X1 = -80
TUNNEL_X2 = 80
TUNNEL_Y1 = -55
TUNNEL_Y2 = 55
GATE_X = 0
GATE_Y = 0
GATE_DISTANCE = 42
GOAL_X = 420
GOAL_Y = 215
GOAL_RADIUS = 30
RESET_COOLDOWN = 35
# ------------------------------------------------------------
# RC
# ------------------------------------------------------------
RC_NODE_COUNT = 22
RC_BASE = 0.65
RC_MAX = 2.0
# ------------------------------------------------------------
# 喰種
# ------------------------------------------------------------
KAKUGAN_THRESHOLD = 0.72
RC_ABSORPTION = 0.012
RC_DECAY = 0.004
KAGUNE_MAX = 1.0
# ------------------------------------------------------------
# 赫子
# ------------------------------------------------------------
KAGUNE_DAMAGE = 0.35
KAGUNE_PUSH = 2.2
KAGUNE_RANGE = 80

# ------------------------------------------------------------
# 赫子筋 / LIQUID MUSCLE
# ------------------------------------------------------------
# 赫子を「棒」や「固定された4本の線」ではなく、
# 流動的な筋肉組織として扱う。
KAGUNE_SEGMENTS = 18
KAGUNE_BASE_LENGTH = 24
KAGUNE_MAX_LENGTH = 118
KAGUNE_MUSCLE_WIDTH = 7
KAGUNE_FLOW_SPEED = 0.18
KAGUNE_CONTRACTION = 0.72
KAGUNE_CLAW_HARDNESS = 0.78
KAGUNE_CLAW_LENGTH = 22
KAGUNE_DEFENSE_RADIUS = 30
KAGUNE_GRAB_RANGE = 92
KAGUNE_MUTATION_RATE = 0.16
KAGUNE_BRANCHES = 3
KAGUNE_BRANCH_LENGTH = 46
KAGUNE_RETURN_RATE = 0.035
KAGUNE_TISSUE_DRAG = 0.22
KAGUNE_TISSUE_MEMORY = 180
KAGUNE_MORPHOLOGY_NOVELTY = 0.85
KAGUNE_BODY_BLEND = 0.18
KAGUNE_CONTACT_ADAPT = 0.12
# ------------------------------------------------------------
# 捕食者
# ------------------------------------------------------------
MONSTER_SPEED = 4.6
MONSTER_DETECTION = 400
MONSTER_MEMORY = 100
# ============================================================
# ACTION
# ============================================================
ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3
ACTION_BRAKE = 4
ACTION_WAIT = 5
ACTION_KAGUNE = 6
ACTION_KAGUNE_CLAW = 7
ACTION_KAGUNE_FLOW = 8
ACTIONS = [
    ACTION_NONE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_JUMP,
    ACTION_BRAKE,
    ACTION_WAIT,
    ACTION_KAGUNE,
    ACTION_KAGUNE_CLAW,
    ACTION_KAGUNE_FLOW,
]
ACTION_NAMES = {
    ACTION_NONE: "NONE",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_JUMP: "JUMP",
    ACTION_BRAKE: "BRAKE",
    ACTION_WAIT: "WAIT",
    ACTION_KAGUNE: "KAGUNE",
    ACTION_KAGUNE_CLAW: "KAGUNE_CLAW",
    ACTION_KAGUNE_FLOW: "KAGUNE_FLOW",
}
# ============================================================
# UTILITY
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))
def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)
def l1(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(a[:n] - b[:n])))
def similarity(a, b):
    return math.exp(-4.0 * l1(a, b))
# ============================================================
# RC FIELD
# ============================================================
class RCNode:
    def __init__(self, node_id, x, y):
        self.id = node_id
        self.x = x
        self.y = y
        self.base_x = x
        self.base_y = y
        self.energy = random.uniform(0.35, 1.4)
        self.phase = random.uniform(0, math.tau)
        self.radius = random.uniform(60, 150)
    def update(self, t):
        wave = math.sin(
            self.phase + t * 0.035
        )
        self.energy += wave * 0.012
        self.energy += random.uniform(-0.018, 0.018)
        self.energy = clamp(
            self.energy,
            0.05,
            RC_MAX
        )
        self.x = (
            self.base_x
            + math.sin(
                self.phase + t * 0.009
            ) * 45
        )
        self.y = (
            self.base_y
            + math.cos(
                self.phase * 1.7
                + t * 0.012
            ) * 35
        )
class RCField:
    def __init__(self):
        self.nodes = []
        self.global_energy = RC_BASE
        self.reset()
    def reset(self):
        self.nodes.clear()
        for i in range(RC_NODE_COUNT):
            if i < RC_NODE_COUNT // 3:
                x = random.uniform(
                    LEFT_ROOM_X1 + 30,
                    LEFT_ROOM_X2 - 20
                )
            elif i < RC_NODE_COUNT * 2 // 3:
                x = random.uniform(
                    RIGHT_ROOM_X1 + 20,
                    RIGHT_ROOM_X2 - 30
                )
            else:
                x = random.uniform(
                    TUNNEL_X1 - 60,
                    TUNNEL_X2 + 60
                )
            y = random.uniform(
                WORLD_BOTTOM + 30,
                WORLD_TOP - 30
            )
            self.nodes.append(
                RCNode(i, x, y)
            )
    def update(self, t):
        for node in self.nodes:
            node.update(t)
        self.global_energy = float(
            np.mean(
                [n.energy for n in self.nodes]
            )
        )
    def energy_at(self, x, y):
        total = 0.0
        weight_sum = 0.0
        for node in self.nodes:
            d = distance(
                x,
                y,
                node.x,
                node.y
            )
            w = math.exp(
                -(d * d)
                / (2 * node.radius * node.radius)
            )
            total += node.energy * w
            weight_sum += w
        if weight_sum <= 0.001:
            return self.global_energy
        return clamp(
            total / weight_sum,
            0.0,
            RC_MAX
        )
    def gradient(self, x, y):
        s = 12
        gx = (
            self.energy_at(x + s, y)
            - self.energy_at(x - s, y)
        ) / (2 * s)
        gy = (
            self.energy_at(x, y + s)
            - self.energy_at(x, y - s)
        ) / (2 * s)
        return gx, gy
    def disturb(self, x, y, amount):
        if not self.nodes:
            return
        node = min(
            self.nodes,
            key=lambda n:
            distance(x, y, n.x, n.y)
        )
        node.energy = clamp(
            node.energy + amount,
            0.05,
            RC_MAX
        )
# ============================================================
# LIVING OBSTACLE
# ============================================================
class LivingObstacle:
    def __init__(
        self,
        x1,
        y1,
        x2,
        y2,
        movement=0,
        pulse=0.1
    ):
        self.base = (
            x1,
            y1,
            x2,
            y2
        )
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.movement = movement
        self.pulse = pulse
        self.phase = random.random() * math.tau
    def update(self, t, rc):
        x1, y1, x2, y2 = self.base
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        energy = rc.energy_at(
            cx,
            cy
        )
        scale = clamp(
            energy / RC_BASE,
            0.55,
            1.7
        )
        sx = (
            1
            + math.sin(
                self.phase + t * 0.026
            )
            * self.pulse
            * scale
        )
        sy = (
            1
            + math.cos(
                self.phase * 1.7
                + t * 0.021
            )
            * self.pulse
            * scale
        )
        dx = (
            math.sin(
                self.phase + t * 0.012
            )
            * self.movement
        )
        dy = (
            math.cos(
                self.phase + t * 0.009
            )
            * self.movement
            * 0.7
        )
        cx += dx
        cy += dy
        self.x1 = cx - w * sx / 2
        self.x2 = cx + w * sx / 2
        self.y1 = cy - h * sy / 2
        self.y2 = cy + h * sy / 2
    def contains(self, x, y, radius=0):
        return (
            self.x1 - radius < x < self.x2 + radius
            and
            self.y1 - radius < y < self.y2 + radius
        )
# ============================================================
# KAGUNE LIQUID MUSCLE
# ============================================================
class KaguneMuscle:
    """
    赫子を「武器」から切り離し、身体の外へ流れ出す生体組織として扱う。

    重要なのは「攻撃する」ではなく、
        内部状態 -> 組織の流動 -> 形態 -> 接触 -> 再吸収
    という連続変換。

    形態は固定されない。先端は爪にも鞭にも盾にもなり得るし、
    その境界も曖昧。人間が見れば少し気味が悪いくらいがちょうどいい。
    """
    def __init__(self, owner):
        self.owner = owner
        self.energy = 0.0
        self.extension = 0.08
        self.flow = 0.0
        self.phase = random.random() * math.tau
        self.mode = "absorbed"
        self.target_angle = 0.0
        self.tip_x = owner.x
        self.tip_y = owner.y
        self.contact = 0.0
        self.claw = 0.0
        self.elasticity = random.uniform(0.72, 1.18)
        self.density = 0.25
        self.temperature = 0.0
        self.boundary = 0.0
        self.morphology_age = 0
        self.morphology_change = 0.0
        self.reabsorption = 0.0
        self.branches = []
        self.history = deque(maxlen=KAGUNE_TISSUE_MEMORY)
        self.segments = [[owner.x, owner.y, 0.0, 0.0]
                         for _ in range(KAGUNE_SEGMENTS)]
        self._make_branches()

    def _make_branches(self):
        self.branches = []
        for b in range(KAGUNE_BRANCHES):
            self.branches.append([
                [self.owner.x, self.owner.y] for _ in range(5)
            ])

    def reset(self):
        self.energy = 0.0
        self.extension = 0.08
        self.flow = 0.0
        self.phase = random.random() * math.tau
        self.mode = "absorbed"
        self.contact = 0.0
        self.claw = 0.0
        self.density = 0.25
        self.boundary = 0.0
        self.morphology_age = 0
        self.morphology_change = 0.0
        self.reabsorption = 0.0
        self.history.clear()
        self._collapse_to_body()

    def _collapse_to_body(self):
        for seg in self.segments:
            seg[0] = self.owner.x
            seg[1] = self.owner.y
            seg[2] = self.owner.x
            seg[3] = self.owner.y
        for branch in self.branches:
            for point in branch:
                point[0] = self.owner.x
                point[1] = self.owner.y
        self.tip_x = self.owner.x
        self.tip_y = self.owner.y

    def stimulate(self, amount):
        self.energy = clamp(self.energy + amount, 0.0, 1.0)

    def signature(self):
        """形態そのものを世界モデルが記憶できる低次元表現。"""
        if not self.segments:
            return np.zeros(12, dtype=np.float32)
        lengths = []
        bends = []
        for i, seg in enumerate(self.segments):
            if i == 0:
                px, py = self.owner.x, self.owner.y
            else:
                px, py = self.segments[i-1][0], self.segments[i-1][1]
            lengths.append(math.hypot(seg[0]-px, seg[1]-py))
            bends.append(math.atan2(seg[1]-py, seg[0]-px) / math.pi)
        sample = self.segments[::max(1, len(self.segments)//6)]
        vals = [
            self.energy, self.extension, self.flow, self.claw,
            self.density, self.boundary, self.contact,
            self.morphology_change, self.reabsorption,
            float(np.mean(lengths)), float(np.std(lengths)),
            float(np.mean(bends)),
        ]
        return np.asarray(vals, dtype=np.float32)

    def choose_shape(self, action, threat, contact_pressure):
        # 行動は「武器選択」ではなく、組織への刺激。
        if action == ACTION_KAGUNE_CLAW:
            self.mode = "hardening"
        elif action == ACTION_KAGUNE_FLOW:
            self.mode = "flow"
        elif threat > 0.78:
            self.mode = "shield"
        elif contact_pressure > 0.55:
            self.mode = "adaptive"
        elif self.energy > 0.52:
            self.mode = "fluid"
        elif self.energy > 0.16:
            self.mode = "thread"
        else:
            self.mode = "absorbing"

    def update(self, action, threat):
        owner = self.owner
        old_sig = self.signature().copy()
        self.phase += KAGUNE_FLOW_SPEED + self.energy * 0.14
        self.flow = 0.5 + 0.5 * math.sin(self.phase)
        self.morphology_age += 1

        if action in (ACTION_KAGUNE, ACTION_KAGUNE_CLAW, ACTION_KAGUNE_FLOW):
            self.stimulate(0.13 + owner.rc * 0.025)
        else:
            # 筋肉は常に少しずつ身体へ戻る。出したままの剣ではない。
            self.energy *= (1.0 - KAGUNE_RETURN_RATE)

        # 接触経験によって可塑性が変わる。
        contact_pressure = self.contact
        self.choose_shape(action, threat, contact_pressure)

        if self.mode == "absorbing":
            desired_length = KAGUNE_BASE_LENGTH * 0.18
            self.reabsorption = min(1.0, self.reabsorption + 0.06)
        elif self.mode == "thread":
            desired_length = KAGUNE_BASE_LENGTH * (0.45 + self.energy * 0.45)
            self.reabsorption *= 0.96
        elif self.mode == "hardening":
            desired_length = KAGUNE_MAX_LENGTH * (0.50 + 0.40 * self.energy)
            self.reabsorption *= 0.92
        elif self.mode == "shield":
            desired_length = KAGUNE_MAX_LENGTH * (0.45 + 0.30 * self.energy)
            self.reabsorption *= 0.95
        else:
            desired_length = KAGUNE_BASE_LENGTH + self.energy * (KAGUNE_MAX_LENGTH - KAGUNE_BASE_LENGTH)
            self.reabsorption *= 0.97

        target_extension = clamp(desired_length / KAGUNE_MAX_LENGTH, 0.02, 1.0)
        self.extension += (target_extension - self.extension) * (KAGUNE_CONTRACTION * 0.45)
        self.density += ((0.18 + self.energy * 0.65 + self.claw * 0.28) - self.density) * 0.10
        self.boundary += ((self.energy * 0.9) - self.boundary) * KAGUNE_BODY_BLEND

        # 身体から外へ「流出」する主幹。
        prev_x, prev_y = owner.x, owner.y
        for i, seg in enumerate(self.segments):
            q = (i + 1) / KAGUNE_SEGMENTS
            wave = math.sin(self.phase + i * 0.69 + owner.vx * 0.10)
            wave2 = math.cos(self.phase * 0.73 + i * 0.41)

            if self.mode == "shield":
                angle = owner.heading * 0.25 + q * math.tau * 0.86 + wave * 0.18
                radius = desired_length * (0.28 + 0.22 * q)
                tx = owner.x + math.cos(angle) * radius
                ty = owner.y + math.sin(angle) * radius
            else:
                # 流動筋肉は一方向へまっすぐ伸びず、遅れを持つ。
                bend = wave * (0.16 + q * 0.34) + wave2 * 0.11
                angle = owner.heading * 0.10 + bend + self.flow * 0.20
                local_length = desired_length / KAGUNE_SEGMENTS
                tx = prev_x + math.cos(angle) * local_length
                ty = prev_y + math.sin(angle) * local_length

            alpha = clamp(KAGUNE_TISSUE_DRAG + self.energy * 0.24, 0.16, 0.52)
            seg[0] += (tx - seg[0]) * alpha
            seg[1] += (ty - seg[1]) * alpha
            seg[2], seg[3] = prev_x, prev_y
            prev_x, prev_y = seg[0], seg[1]

        self.tip_x, self.tip_y = self.segments[-1][0], self.segments[-1][1]

        # 先端硬化。爪は「物」ではなく状態。
        target_claw = KAGUNE_CLAW_HARDNESS if self.mode == "hardening" else 0.0
        self.claw += (target_claw - self.claw) * 0.18

        # 主幹から三方向へ流れる小さな枝。固定武器ではなく一時的な筋束。
        for b, branch in enumerate(self.branches):
            start_index = 7 + b * 3
            anchor = self.segments[min(start_index, len(self.segments)-1)]
            px, py = anchor[0], anchor[1]
            spread = (-0.42 + b * 0.42) + math.sin(self.phase * 0.7 + b) * 0.10
            for j, point in enumerate(branch):
                if self.energy < 0.28:
                    tx, ty = owner.x, owner.y
                else:
                    q = (j + 1) / len(branch)
                    angle = owner.heading * 0.10 + spread + math.sin(self.phase + j) * 0.08
                    length = KAGUNE_BRANCH_LENGTH * self.energy * q * (0.55 if self.mode == "shield" else 1.0)
                    tx = px + math.cos(angle) * length / len(branch)
                    ty = py + math.sin(angle) * length / len(branch)
                point[0] += (tx - point[0]) * 0.24
                point[1] += (ty - point[1]) * 0.24
                px, py = point[0], point[1]

        new_sig = self.signature()
        self.morphology_change = float(np.mean(np.abs(new_sig - old_sig)))
        self.history.append(new_sig.copy())
        self.contact *= 0.92

    def attack(self, agents):
        if self.energy < 0.10:
            return
        owner = self.owner
        targets = []
        # 接触点は主幹だけでなく枝にも存在する。
        points = [(self.tip_x, self.tip_y)]
        for branch in self.branches:
            if branch:
                points.append(tuple(branch[-1]))
        for target in agents:
            if target.id == owner.id or target.dead:
                continue
            nearest = min(points, key=lambda p: distance(p[0], p[1], target.x, target.y))
            d = distance(nearest[0], nearest[1], target.x, target.y)
            if d > KAGUNE_GRAB_RANGE:
                continue
            nx = (target.x - nearest[0]) / max(0.01, d)
            ny = (target.y - nearest[1]) / max(0.01, d)
            pressure = self.energy * (0.55 + self.density * 0.7 + self.claw * 0.9)
            target.vx += nx * KAGUNE_PUSH * pressure
            target.vy += ny * KAGUNE_PUSH * pressure
            target.risk = clamp(target.risk + KAGUNE_DAMAGE * pressure * 0.42, 0, 1)
            self.contact = min(1.0, self.contact + KAGUNE_CONTACT_ADAPT)
            self.energy = clamp(self.energy - 0.018, 0, 1)

    def collide_with_monster(self, monster):
        if self.energy < 0.15:
            return
        points = [(self.tip_x, self.tip_y)] + [tuple(b[-1]) for b in self.branches if b]
        nearest = min(points, key=lambda p: distance(p[0], p[1], monster.x, monster.y))
        d = distance(nearest[0], nearest[1], monster.x, monster.y)
        if d < KAGUNE_GRAB_RANGE:
            nx = (monster.x - nearest[0]) / max(0.01, d)
            ny = (monster.y - nearest[1]) / max(0.01, d)
            force = KAGUNE_PUSH * self.energy * (0.45 + self.claw + self.density * 0.45)
            monster.vx += nx * force
            monster.vy += ny * force
            self.contact = min(1.0, self.contact + 0.15)

    def defense(self, x, y):
        if self.mode != "shield":
            return 0.0
        d = distance(self.owner.x, self.owner.y, x, y)
        if d >= KAGUNE_DEFENSE_RADIUS:
            return 0.0
        return clamp((1.0 - d / KAGUNE_DEFENSE_RADIUS) * self.energy * (0.45 + self.density * 0.35), 0, 1)

# ============================================================
# CHIMIMORYO
# ============================================================
class Chimimoryo:
    def __init__(self, entity_id, world):
        self.id = entity_id
        self.world = world
        self.x = random.uniform(
            WORLD_LEFT + 50,
            WORLD_RIGHT - 50
        )
        self.y = random.uniform(
            WORLD_BOTTOM + 50,
            WORLD_TOP - 50
        )
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.energy = random.uniform(
            0.3,
            1.2
        )
        self.radius = random.uniform(
            7,
            17
        )
        self.phase = random.random() * math.tau
        self.mode = random.choice(
            [
                "drifter",
                "hunter",
                "swarm",
                "flee",
            ]
        )
        self.hits = 0
    def reset(self):
        self.x = random.uniform(
            WORLD_LEFT + 50,
            WORLD_RIGHT - 50
        )
        self.y = random.uniform(
            WORLD_BOTTOM + 50,
            WORLD_TOP - 50
        )
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.energy = random.uniform(
            0.3,
            1.2
        )
        self.hits = 0
    def update(self, agents):
        rc = self.world.rc
        local = rc.energy_at(
            self.x,
            self.y
        )
        gx, gy = rc.gradient(
            self.x,
            self.y
        )
        self.energy += (
            local - self.energy
        ) * 0.02
        self.vx += gx * 14
        self.vy += gy * 14
        target = min(
            [
                a for a in agents
                if not a.dead
            ],
            key=lambda a:
            distance(
                self.x,
                self.y,
                a.x,
                a.y
            ),
            default=None
        )
        if target:
            dx = target.x - self.x
            dy = target.y - self.y
            d = max(
                0.01,
                math.hypot(dx, dy)
            )
            if self.mode == "hunter" and d < 260:
                self.vx += (
                    dx / d
                    * 0.16
                )
                self.vy += (
                    dy / d
                    * 0.16
                )
            elif self.mode == "flee" and d < 180:
                self.vx -= (
                    dx / d
                    * 0.14
                )
                self.vy -= (
                    dy / d
                    * 0.14
                )
            elif self.mode == "swarm":
                self.vx += (
                    dx / d
                    * 0.025
                )
                self.vy += (
                    dy / d
                    * 0.025
                )
        self.vx += math.sin(
            self.phase + self.world.time * 0.05
        ) * 0.05
        self.vy += math.cos(
            self.phase + self.world.time * 0.047
        ) * 0.05
        speed = math.hypot(
            self.vx,
            self.vy
        )
        max_speed = 2.2 + self.energy * 1.4
        if speed > max_speed:
            self.vx = (
                self.vx / speed
                * max_speed
            )
            self.vy = (
                self.vy / speed
                * max_speed
            )
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.97
        self.vy *= 0.97
        self.x = clamp(
            self.x,
            WORLD_LEFT + 20,
            WORLD_RIGHT - 20
        )
        self.y = clamp(
            self.y,
            WORLD_BOTTOM + 20,
            WORLD_TOP - 20
        )
        for agent in agents:
            self.hit(agent)
    def hit(self, agent):
        d = distance(
            self.x,
            self.y,
            agent.x,
            agent.y
        )
        if d < self.radius + 10:
            if d < 0.01:
                return
            nx = (
                agent.x - self.x
            ) / d
            ny = (
                agent.y - self.y
            ) / d
            agent.vx += nx * 1.4
            agent.vy += ny * 1.4
            agent.risk = clamp(
                agent.risk + 0.04,
                0,
                1
            )
            self.hits += 1
            self.world.rc.disturb(
                self.x,
                self.y,
                0.05
            )
# ============================================================
# SHARED MEMORY
# ============================================================
class Blackboard:
    def __init__(self):
        self.data = {}
    def update(self, agent):
        self.data[agent.id] = {
            "x": agent.x,
            "y": agent.y,
            "vx": agent.vx,
            "vy": agent.vy,
            "risk": agent.risk,
            "rc": agent.rc,
            "kakugan": agent.kakugan,
            "action": agent.last_action,
            "kagune_mode": agent.kagune_muscle.mode,
            "kagune_energy": agent.kagune_muscle.energy,
            "kagune_claw": agent.kagune_muscle.claw,
            "kagune_change": agent.kagune_muscle.morphology_change,
        }
    def snapshot(self):
        return dict(self.data)
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
        viewer = agents[viewer_id]
        def dot(channel, x, y, value):
            # 世界座標 -> 小さいセル
            px = int(
                (x - WORLD_LEFT)
                / (WORLD_RIGHT - WORLD_LEFT)
                * OBS_W
            )
            py = int(
                (y - WORLD_BOTTOM)
                / (WORLD_TOP - WORLD_BOTTOM)
                * OBS_H
            )
            if (
                0 <= px < OBS_W
                and 0 <= py < OBS_H
            ):
                img[
                    OBS_H - 1 - py,
                    px,
                    channel
                ] = max(
                    img[
                        OBS_H - 1 - py,
                        px,
                        channel
                    ],
                    value
                )
        # 0 = 自分
        dot(
            0,
            viewer.x,
            viewer.y,
            1.0
        )
        # 1 = 他の喰種
        for agent in agents:
            if agent.id != viewer_id:
                dot(
                    1,
                    agent.x,
                    agent.y,
                    0.8
                )
        # 2 = 壁
        for obstacle in self.world.obstacles:
            cx = (
                obstacle.x1
                + obstacle.x2
            ) / 2
            cy = (
                obstacle.y1
                + obstacle.y2
            ) / 2
            dot(
                2,
                cx,
                cy,
                0.8
            )
        # 3 = RC
        for node in self.world.rc.nodes:
            dot(
                3,
                node.x,
                node.y,
                clamp(
                    node.energy / RC_MAX,
                    0,
                    1
                )
            )
        # 4 = 捕食者
        dot(
            4,
            self.monster.x,
            self.monster.y,
            1.0
        )
        # 5 = chimimoryo
        for entity in self.world.chimimoryo:
            dot(
                5,
                entity.x,
                entity.y,
                clamp(
                    entity.energy / RC_MAX,
                    0,
                    1
                )
            )
        # 6 = 自分の流動筋肉
        muscle = viewer.kagune_muscle
        for seg in muscle.segments:
            dot(6, seg[0], seg[1], clamp(muscle.energy, 0, 1))
        # 7 = 赫子の硬化 / 爪
        dot(7, muscle.tip_x, muscle.tip_y, clamp(muscle.claw, 0, 1))
        # 8 = 形態変化量
        dot(8, viewer.x, viewer.y, clamp(muscle.morphology_change * 8.0, 0, 1))
        return img.reshape(-1)
# ============================================================
# EXPERIENCE
# ============================================================
class Experience:
    def __init__(
        self,
        before,
        action,
        after,
        error
    ):
        self.before = before
        self.action = action
        self.after = after
        self.error = error
# ============================================================
# WORLD MODEL
# ============================================================
class WorldModel:
    def __init__(self):
        self.transitions = {
            a: []
            for a in ACTIONS
        }
        self.experiences = deque(
            maxlen=EXPERIENCE_SIZE
        )
        self.prediction_error = 0.5
        self.curiosity = 0.5
        self.stability = 0.5
        self.morphology_memory = deque(maxlen=KAGUNE_TISSUE_MEMORY)
        self.morphology_novelty = 0.5
        self.total_steps = 0
    def predict_error(
        self,
        before,
        after,
        action
    ):
        memories = self.transitions[action]
        if not memories:
            return 1.0
        best = min(
            memories,
            key=lambda x:
            l1(x, before)
        )
        prediction = best
        return clamp(
            l1(prediction, after),
            0,
            1
        )
    def learn(
        self,
        before,
        action,
        after
    ):
        error = self.predict_error(
            before,
            after,
            action
        )
        # 身体変形そのものを経験として保存する。
        # 「攻撃が成功したか」ではなく「身体がどう変わったか」を記憶する。
        # 呼び出し側の agent は action の後に signature を登録する。
        memories = self.transitions[action]
        if len(memories) < 20:
            memories.append(
                after.copy()
            )
        else:
            idx = random.randrange(
                len(memories)
            )
            memories[idx] = (
                memories[idx] * 0.92
                + after * 0.08
            )
        self.experiences.append(
            Experience(
                before,
                action,
                after,
                error
            )
        )
        self.prediction_error = (
            self.prediction_error * 0.96
            + error * 0.04
        )
        self.curiosity = (
            0.65 * self.prediction_error
            + 0.35
            * min(
                1.0,
                len(
                    memories
                ) / 50
            )
        )
        self.stability = clamp(
            1.0
            - self.prediction_error,
            0,
            1
        )
        self.total_steps += 1
        return error
    def replay(self, amount=150):
        if not self.experiences:
            return
        for _ in range(
            min(
                amount,
                len(self.experiences)
            )
        ):
            exp = random.choice(
                list(self.experiences)
            )
            self.learn(
                exp.before,
                exp.action,
                exp.after
            )
    def select_action(
        self,
        observation,
        agent
    ):
        scores = {}
        for action in ACTIONS:
            activity = random.uniform(
                -0.05,
                0.05
            )
            novelty = (
                self.curiosity
                * random.random()
            )
            # 思想:
            # 不確実性が高いほど、
            # 「見ていない方向」を選びやすい。
            if action == ACTION_WAIT:
                novelty *= 0.35
            if action in (
                ACTION_KAGUNE,
                ACTION_KAGUNE_CLAW,
                ACTION_KAGUNE_FLOW
            ):
                novelty *= (
                    1.0
                    + agent.risk
                    + agent.rc * 0.15
                )
            if action == ACTION_KAGUNE_CLAW:
                novelty *= 1.0 + agent.kagune_muscle.claw * 0.8
            if action == ACTION_KAGUNE_FLOW:
                novelty *= 1.0 + agent.kagune_muscle.flow * 0.3
            if action in (ACTION_KAGUNE, ACTION_KAGUNE_CLAW, ACTION_KAGUNE_FLOW):
                # 報酬ではなく「まだ見ていない身体変形」を選びやすくする。
                morph = agent.kagune_muscle.signature()
                if self.morphology_memory:
                    nearest = min(float(np.mean(np.abs(m - morph))) for m in self.morphology_memory)
                    morph_novelty = clamp(nearest * 3.5, 0, 1)
                else:
                    morph_novelty = 1.0
                if action == ACTION_KAGUNE_CLAW:
                    morph_novelty *= 1.15
                if action == ACTION_KAGUNE_FLOW:
                    morph_novelty *= 1.25
                novelty += KAGUNE_MORPHOLOGY_NOVELTY * morph_novelty
            scores[action] = novelty + activity
        # 共有状態
        others = [
            x for x in agent.blackboard.data.values()
            if x is not None
        ]
        if others:
            avg_risk = np.mean(
                [
                    x["risk"]
                    for x in others
                ]
            )
            if avg_risk > 0.6:
                scores[ACTION_KAGUNE] += 0.35
                scores[ACTION_BRAKE] += 0.15
        # ゴールそのものを報酬にはしない。
        # ただし環境状態として知覚する。
        if agent.x < 0:
            scores[ACTION_RIGHT] += 0.12
        else:
            scores[ACTION_LEFT] += 0.04
        return max(
            ACTIONS,
            key=lambda a:
            scores[a]
        )
    def statistics(self):
        return {
            "experience":
                len(self.experiences),
            "error":
                self.prediction_error,
            "curiosity":
                self.curiosity,
            "stability":
                self.stability,
        }
# ============================================================
# PREDICTIVE MONSTER
# ============================================================
class PredictiveMonster:
    def __init__(self, world):
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
                NUM_GHOULS
            )
        }
        self.switches = 0
    def reset(self):
        self.x = 300
        self.y = 20
        self.vx = 0
        self.vy = 0
        self.target_id = None
        for m in self.memory.values():
            m.clear()
    def update(self, agents):
        for agent in agents:
            self.memory[
                agent.id
            ].append(
                (
                    agent.x,
                    agent.y,
                    agent.vx,
                    agent.vy
                )
            )
        candidates = []
        for agent in agents:
            if agent.dead:
                continue
            d = distance(
                self.x,
                self.y,
                agent.x,
                agent.y
            )
            if d > MONSTER_DETECTION:
                continue
            mem = list(
                self.memory[agent.id]
            )
            if len(mem) >= 3:
                vx = np.mean(
                    [m[2] for m in mem[-5:]]
                )
                vy = np.mean(
                    [m[3] for m in mem[-5:]]
                )
            else:
                vx = agent.vx
                vy = agent.vy
            horizon = clamp(
                d / 25,
                3,
                18
            )
            px = (
                agent.x
                + vx * horizon
            )
            py = (
                agent.y
                + vy * horizon
            )
            score = (
                -distance(
                    self.x,
                    self.y,
                    px,
                    py
                )
                + agent.risk * 100
                + agent.rc * 15
            )
            candidates.append(
                (
                    score,
                    agent,
                    px,
                    py
                )
            )
        if not candidates:
            return
        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )
        _, target, px, py = candidates[0]
        if (
            self.target_id is not None
            and self.target_id != target.id
        ):
            self.switches += 1
        self.target_id = target.id
        dx = px - self.x
        dy = py - self.y
        d = max(
            0.01,
            math.hypot(dx, dy)
        )
        self.vx += (
            dx / d * 0.30
        )
        self.vy += (
            dy / d * 0.30
        )
        speed = math.hypot(
            self.vx,
            self.vy
        )
        if speed > MONSTER_SPEED:
            self.vx = (
                self.vx / speed
                * MONSTER_SPEED
            )
            self.vy = (
                self.vy / speed
                * MONSTER_SPEED
            )
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.94
        self.vy *= 0.94
        self.x = clamp(
            self.x,
            WORLD_LEFT + 25,
            WORLD_RIGHT - 25
        )
        self.y = clamp(
            self.y,
            WORLD_BOTTOM + 25,
            WORLD_TOP - 25
        )
# ============================================================
# GHOUL
# ============================================================
class Ghoul:
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
        self.x = (
            LEFT_ROOM_X1
            + 55
            + agent_id * 40
        )
        self.y = WORLD_BOTTOM + 60
        self.vx = 0
        self.vy = 0
        self.grounded = True
        self.last_action = ACTION_NONE
        self.risk = 0.0
        # RC細胞
        self.rc = random.uniform(
            0.45,
            0.9
        )
        # 赫眼
        self.kakugan = False
        # 赫子
        self.kagune = 0.0
        # 赫子そのものが流動筋肉
        self.kagune_muscle = KaguneMuscle(self)
        self.dead = False
        self.goal = False
        self.heading = 1
        self.jumps = 0
        self.last_reset = -999
        self.error = 0
        self.role = [
            "EXPLORER",
            "OBSERVER",
            "GUARD"
        ][agent_id]
    def reset(self):
        self.x = (
            LEFT_ROOM_X1
            + 55
            + self.id * 40
        )
        self.y = WORLD_BOTTOM + 60
        self.vx = 0
        self.vy = 0
        self.grounded = True
        self.risk = 0.0
        self.rc = random.uniform(
            0.45,
            0.9
        )
        self.kakugan = False
        self.kagune = 0
        self.kagune_muscle.reset()
        self.dead = False
        self.goal = False
        self.last_action = ACTION_NONE
    def compute_internal_state(self):
        local_rc = self.world.rc.energy_at(
            self.x,
            self.y
        )
        # 喰種は環境を食う。
        self.rc += (
            local_rc - self.rc
        ) * 0.025
        self.rc -= RC_DECAY
        self.rc = clamp(
            self.rc,
            0.05,
            RC_MAX
        )
        self.kakugan = (
            self.rc >= KAKUGAN_THRESHOLD
            or self.risk > 0.72
        )
        self.kagune = clamp(
            (
                self.rc * 0.45
                + self.risk * 0.55
            ),
            0,
            KAGUNE_MAX
        )
        # 赫子量ではなく「筋肉としての可塑性」を内部状態へ接続。
        self.kagune_muscle.stimulate(
            max(0.0, self.kagune - 0.30) * 0.08
        )
    def observe(self):
        return self.vision.capture(
            agents,
            self.id,
            self.blackboard
        )
    def select_action(self, obs):
        return self.model.select_action(
            obs,
            self
        )
    def apply(self, action):
        if action == ACTION_LEFT:
            self.vx -= (
                GROUND_ACCEL
                if self.grounded
                else AIR_ACCEL
            )
            self.heading = -1
        elif action == ACTION_RIGHT:
            self.vx += (
                GROUND_ACCEL
                if self.grounded
                else AIR_ACCEL
            )
            self.heading = 1
        elif action == ACTION_JUMP:
            if self.grounded:
                self.vy = JUMP_POWER
                self.grounded = False
                self.jumps = 1
            elif self.jumps < 2:
                self.vy = JUMP_POWER * 0.75
                self.jumps += 1
        elif action == ACTION_BRAKE:
            self.vx *= 0.2
        elif action == ACTION_WAIT:
            self.vx *= 0.88
        elif action == ACTION_KAGUNE:
            self.use_kagune()
        elif action == ACTION_KAGUNE_CLAW:
            self.use_kagune_claw()
        elif action == ACTION_KAGUNE_FLOW:
            self.use_kagune_flow()
        self.vx = clamp(
            self.vx,
            -MAX_SPEED,
            MAX_SPEED
        )
        self.vx *= (
            GROUND_FRICTION
            if self.grounded
            else AIR_FRICTION
        )
        self.x += self.vx
        if not self.grounded:
            self.vy -= GRAVITY
            self.y += self.vy
        self.resolve_world()
    def use_kagune_claw(self):
        self.kagune_muscle.stimulate(0.22)
        self.kagune_muscle.mode = "claw"
        self.kagune_muscle.attack(agents)

    def use_kagune_flow(self):
        # 液状筋を流動化し、形状を探索する。
        self.kagune_muscle.stimulate(0.12)
        self.kagune_muscle.mode = "flow"

    def use_kagune(self):
        if self.kagune <= 0.05:
            return
        self.world.rc.disturb(
            self.x,
            self.y,
            -0.018
        )
        self.kagune_muscle.stimulate(
            0.18 * self.kagune
        )
        self.kagune_muscle.mode = "fluid"
        self.kagune_muscle.attack(agents)

    def resolve_world(self):
        # 壁
        if self.x < WORLD_LEFT + 15:
            self.x = WORLD_LEFT + 20
            self.vx *= -0.3
        if self.x > WORLD_RIGHT - 15:
            self.x = WORLD_RIGHT - 20
            self.vx *= -0.3
        if self.y < WORLD_BOTTOM + 15:
            self.y = WORLD_BOTTOM + 20
            self.vy = 0
            self.grounded = True
        if self.y > WORLD_TOP - 15:
            self.y = WORLD_TOP - 20
            self.vy *= -0.3
        # 生きた障害物
        for obstacle in self.world.obstacles:
            if obstacle.contains(
                self.x,
                self.y,
                10
            ):
                dl = abs(
                    self.x - obstacle.x1
                )
                dr = abs(
                    self.x - obstacle.x2
                )
                db = abs(
                    self.y - obstacle.y1
                )
                dt = abs(
                    self.y - obstacle.y2
                )
                m = min(
                    dl,
                    dr,
                    db,
                    dt
                )
                if m == dl:
                    self.x = (
                        obstacle.x1 - 12
                    )
                    self.vx *= -0.3
                elif m == dr:
                    self.x = (
                        obstacle.x2 + 12
                    )
                    self.vx *= -0.3
                elif m == db:
                    self.y = (
                        obstacle.y1 - 12
                    )
                    self.vy = 0
                    self.grounded = True
                else:
                    self.y = (
                        obstacle.y2 + 12
                    )
                    self.vy = 0
        # ゲート
        if (
            not self.world.gate_open
            and
            distance(
                self.x,
                self.y,
                GATE_X,
                GATE_Y
            ) < GATE_DISTANCE
        ):
            self.world.gate_open = True
        # ゴール
        if (
            distance(
                self.x,
                self.y,
                GOAL_X,
                GOAL_Y
            ) < GOAL_RADIUS
        ):
            self.goal = True
# ============================================================
# WORLD
# ============================================================
class World:
    def __init__(self):
        self.time = 0
        self.gate_open = False
        self.rc = RCField()
        self.obstacles = []
        self.chimimoryo = []
        self.reset_events = 0
        self.create_obstacles()
        for i in range(8):
            self.chimimoryo.append(
                Chimimoryo(
                    i,
                    self
                )
            )
    def create_obstacles(self):
        self.obstacles = [
            LivingObstacle(
                -400, -205,
                -275, -155,
                8, 0.12
            ),
            LivingObstacle(
                -235, 70,
                -110, 115,
                13, 0.15
            ),
            LivingObstacle(
                -470, -30,
                -370, 15,
                7, 0.18
            ),
            LivingObstacle(
                -105, -210,
                -72, -75,
                5, 0.13
            ),
            LivingObstacle(
                72, 75,
                105, 210,
                6, 0.16
            ),
            LivingObstacle(
                170, -190,
                285, -135,
                15, 0.14
            ),
            LivingObstacle(
                310, 35,
                440, 85,
                12, 0.17
            ),
            LivingObstacle(
                150, 125,
                245, 165,
                18, 0.13
            ),
            LivingObstacle(
                335, 135,
                385, 185,
                9, 0.21
            ),
        ]
    def reset(self):
        self.time = 0
        self.gate_open = False
        self.reset_events = 0
        self.rc.reset()
        for entity in self.chimimoryo:
            entity.reset()
        for obstacle in self.obstacles:
            obstacle.update(
                self.time,
                self.rc
            )
    def dynamic_bounds(self):
        e = clamp(
            self.rc.global_energy,
            0.5,
            1.5
        )
        bx = 12 * e
        by = 8 * e
        left = (
            WORLD_LEFT
            + math.sin(
                self.time * 0.021
            ) * bx
        )
        right = (
            WORLD_RIGHT
            - math.sin(
                self.time * 0.019
                + 1.8
            ) * bx
        )
        bottom = (
            WORLD_BOTTOM
            + math.cos(
                self.time * 0.017
            ) * by
        )
        top = (
            WORLD_TOP
            - math.cos(
                self.time * 0.015
                + 2
            ) * by
        )
        return (
            left,
            right,
            bottom,
            top
        )
    def step(self, agents, actions):
        self.time += 1
        self.rc.update(
            self.time
        )
        for obstacle in self.obstacles:
            obstacle.update(
                self.time,
                self.rc
            )
        for entity in self.chimimoryo:
            entity.update(
                agents
            )
        for agent, action in zip(
            agents,
            actions
        ):
            if not agent.dead:
                agent.compute_internal_state()
                agent.apply(action)
                agent.kagune_muscle.update(
                    action,
                    agent.risk
                )
                agent.kagune_muscle.attack(agents)
                agent.kagune_muscle.collide_with_monster(
                    monster
                )
# ============================================================
# TURTLE
# ============================================================
screen = turtle.Screen()
screen.setup(
    SCREEN_W,
    SCREEN_H
)
screen.bgcolor(
    "#050509"
)
screen.title(
    "TOKYO GHOUL : BOX OF CHAOS"
)
screen.tracer(False)
drawer = turtle.Turtle(
    visible=False
)
drawer.speed(0)
drawer.penup()
ui = turtle.Turtle(
    visible=False
)
ui.speed(0)
ui.penup()
# ============================================================
# DRAW
# ============================================================
def line(
    t,
    x1,
    y1,
    x2,
    y2,
    color,
    width=2
):
    t.color(color)
    t.pensize(width)
    t.goto(x1, y1)
    t.pendown()
    t.goto(x2, y2)
    t.penup()
def text(
    x,
    y,
    value,
    size=9,
    color="#cccccc"
):
    ui.goto(x, y)
    ui.color(color)
    ui.write(
        value,
        font=(
            "Arial",
            size,
            "normal"
        )
    )
def draw_world():
    drawer.clear()
    left, right, bottom, top = (
        world.dynamic_bounds()
    )
    # --------------------------------------------------------
    # 箱
    # --------------------------------------------------------
    rc = world.rc.global_energy
    wall_color = (
        "#a42d68"
        if rc > 0.9
        else "#3c2440"
    )
    line(
        drawer,
        left,
        bottom,
        left,
        top,
        wall_color,
        5
    )
    line(
        drawer,
        right,
        bottom,
        right,
        top,
        wall_color,
        5
    )
    line(
        drawer,
        left,
        top,
        right,
        top,
        wall_color,
        5
    )
    line(
        drawer,
        left,
        bottom,
        right,
        bottom,
        wall_color,
        5
    )
    # --------------------------------------------------------
    # トンネル
    # --------------------------------------------------------
    tunnel_wave = (
        math.sin(
            world.time * 0.032
        )
        * 16
    )
    lower = TUNNEL_Y1 - tunnel_wave
    upper = TUNNEL_Y2 + tunnel_wave
    tunnel_color = (
        "#ff277d"
        if rc > 0.85
        else "#632c5c"
    )
    line(
        drawer,
        TUNNEL_X1,
        lower,
        TUNNEL_X2,
        lower,
        tunnel_color,
        4
    )
    line(
        drawer,
        TUNNEL_X1,
        upper,
        TUNNEL_X2,
        upper,
        tunnel_color,
        4
    )
    # --------------------------------------------------------
    # RC node
    # --------------------------------------------------------
    for node in world.rc.nodes:
        r = int(
            clamp(
                2 + node.energy * 3,
                2,
                8
            )
        )
        drawer.goto(
            node.x,
            node.y
        )
        drawer.dot(
            r,
            "#8d235d"
        )
        drawer.dot(
            max(1, r // 2),
            "#ff3f98"
        )
    # --------------------------------------------------------
    # 障害物
    # --------------------------------------------------------
    for obstacle in world.obstacles:
        line(
            drawer,
            obstacle.x1,
            obstacle.y1,
            obstacle.x2,
            obstacle.y1,
            "#542744",
            3
        )
        line(
            drawer,
            obstacle.x2,
            obstacle.y1,
            obstacle.x2,
            obstacle.y2,
            "#542744",
            3
        )
        line(
            drawer,
            obstacle.x2,
            obstacle.y2,
            obstacle.x1,
            obstacle.y2,
            "#542744",
            3
        )
        line(
            drawer,
            obstacle.x1,
            obstacle.y2,
            obstacle.x1,
            obstacle.y1,
            "#542744",
            3
        )
    # --------------------------------------------------------
    # ゲート
    # --------------------------------------------------------
    drawer.goto(
        GATE_X,
        GATE_Y
    )
    drawer.dot(
        34,
        "#52ff9b"
        if world.gate_open
        else "#ffcc32"
    )
    # --------------------------------------------------------
    # GOAL
    # --------------------------------------------------------
    drawer.goto(
        GOAL_X,
        GOAL_Y
    )
    drawer.dot(
        38,
        "#39ff9a"
    )
    drawer.dot(
        16,
        "#07130d"
    )
    # --------------------------------------------------------
    # Chimimoryo
    # --------------------------------------------------------
    colors = {
        "drifter": "#a642ff",
        "hunter": "#ff315c",
        "swarm": "#42ffd0",
        "flee": "#596eff",
    }
    for entity in world.chimimoryo:
        color = colors[
            entity.mode
        ]
        pulse = math.sin(
            entity.phase
            + world.time * 0.08
        )
        r = entity.radius + pulse * 2
        drawer.goto(
            entity.x,
            entity.y
        )
        drawer.dot(
            r * 2,
            color
        )
        drawer.dot(
            max(
                2,
                r * 0.45
            ),
            "#09030d"
        )
    # --------------------------------------------------------
    # 喰種
    # --------------------------------------------------------
    for agent in agents:
        # --------------------------------------------------------
        # 赫子 = 液状の筋肉
        # --------------------------------------------------------
        muscle = agent.kagune_muscle
        if muscle.energy > 0.05:
            prev_x = agent.x
            prev_y = agent.y
            for i, seg in enumerate(muscle.segments):
                thickness = max(
                    1,
                    int(
                        KAGUNE_MUSCLE_WIDTH
                        * (0.55 + muscle.energy * 0.7)
                        * (1.0 - i / (KAGUNE_SEGMENTS * 2))
                    )
                )
                line(
                    drawer,
                    prev_x,
                    prev_y,
                    seg[0],
                    seg[1],
                    "#d51c5b" if muscle.mode != "claw"
                    else "#ff315c",
                    thickness
                )
                prev_x = seg[0]
                prev_y = seg[1]

            # 枝分かれした筋束。固定武器ではなく、一時的な流出。
            for branch in muscle.branches:
                prev = None
                for j, point in enumerate(branch):
                    if prev is not None:
                        line(
                            drawer, prev[0], prev[1], point[0], point[1],
                            "#8f164d" if muscle.mode != "hardening" else "#c72c68",
                            max(1, int(2 + muscle.energy * 2 - j * 0.2))
                        )
                    prev = point
            # 身体との境界が曖昧になる。
            if muscle.boundary > 0.10:
                drawer.goto(agent.x, agent.y)
                drawer.dot(18 + muscle.boundary * 22, "#5e1038")
            # 爪: 液状筋の先端だけ一時硬化
            if muscle.claw > 0.12:
                tx = muscle.tip_x
                ty = muscle.tip_y
                angle = math.atan2(
                    ty - agent.y,
                    tx - agent.x
                )
                for s in (-0.22, 0.0, 0.22):
                    a = angle + s
                    length = (
                        KAGUNE_CLAW_LENGTH
                        * muscle.claw
                    )
                    line(
                        drawer,
                        tx,
                        ty,
                        tx + math.cos(a) * length,
                        ty + math.sin(a) * length,
                        "#ffd1df",
                        max(1, int(3 * muscle.claw))
                    )
        # 身体
        drawer.goto(
            agent.x,
            agent.y
        )
        drawer.dot(
            18,
            agent.color
        )
        # 赫眼
        if agent.kakugan:
            drawer.goto(
                agent.x - 5,
                agent.y + 3
            )
            drawer.dot(
                5,
                "#ff003f"
            )
            drawer.goto(
                agent.x + 5,
                agent.y + 3
            )
            drawer.dot(
                5,
                "#ff003f"
            )
        text(
            agent.x - 10,
            agent.y + 16,
            f"A{agent.id}",
            7,
            agent.color
        )
    # --------------------------------------------------------
    # Monster
    # --------------------------------------------------------
    drawer.goto(
        monster.x,
        monster.y
    )
    drawer.dot(
        36,
        "#ff174e"
    )
    drawer.dot(
        12,
        "#190208"
    )
    if monster.target_id is not None:
        text(
            monster.x - 15,
            monster.y + 25,
            f"A{monster.target_id}",
            7,
            "#ff8195"
        )
def draw_ui():
    ui.clear()
    stats = model.statistics()
    text(
        -500,
        340,
        "TOKYO GHOUL : BOX OF CHAOS",
        17,
        "#ffffff"
    )
    text(
        -500,
        318,
        "REWARD-FREE / PREDICTION / CURIOSITY / SHARED MEMORY",
        9,
        "#ff4d96"
    )
    text(
        -500,
        296,
        f"EP {episode + 1}/{MAX_EPISODES} "
        f"STEP {step}/{STEPS_PER_EPISODE}",
        9,
        "#7dffd1"
    )
    text(
        -500,
        274,
        f"RC {world.rc.global_energy:.3f}",
        9,
        "#ff4d9c"
    )
    text(
        -500,
        255,
        f"GATE {'OPEN' if world.gate_open else 'CLOSED'}",
        9,
        "#66ffae"
        if world.gate_open
        else "#ffcc44"
    )
    text(
        -500,
        236,
        "NO REWARD",
        10,
        "#ffcf55"
    )
    text(
        -500,
        216,
        "未知 = 予測誤差 = 次の観測",
        9,
        "#c58cff"
    )
    text(
        210,
        300,
        "PREDICTIVE PREDATOR",
        10,
        "#ff4060"
    )
    text(
        210,
        280,
        f"TARGET A{monster.target_id}",
        8,
        "#ff8a9c"
    )
    text(
        210,
        262,
        f"SWITCH {monster.switches}",
        8,
        "#ff8a9c"
    )
    text(
        210,
        242,
        f"X {monster.x:+.0f} Y {monster.y:+.0f}",
        8,
        "#ff8a9c"
    )
    text(
        500,
        300,
        f"EXPERIENCE {stats['experience']}",
        9
    )
    text(
        500,
        280,
        f"PREDICTION ERROR {stats['error']:.3f}",
        9,
        "#ffbf55"
    )
    text(
        500,
        260,
        f"CURIOSITY {stats['curiosity']:.3f}",
        9,
        "#d889ff"
    )
    text(
        500,
        240,
        f"STABILITY {stats['stability']:.3f}",
        9,
        "#69d7ff"
    )
    text(
        500,
        220,
        f"MORPH MEMORY {len(model.morphology_memory)}",
        9,
        "#ff6ba6"
    )
    text(
        500,
        202,
        f"MORPH NOVELTY {model.morphology_novelty:.3f}",
        9,
        "#ffb1d0"
    )
    y = 150
    for agent in agents:
        text(
            -500,
            y,
            (
                f"A{agent.id} "
                f"{agent.role:<9} "
                f"{ACTION_NAMES[agent.last_action]:<7} "
                f"RC={agent.rc:.2f} "
                f"RISK={agent.risk:.2f} "
                f"MUSCLE={agent.kagune_muscle.mode:<10} "
                f"FLOW={agent.kagune_muscle.flow:.2f} "
                f"CLAW={agent.kagune_muscle.claw:.2f}"
            ),
            8,
            agent.color
        )
        y -= 18
    text(
        500,
        165,
        "思想 / INTERNAL AXIOMS",
        9,
        "#ffffff"
    )
    text(
        500,
        145,
        "1. 生存は報酬ではない",
        8,
        "#bbbbbb"
    )
    text(
        500,
        128,
        "2. 世界は理解されない",
        8,
        "#bbbbbb"
    )
    text(
        500,
        111,
        "3. 予測できない差異を記憶する",
        8,
        "#bbbbbb"
    )
    text(
        500,
        94,
        "4. 他者の記憶を自分の世界へ混ぜる",
        8,
        "#bbbbbb"
    )
    text(
        500,
        77,
        "5. 箱の外ではなく、箱の中を変える",
        8,
        "#ff6ba6"
    )
    text(
        500,
        60,
        "6. 赫子は武器ではなく、流動する筋肉である",
        8,
        "#ff6ba6"
    )
    text(
        500,
        43,
        "7. 爪は固定部品ではなく、一時的な硬化である",
        8,
        "#ff6ba6"
    )
    text(500, 26, "8. 身体の境界は固定されない", 8, "#ff6ba6")
    text(500, 9, "9. 変形そのものを経験として記憶する", 8, "#ff6ba6")
    text(500, -8, "10. 赫子は出るのではなく、身体が流れ出る", 8, "#ff6ba6")
# ============================================================
# GLOBAL OBJECTS
# ============================================================
world = World()
monster = PredictiveMonster(
    world
)
blackboard = Blackboard()
model = WorldModel()
vision = VisualField(
    world,
    monster
)
colors = [
    "#00eaff",
    "#55ff9b",
    "#ff9f43"
]
agents = [
    Ghoul(
        i,
        colors[i],
        world,
        model,
        vision,
        blackboard
    )
    for i in range(
        NUM_GHOULS
    )
]
# ============================================================
# EPISODE
# ============================================================
episode = 0
step = 0
finished = False
def reset_episode():
    global step
    step = 0
    world.reset()
    monster.reset()
    blackboard.data.clear()
    for agent in agents:
        agent.reset()
def end_episode():
    global episode
    global finished
    # --------------------------------------------------------
    # 報酬ではなく、記憶を再活性化
    # --------------------------------------------------------
    model.replay(
        250
    )
    episode += 1
    if episode >= MAX_EPISODES:
        finished = True
        draw_world()
        draw_ui()
        text(
            -160,
            -325,
            "BOX CLOSED",
            18,
            "#ffffff"
        )
        screen.update()
        return
    screen.ontimer(
        start_episode,
        700
    )
def run_step():
    global step
    if finished:
        return
    # --------------------------------------------------------
    # 1. 内部状態
    # --------------------------------------------------------
    for agent in agents:
        agent.compute_internal_state()
        # リスクは時間で少しずつ戻る
        agent.risk *= 0.994
        blackboard.update(
            agent
        )
    # --------------------------------------------------------
    # 2. 捕食者
    # --------------------------------------------------------
    monster.update(
        agents
    )
    # --------------------------------------------------------
    # 3. 観測と行動
    # --------------------------------------------------------
    observations = []
    actions = []
    for agent in agents:
        if agent.dead:
            observations.append(None)
            actions.append(
                ACTION_WAIT
            )
            continue
        before = agent.observe()
        action = agent.select_action(
            before
        )
        observations.append(
            before
        )
        actions.append(
            action
        )
    # --------------------------------------------------------
    # 4. 世界を進める
    # --------------------------------------------------------
    world.step(
        agents,
        actions
    )
    # --------------------------------------------------------
    # 5. 新しい共有記憶
    # --------------------------------------------------------
    for agent in agents:
        agent.compute_internal_state()
        blackboard.update(
            agent
        )
    # --------------------------------------------------------
    # 6. World Model
    # --------------------------------------------------------
    for agent, before, action in zip(
        agents,
        observations,
        actions
    ):
        if before is None:
            continue
        after = agent.observe()
        agent.error = model.learn(
            before,
            action,
            after
        )
        morph = agent.kagune_muscle.signature().copy()
        if not model.morphology_memory or min(float(np.mean(np.abs(m - morph))) for m in model.morphology_memory) > 0.045:
            model.morphology_memory.append(morph)
        if model.morphology_memory:
            model.morphology_novelty = clamp(
                min(float(np.mean(np.abs(m - morph))) for m in model.morphology_memory) * 4.0,
                0, 1
            )
        agent.last_action = action
    # --------------------------------------------------------
    # 7. 描画
    # --------------------------------------------------------
    draw_world()
    draw_ui()
    screen.update()
    step += 1
    if step < STEPS_PER_EPISODE:
        screen.ontimer(
            run_step,
            16
        )
    else:
        screen.ontimer(
            end_episode,
            200
        )
def start_episode():
    if finished:
        return
    reset_episode()
    run_step()
# ============================================================
# START
# ============================================================
reset_episode()
draw_world()
draw_ui()
screen.update()
screen.ontimer(
    start_episode,
    700
)
turtle.done()
