import turtle
import time

# ============================================================
# Gravity Gate
# Python turtle only
# Pygame 不使用
#
# 操作:
#   ↑ / W = 上
#   ↓ / S = 下
#   ← / A = 左
#   → / D = 右
#   R     = ステージリスタート
#
# ルール:
#   ・重力方向を変更するとプレイヤーとブロックが移動
#   ・ブロックまたはプレイヤーでスイッチを押す
#   ・スイッチが押されるとゲートが開く
#   ・ゴールに到達するとステージクリア
# ============================================================


# ============================================================
# 基本設定
# ============================================================

SCREEN_W = 900
SCREEN_H = 650

TILE = 42

MOVE_INTERVAL = 0.08
last_input_time = 0


# ============================================================
# ステージ
#
# # = 壁
# P = プレイヤー
# B = ブロック
# S = スイッチ
# D = ゲート
# G = ゴール
# . = 空間
# ============================================================

STAGES = [

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------
    [
        "###############",
        "#P            #",
        "#             #",
        "#     B       #",
        "#             #",
        "#       S   D #",
        "#             #",
        "#           G #",
        "###############",
    ],

    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------
    [
        "###############",
        "#P       #    #",
        "#        #    #",
        "#    B   #    #",
        "#        #    #",
        "#    S        #",
        "#             #",
        "#          G  #",
        "###############",
    ],

    # --------------------------------------------------------
    # STAGE 3
    # --------------------------------------------------------
    [
        "###############",
        "#P            #",
        "#   ###       #",
        "#   # B       #",
        "#   #         #",
        "#   S      D  #",
        "#             #",
        "#          G  #",
        "###############",
    ],

    # --------------------------------------------------------
    # STAGE 4
    # --------------------------------------------------------
    [
        "###############",
        "#P            #",
        "#      #      #",
        "#  B   #   D  #",
        "#      #      #",
        "#  S   #      #",
        "#      #      #",
        "#          G  #",
        "###############",
    ],

    # --------------------------------------------------------
    # STAGE 5
    # --------------------------------------------------------
    [
        "###############",
        "#P       B    #",
        "#   ###       #",
        "#             #",
        "#   S     D   #",
        "#             #",
        "#       B     #",
        "#          G  #",
        "###############",
    ],
]


# ============================================================
# Turtle画面
# ============================================================

screen = turtle.Screen()
screen.setup(SCREEN_W, SCREEN_H)
screen.title("Gravity Gate")
screen.bgcolor("#101522")
screen.tracer(0, 0)


# ============================================================
# Turtle生成
# ============================================================

def make_turtle(shape="square", color="white"):
    t = turtle.Turtle()
    t.shape(shape)
    t.color(color)
    t.penup()
    t.speed(0)
    return t


wall_t = make_turtle("square", "#30384f")
player_t = make_turtle("circle", "#55d6ff")
block_t = make_turtle("square", "#d89cff")
switch_t = make_turtle("square", "#3987ff")
gate_t = make_turtle("square", "#246bdb")
goal_t = make_turtle("circle", "#58ff91")

ui_t = make_turtle()
ui_t.hideturtle()

message_t = make_turtle()
message_t.hideturtle()


# ============================================================
# ゲーム状態
# ============================================================

grid = []

walls = set()
blocks = []

switch_pos = None
gate_pos = None
goal_pos = None

player = [0, 0]

gravity = [0, -1]

stage_index = 0

gate_open = False

game_running = True
transitioning = False


# ============================================================
# 座標変換
# ============================================================

def grid_to_screen(x, y):

    width = len(grid[0])
    height = len(grid)

    sx = (x - width / 2) * TILE + TILE / 2
    sy = (height / 2 - y) * TILE - TILE / 2

    return sx, sy


def place(t, x, y):

    sx, sy = grid_to_screen(x, y)
    t.goto(sx, sy)


# ============================================================
# 画面サイズ調整
# ============================================================

def resize_shapes():

    wall_t.shapesize(
        TILE / 20,
        TILE / 20
    )

    block_t.shapesize(
        TILE / 20,
        TILE / 20
    )

    switch_t.shapesize(
        TILE / 20,
        TILE / 20
    )

    gate_t.shapesize(
        TILE / 20,
        TILE / 20
    )

    player_t.shapesize(
        TILE / 20,
        TILE / 20
    )

    goal_t.shapesize(
        TILE / 20,
        TILE / 20
    )


# ============================================================
# UI
# ============================================================

def gravity_name():

    names = {
        (0, -1): "DOWN",
        (0, 1): "UP",
        (-1, 0): "LEFT",
        (1, 0): "RIGHT"
    }

    return names.get(tuple(gravity), "UNKNOWN")


def draw_ui():

    ui_t.clear()

    ui_t.color("#ffffff")

    ui_t.goto(
        0,
        -SCREEN_H / 2 + 35
    )

    ui_t.write(
        "STAGE {}/{}    GRAVITY: {}    GATE: {}".format(
            stage_index + 1,
            len(STAGES),
            gravity_name(),
            "OPEN" if gate_open else "CLOSED"
        ),
        align="center",
        font=("Arial", 16, "bold")
    )

    ui_t.goto(
        0,
        SCREEN_H / 2 - 40
    )

    ui_t.write(
        "WASD / ARROW = GRAVITY     R = RESTART",
        align="center",
        font=("Arial", 12, "normal")
    )


# ============================================================
# ステージ描画
# ============================================================

def draw_stage():

    wall_t.clear()
    player_t.clear()
    block_t.clear()
    switch_t.clear()
    gate_t.clear()
    goal_t.clear()

    # 壁
    for x, y in walls:
        place(wall_t, x, y)

    # ブロック
    for b in blocks:
        place(
            block_t,
            b[0],
            b[1]
        )

    # スイッチ
    if switch_pos is not None:

        place(
            switch_t,
            switch_pos[0],
            switch_pos[1]
        )

    # ゲート
    if gate_pos is not None and not gate_open:

        place(
            gate_t,
            gate_pos[0],
            gate_pos[1]
        )

    # ゴール
    if goal_pos is not None:

        place(
            goal_t,
            goal_pos[0],
            goal_pos[1]
        )

    # プレイヤー
    place(
        player_t,
        player[0],
        player[1]
    )

    draw_ui()

    screen.update()


# ============================================================
# ステージ読み込み
# ============================================================

def load_stage(index):

    global grid
    global walls
    global blocks
    global switch_pos
    global gate_pos
    global goal_pos
    global player
    global gravity
    global gate_open
    global transitioning

    transitioning = False

    grid = STAGES[index]

    walls = set()
    blocks = []

    switch_pos = None
    gate_pos = None
    goal_pos = None

    player = [0, 0]

    gravity = [0, -1]

    gate_open = False

    message_t.clear()

    for y, row in enumerate(grid):

        for x, cell in enumerate(row):

            if cell == "#":

                walls.add((x, y))

            elif cell == "P":

                player = [x, y]

            elif cell == "B":

                blocks.append([x, y])

            elif cell == "S":

                switch_pos = (x, y)

            elif cell == "D":

                gate_pos = (x, y)

            elif cell == "G":

                goal_pos = (x, y)

    resize_shapes()

    update_switch()

    draw_stage()


# ============================================================
# 壁判定
# ============================================================

def is_wall(x, y):

    # ステージ外
    if y < 0:
        return True

    if y >= len(grid):
        return True

    if x < 0:
        return True

    if x >= len(grid[0]):
        return True

    # 通常の壁
    if (x, y) in walls:
        return True

    # 閉じたゲート
    if gate_pos is not None:

        if not gate_open:

            if (x, y) == gate_pos:
                return True

    return False


# ============================================================
# ブロック検索
# ============================================================

def block_at(x, y, ignore=None):

    for i, b in enumerate(blocks):

        if ignore is not None and i == ignore:
            continue

        if b[0] == x and b[1] == y:
            return i

    return None


# ============================================================
# スイッチ
# ============================================================

def update_switch():

    global gate_open

    if switch_pos is None:

        gate_open = False
        return

    # プレイヤーがスイッチ上
    player_on = (
        tuple(player) == tuple(switch_pos)
    )

    # ブロックがスイッチ上
    block_on = (
        block_at(
            switch_pos[0],
            switch_pos[1]
        ) is not None
    )

    gate_open = player_on or block_on


# ============================================================
# プレイヤー移動
# ============================================================

def try_move_player(dx, dy):

    nx = player[0] + dx
    ny = player[1] + dy

    # 壁
    if is_wall(nx, ny):
        return False

    # ブロック
    bi = block_at(nx, ny)

    if bi is not None:

        bnx = blocks[bi][0] + dx
        bny = blocks[bi][1] + dy

        # ブロックの先が壁
        if is_wall(bnx, bny):
            return False

        # ブロックの先に別ブロック
        if block_at(
            bnx,
            bny,
            ignore=bi
        ) is not None:

            return False

        # ブロックを押す
        blocks[bi][0] = bnx
        blocks[bi][1] = bny

    # プレイヤー移動
    player[0] = nx
    player[1] = ny

    return True


# ============================================================
# ブロックの重力移動
# ============================================================

def move_blocks_with_gravity(dx, dy):

    # 重力方向に近い順から処理
    # これによりブロック同士の不自然な重なりを防ぐ

    if dx > 0:
        order = sorted(
            range(len(blocks)),
            key=lambda i: blocks[i][0],
            reverse=True
        )

    elif dx < 0:
        order = sorted(
            range(len(blocks)),
            key=lambda i: blocks[i][0]
        )

    elif dy > 0:
        order = sorted(
            range(len(blocks)),
            key=lambda i: blocks[i][1]
        )

    else:
        order = sorted(
            range(len(blocks)),
            key=lambda i: blocks[i][1],
            reverse=True
        )

    for i in order:

        b = blocks[i]

        nx = b[0] + dx
        ny = b[1] + dy

        if is_wall(nx, ny):
            continue

        # プレイヤー位置
        if [nx, ny] == player:
            continue

        # 他ブロック
        other = block_at(
            nx,
            ny,
            ignore=i
        )

        if other is not None:
            continue

        b[0] = nx
        b[1] = ny


# ============================================================
# 重力適用
# ============================================================

def apply_gravity():

    global player

    dx, dy = gravity

    # --------------------------------------------------------
    # プレイヤーとブロックを少しずつ移動
    # --------------------------------------------------------

    for _ in range(100):

        moved = False

        # プレイヤーの次位置
        nx = player[0] + dx
        ny = player[1] + dy

        # ----------------------------------------------------
        # プレイヤー前方にブロックがある
        # ----------------------------------------------------

        bi = block_at(nx, ny)

        if bi is not None:

            bnx = blocks[bi][0] + dx
            bny = blocks[bi][1] + dy

            # ブロックが動けるか
            if (
                not is_wall(bnx, bny)
                and block_at(
                    bnx,
                    bny,
                    ignore=bi
                ) is None
                and [bnx, bny] != player
            ):

                blocks[bi][0] = bnx
                blocks[bi][1] = bny

                player[0] = nx
                player[1] = ny

                moved = True

        else:

            # プレイヤーだけ移動
            if not is_wall(nx, ny):

                player[0] = nx
                player[1] = ny

                moved = True

        update_switch()

        # ゴール
        if check_goal():
            return True

        if not moved:
            break

    return check_goal()


# ============================================================
# ゴール判定
# ============================================================

def check_goal():

    if goal_pos is None:
        return False

    return (
        player[0] == goal_pos[0]
        and player[1] == goal_pos[1]
    )


# ============================================================
# クリアメッセージ
# ============================================================

def show_message(text, subtext=""):

    message_t.clear()

    message_t.color("#ffffff")

    message_t.goto(0, 70)

    message_t.write(
        text,
        align="center",
        font=("Arial", 34, "bold")
    )

    if subtext:

        message_t.goto(0, 20)

        message_t.write(
            subtext,
            align="center",
            font=("Arial", 16, "normal")
        )

    screen.update()


# ============================================================
# ステージクリア
# ============================================================

def stage_clear():

    global stage_index
    global transitioning

    if transitioning:
        return

    transitioning = True

    show_message(
        "STAGE CLEAR!",
        "Next stage..."
    )

    # Turtleのイベントを止めない
    screen.ontimer(
        next_stage,
        900
    )


# ============================================================
# 次ステージ
# ============================================================

def next_stage():

    global stage_index
    global transitioning

    if stage_index + 1 < len(STAGES):

        stage_index += 1

        transitioning = False

        load_stage(stage_index)

    else:

        game_complete()


# ============================================================
# 全クリア
# ============================================================

def game_complete():

    global game_running

    game_running = False

    show_message(
        "ALL STAGES CLEAR!",
        "Congratulations!"
    )


# ============================================================
# 重力変更
# ============================================================

def set_gravity(dx, dy):

    global gravity
    global last_input_time

    if not game_running:
        return

    if transitioning:
        return

    # 入力連打防止
    now = time.monotonic()

    if now - last_input_time < MOVE_INTERVAL:
        return

    last_input_time = now

    gravity = [dx, dy]

    update_switch()

    # 重力を適用
    reached_goal = apply_gravity()

    update_switch()

    draw_stage()

    if reached_goal or check_goal():

        stage_clear()


# ============================================================
# 重力操作
# ============================================================

def gravity_up():
    set_gravity(0, 1)


def gravity_down():
    set_gravity(0, -1)


def gravity_left():
    set_gravity(-1, 0)


def gravity_right():
    set_gravity(1, 0)


# ============================================================
# リスタート
# ============================================================

def restart():

    global transitioning

    if not game_running:
        return

    transitioning = False

    load_stage(stage_index)


# ============================================================
# 全ステージ最初から
# ============================================================

def restart_game():

    global stage_index
    global game_running
    global transitioning

    stage_index = 0
    game_running = True
    transitioning = False

    load_stage(0)


# ============================================================
# キー設定
# ============================================================

screen.listen()

# 矢印
screen.onkey(gravity_up, "Up")
screen.onkey(gravity_down, "Down")
screen.onkey(gravity_left, "Left")
screen.onkey(gravity_right, "Right")

# WASD
screen.onkey(gravity_up, "w")
screen.onkey(gravity_down, "s")
screen.onkey(gravity_left, "a")
screen.onkey(gravity_right, "d")

# 大文字対策
screen.onkey(gravity_up, "W")
screen.onkey(gravity_down, "S")
screen.onkey(gravity_left, "A")
screen.onkey(gravity_right, "D")

# リスタート
screen.onkey(restart, "r")
screen.onkey(restart, "R")

# 全ゲームリスタート
screen.onkey(restart_game, "n")


# ============================================================
# 起動
# ============================================================

load_stage(0)

screen.update()

turtle.mainloop()
