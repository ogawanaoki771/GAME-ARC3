# ```DEMO.py```の説明
# STRANGE PLACE / SELF-ORGANIZING WORLD v3(名称)

## ゲーム仕様書

**Version:** 3.0
**ゲーム形式:** 2Dリアルタイム・物理シミュレーション＋自己組織化World Model
**実装:** Python / Turtle / NumPy
**プレイヤー操作:** なし（自律エージェントによる探索）

---

# 1. ゲーム概要

本ゲームは、通常の2Dアクションゲームの物理法則に、**重力反転・局所重力・時間変動・空間変形・ワームホール・過去の残像**などの異常現象を組み込んだ「奇妙な世界」を構築する。

世界内には複数の自律エージェントが存在する。

エージェントは世界のルールをあらかじめ知らず、視覚情報・自身の物理状態・世界の局所的なコンテキストを観測しながら行動する。

経験は以下の階層で記憶される。

```text
Visual Cell
    ↓
Place State
    ↓
Transformation
    ↓
Meta Transformation
    ↓
Temporal Trace
```

ゲームの目的は、単純なゴール到達ではなく、

> **変化する世界の中から、安定した場所・状態・変化の規則を発見し、内部World Modelを自己組織化させること**

である。

---

# 2. ゲーム画面

画面サイズは以下とする。

| 項目      |       値 |
| ------- | ------: |
| 画面幅     | 1240 px |
| 画面高さ    |  820 px |
| World左端 |    -560 |
| World右端 |     560 |
| World下端 |    -300 |
| World上端 |     290 |

画面は大きく以下の領域から構成される。

```text
┌─────────────────────────────────────────────┐
│                ゲームタイトル               │
│ EPISODE / STEP        GRAVITY STATUS       │
├─────────────────────────────────────────────┤
│                                             │
│                                             │
│                 GAME WORLD                  │
│                                             │
│                                             │
│                                             │
├─────────────────────────────┬───────────────┤
│ Agent Status                │ Place States  │
│                             │               │
│                             │ Transformations│
└─────────────────────────────┴───────────────┘
```

---

# 3. ゲーム進行

ゲームはエピソード単位で進行する。

### エピソード設定

```text
MAX_EPISODES = 14
```

1エピソードあたり、

```text
STEPS_PER_EPISODE = 420
```

ステップ実行後、World Modelの再生・減衰処理を行う。

したがってゲーム全体では、

```text
14 episodes × 420 steps
= 5,880 simulation steps
```

を基本とする。

---

# 4. エージェント

ゲーム内には3体の自律エージェントを配置する。

```text
NUM_AGENTS = 3
```

エージェントにはそれぞれ異なる色を設定する。

| Agent | 色      |
| ----- | ------ |
| A0    | Cyan   |
| A1    | Green  |
| A2    | Orange |

エージェントはTurtle形状で表示される。

---

# 5. エージェントの目的

エージェントには、従来型ゲームのような明示的な「ゴール地点」は存在しない。

エージェントの基本目的は、

1. 世界を観測する
2. 行動する
3. 結果を観測する
4. 変化を記憶する
5. 未知・不安定な現象を探索する
6. 世界モデルを更新する

ことである。

特にTransformationの不安定性や未知性が高い状況は、エージェントにとって探索価値が高い。

---

# 6. 行動仕様

エージェントは以下の6種類の行動を選択する。

| ID | 行動    | 内容        |
| -: | ----- | --------- |
|  0 | NONE  | 何もしない     |
|  1 | LEFT  | 左方向へ加速    |
|  2 | RIGHT | 右方向へ加速    |
|  3 | JUMP  | ジャンプ      |
|  4 | BRAKE | 水平方向速度を減衰 |
|  5 | WAIT  | 待機        |

---

## 6.1 LEFT

左方向へ加速する。

地上では、

```text
GROUND_ACCEL = 1.05
```

空中では、

```text
AIR_ACCEL = 0.65
```

を使用する。

---

## 6.2 RIGHT

右方向へ加速する。

加速度はLEFTと同一。

---

## 6.3 JUMP

接地状態の場合、重力方向と反対方向へジャンプする。

### 通常重力

```text
重力 ↓
JUMP → ↑
```

### 上向き重力

```text
重力 ↑
JUMP → ↓
```

基本ジャンプ力は、

```text
JUMP_POWER = 11.2
```

とする。

最大2回までジャンプ可能。

---

## 6.4 BRAKE

水平方向速度を大幅に減少させる。

---

## 6.5 WAIT

その場で待機しながら、水平方向速度を緩やかに減衰させる。

---

# 7. 自律行動選択

エージェントは完全なランダム行動ではなく、World Modelを利用して行動を決定する。

基本的な評価軸は、

```text
Action Score
    =
Activity Bias
+ Novelty
+ Curiosity
+ Instability
+ Branching
```

である。

また、一定確率で探索行動を強制する。

```text
探索確率 = 13%
```

探索時には、

```text
LEFT
RIGHT
JUMP
WAIT
BRAKE
```

からランダムに選択する。

---

# 8. 物理仕様

## 8.1 基本重力

```text
BASE_GRAVITY = 0.72
```

ただし実際の重力は固定ではない。

---

# 9. 動的重力

本ゲーム最大の特徴。

重力には、

1. グローバル重力方向
2. 空間的な局所重力方向
3. 重力強度

の3要素が存在する。

---

## 9.1 グローバル重力

初期状態：

```text
gravity_sign = +1
```

つまり、

```text
↓ DOWNWARD GRAVITY
```

で開始する。

一定時間経過すると符号が反転する。

```text
GRAVITY_SWITCH_PERIOD = 170 frames
```

したがって、

```text
↓
↓
↓
↓
↓
↓
↓
↑
↑
↑
↑
...
```

という周期的な重力反転が発生する。

---

# 10. 重力反転予告

重力反転の28フレーム前から予告を表示する。

```text
GRAVITY_WARNING_TIME = 28
```

画面中央上部付近に黄色い警告マークを表示する。

表示内容：

* 円
* 三角形
* `!`
* 点滅・パルス

画面UIには、

```text
⚠ GRAVITY REVERSAL IMMINENT
```

を表示する。

---

# 11. 上向き重力

重力が上向きの場合、

```text
GRAVITY ↑ UP
```

となる。

この状態では、エージェントは天井側へ引っ張られる。

天井プラットフォームへの衝突時には反発処理を行う。

```text
vy = -abs(vy) × 0.88
```

反発係数：

```text
UPWARD_GRAVITY_BOUNCE = 0.88
```

さらに水平速度も、

```text
vx × 0.82
```

に減衰する。

---

# 12. 重力衝突イベント

上向き重力中に天井へ衝突すると、ゲーム内イベントとして記録する。

イベント内容：

```text
UPWARD_GRAVITY_IMPACT
```

イベント発生時：

* ImpactMark生成
* Agentの`last_impact`更新
* 反発処理
* 接地状態変更
* ジャンプ回数リセット

を行う。

---

# 13. Impact Mark

衝突地点には大きな視覚エフェクトを生成する。

Impact Markは、

* 外側の衝撃波
* X字
* 中央十字
* 重力方向矢印

から構成される。

### 通常重力

青系：

```text
#55ddff
```

### 上向き重力

赤系：

```text
#ff4055
```

衝突後、時間経過によって縮退・消滅する。

---

# 14. プラットフォーム

世界には通常プラットフォームと天井プラットフォームが存在する。

## 通常プラットフォーム

通常重力時の足場となる。

例：

```text
(-530, -220, -220)
(-470, -300, -105)
(-340, -250, 40)
...
```

---

## 天井プラットフォーム

上向き重力時の足場となる。

例：

```text
(-500, -350, 235)
(-330, -190, 195)
(-150, -20, 250)
...
```

天井プラットフォームは通常プラットフォームより太く表示される。

---

# 15. 動的地形

プラットフォームは完全固定ではない。

Worldの`topology_phase`によって位置が変化する。

通常プラットフォームには、

* wobble
* shear

が適用される。

そのため同じ座標付近でも、時間によって地形が変化する。

---

# 16. Extra Platform

Topology Phaseが一定値を超えると追加プラットフォームが出現する。

```text
topology_phase > 0.48
```

の場合、Extra Platformが段階的に追加される。

これにより、同一地点でも時間によって異なる経路が形成される。

---

# 17. Hazard

世界には危険地帯が存在する。

Hazardは赤色の線として表示される。

AgentがHazard領域に侵入した場合、

```text
Agent → 初期位置へリセット
```

される。

初期位置：

```text
x = -480 + agent_id × 24
y = -195
```

---

# 18. Moving Object

世界には移動するオレンジ色のオブジェクトが存在する。

位置は時間関数によって変化する。

目的は、

* 動的障害
* 視覚変化
* World Modelへの時間情報提供

である。

---

# 19. Strange World Events

世界には以下の異常現象が存在する。

| 現象           |  周期 |
| ------------ | --: |
| Global Phase | 270 |
| Time Warp    | 220 |
| Topology     | 340 |
| Past Leak    | 260 |
| Gravity Flip | 170 |

これらは互いに異なる周期で変動するため、世界は単純な周期運動にはならない。

---

# 20. Time Warp

時間速度は、

```text
time_scale =
0.30 + 1.65 × time_phase
```

によって変化する。

したがってWorldの時間進行速度は、

```text
最小 ≒ 0.30
最大 ≒ 1.95
```

の範囲で変動する。

これにより、

* 移動速度
* 重力作用
* Moving Object
* 地形変化

などの時間依存現象が変化する。

---

# 21. Local Gravity

重力は世界全体で完全に均一ではない。

位置`x,y`と時間によって局所重力が変化する。

そのため、

```text
画面左では ↓
中央では ↑
右では ↓
```

のような状態が発生する可能性がある。

これは本ゲームにおける「奇妙な世界」の重要な要素である。

---

# 22. Mirror

Topology Phaseが高い状態ではMirror現象が発生する。

```text
topology_phase > 0.82
```

で有効になる。

一定領域では、14フレームごとに水平方向速度が反転する。

```text
vx → -vx
```

これにより、通常の操作感覚とは異なる空間を形成する。

---

# 23. Wormhole

Global Phaseが高いとWormholeが出現する。

Wormhole位置：

```text
x ≈ 0
y ≈ 130
```

一定範囲内にAgentが入ると、反対側へ転送する。

例：

```text
左側 → 右側
右側 → 左側
```

転送後は、

* x座標反転
* y座標変更
* vx反転
* vy減衰

を行う。

---

# 24. Past Leak

Past Leakが強い場合、過去を示すGhostが出現する。

Ghostは紫色のオブジェクトとして表示される。

一定条件：

```text
past_leak_strength > 0.35
```

で表示する。

Ghostは主として視覚的・記憶的な時間情報として機能する。

---

# 25. 観測システム

Agentは世界全体を直接取得するのではなく、Visual Fieldから観測画像を取得する。

観測サイズ：

```text
72 × 44
```

チャンネル数：

```text
4
```

したがって観測データ量は、

```text
72 × 44 × 4
= 12,672 values
```

となる。

---

# 26. Visual Channels

| Channel | 内容                                        |
| ------- | ----------------------------------------- |
| 0       | 地形・プラットフォーム                               |
| 1       | Hazard / Moving Object / Ghost / Wormhole |
| 2       | Agent                                     |
| 3       | Gravity / Temporal Information            |

これによりAgentは画像的なWorld Stateを取得できる。

---

# 27. Agent Body State

視覚情報とは別に、自身の身体状態を取得する。

Body State：

```text
x
y
vx
vy
grounded
jumps
heading
recent impact
```

位置・速度は正規化して記録する。

---

# 28. Local Context

Agentは現在地点の局所的な世界情報も取得する。

Context：

```text
local gravity
local phase
time scale
wormhole strength
past leak strength
mirror strength
global gravity sign
```

これにより、単なる画像認識だけではなく、

> 「現在の世界がどのような法則状態にあるか」

を記憶できる。

---

# 29. World Model

World Modelは以下の5階層から構成される。

```text
Visual Representation
        ↓
Place Representation
        ↓
Transformation Representation
        ↓
Meta Transformation
        ↓
Temporal Memory
```

---

# 30. Visual Cell

Visual Cellは局所的な視覚パターンを記憶する。

画像を6×6 pixel相当のPatchに分割する。

```text
PATCH = 6
```

各Patchについて、

```text
平均RGBA相当特徴
```

を計算する。

類似度が閾値未満の場合、新しいVisualCellを生成する。

```text
VISUAL_SIM_THRESHOLD = 0.74
```

---

# 31. Place State

Place Stateは、

> 「ここはどのような場所・状態なのか」

を表現する。

Place Stateは、

* Visual Feature
* Body State
* Local Context

から構成される。

類似度：

```text
PLACE_SIM_THRESHOLD = 0.80
```

を下回る場合、新しいPlace Stateを生成する。

---

# 32. Place Split

同じPlaceに異なる状態が長期間混在した場合、そのPlaceを分割する。

Split Pressureが増加し、

```text
split_pressure >= 1.0
```

かつ十分な履歴が存在する場合にSplitを検討する。

これにより、

```text
同じ場所
   ↓
通常重力時の場所
上向き重力時の場所
```

のような状態分離が可能になる。

---

# 33. Transformation

Transformationは、

```text
Place A
   +
Action
   ↓
Place B
```

という変化を記憶する。

同時にState Delta、

```text
ΔState = State_after - State_before
```

を記録する。

---

# 34. Transformationの評価

Transformationには以下の情報を持たせる。

* Visit count
* Error
* Stability
* Energy
* Action distribution
* Context history
* Curiosity

Transformationが不安定なほど探索価値が高くなる。

---

# 35. Curiosity

Curiosityは主に、

```text
Novelty
+
Instability
+
Isolation
+
Low Stability
```

から計算する。

未知のTransformationや不安定なTransformationは高いCuriosityを持つ。

---

# 36. Meta Transformation

Meta Transformationは、

> 「Transformationそのものがどのように変化するか」

を記憶する。

例えば、

```text
Transformation A
      ↓
Transformation B
```

という連続関係から、

```text
Meta Transformation
```

を生成する。

これは通常の状態遷移より高次の世界構造を表現する。

---

# 37. Temporal Trace

Temporal Traceは過去の状態を保持する。

記録内容：

* Feature
* Body
* Place ID
* Time Index
* Strength

最大700件程度を保持する。

時間経過によってStrengthは減衰する。

---

# 38. Replay

エピソード終了後、World Modelは過去のTransformationをReplayする。

1回のReplayフェーズで、

```text
520 iterations
```

を実行する。

Replay対象は、

* Errorが大きい
* Curiosityが高い
* Stabilityが低い

Transformationほど選ばれやすい。

---

# 39. Memory Decay

記憶は永続的ではない。

以下が徐々に減衰する。

* VisualCell activation
* VisualCell energy
* Place activation
* Place energy
* Transformation energy
* MetaTransformation energy
* TemporalTrace strength

これにより、世界モデルが無限に膨張することを抑える。

---

# 40. エピソードリセット

エピソード終了時には、

* World時間
* Agent位置
* Agent速度
* Agent状態
* Gravity状態
* Moving Object
* Ghost
* Impact Mark

などの一時的なWorld Stateをリセットする。

一方、

```text
WorldModel
```

はリセットしない。

つまり、

```text
Episode 1
   ↓
Memory
   ↓
Episode 2
   ↓
Memory
   ↓
Episode 3
   ↓
...
```

という累積学習構造になる。

---

# 41. 重要な設計原則

本ゲームでは、**世界はAgentのために固定されない**。

世界は、

```text
重力
地形
時間
空間
危険
異常現象
```

が常に変化する。

したがってAgentは、

> 「この場所では、この行動をすれば、必ずこうなる」

という単純なルールだけでは世界を説明できない。

Agentは、

```text
場所
+
時間
+
重力
+
局所状態
+
過去
```

を組み合わせて世界を理解する必要がある。

---

# 42. 勝敗条件

現行v3では、明確なプレイヤー勝敗条件は設定しない。

ゲームの終了条件は、

```text
14 Episodes 完了
```

とする。

終了時に、

```text
SIMULATION FINISHED
```

を表示する。

ゲームの評価対象は「勝利」ではなく、

* Visual Cell数
* Place State数
* Transformation数
* Meta Transformation数
* Prediction / Transition Error
* Stability
* Split数
* Curiosity

などのWorld Model形成状況とする。

---

# 43. UI表示項目

画面には最低限以下を表示する。

### World情報

```text
GRAVITY ↑ UP
GRAVITY ↓ DOWN

Flip in XX
```

### 学習情報

```text
VisualCells
PlaceStates
Transforms
Meta
Err
Stability
Splits
```

### Agent情報

```text
A0 action x y vx vy err
A1 action x y vx vy err
A2 action x y vx vy err
```

### World Model

```text
PLACE STATES

TRANSFORMATION CURIOSITY
```

---

# 44. ゲームループ

ゲーム全体の基本ループは以下。

```text
┌───────────────────────┐
│ World Update          │
│  ├─ Gravity           │
│  ├─ Time              │
│  ├─ Topology          │
│  ├─ Wormhole          │
│  └─ Past Leak         │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Agent Observation     │
│  ├─ Visual            │
│  ├─ Body              │
│  └─ Context           │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Action Selection      │
│  ├─ Curiosity         │
│  ├─ Novelty           │
│  └─ Exploration       │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Physics               │
│  ├─ Movement          │
│  ├─ Gravity           │
│  ├─ Collision         │
│  └─ Hazard            │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ After Observation     │
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ World Model Learning  │
│  ├─ VisualCell        │
│  ├─ PlaceState        │
│  ├─ Transformation    │
│  ├─ Meta              │
│  └─ TemporalTrace     │
└───────────┬───────────┘
            ↓
       Next Step
```

エピソード終了後は、

```text
Replay
  ↓
Decay
  ↓
Episode Reset
  ↓
Next Episode
```

とする。

---

# 45. v3のゲームコンセプト

本作の核心は、

> **「世界のルールを教えられていない存在が、変化し続ける世界を経験だけから理解できるか」**

という実験である。

特に重要なゲーム要素は、

```text
固定された世界
        ↓
ではなく

変化する世界
        ↓
重力反転
        ↓
地形変形
        ↓
時間変動
        ↓
空間転送
        ↓
過去の出現
        ↓
Agentが経験する
        ↓
記憶を形成する
        ↓
世界モデルが自己組織化する
```

という循環構造にある。

したがって本ゲームでは、**「ゲーム世界そのもの」と「世界を理解するAgentの内部モデル」の両方がゲームの主要な状態**となる。
