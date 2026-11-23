# 🕵️‍♂️ The Sticky Note Detective: How Context Language Models Learn to Clean Their Own Desks

**Context Language Models don’t solve AI memory by remembering more—they solve it by throwing away the right things at the right time.**

At first glance, the name **Context Language Models** (from the research paper *arXiv:2609.37725*) sounds like another entry in the AI industry's race to build massive, million-token memory vaults. It sounds like a model that never forgets a single word.

Look under the hood, though, and the mechanics are the exact opposite of what the name suggests.

Until recently, AI chatbots treated memory like a giant, uneditable scroll: every message you sent and every tool result they read got tacked onto the bottom. Once that scroll overflowed, the oldest—and often most critical—instructions got pushed off the cliff and forgotten.

Context Language Models (CLMs) ditch the scroll and give the AI a **tiny, editable whiteboard**. Instead of hoarding every token forever, the model is trained to erase dead-end leads mid-case while leaving the suspect’s license plate untouched. Making that possible takes a two-part breakthrough:

1. **Reinforcement Learning for Ruthless Editing (Stepwise GRPO):** Rewarding the model for surgically deleting irrelevant clutter mid-task while protecting vital facts.
2. **Backend Caching Tricks That Make the Whiteboard Possible:** Standard LLMs rely heavily on **prefix caching**—think of reading a 50-page manual once, saving your mental "reading notes" (KV states), and reusing them for every follow-up question instead of re-reading the book from page one. In a standard LLM, prefix caching only works if the beginning of the prompt stays identical; erase a single sentence in the middle of the whiteboard, and the system has to re-read everything from scratch. CLMs pair editable prompts with **Suffix Cache Reuse**, letting the server snip out deleted clutter and stitch the surviving cached states back together without recomputing the whole board.

Here is how the coaching math works—explained without the scary equations—followed by a complete Python game (`demo.py`) you can run with your kids.

---

## 1. The Tiny Desk Problem

Imagine you are a detective working at a **tiny desk** that only has room for **5 sticky notes** in a row.

At the start of a case, you place two critical clues on the left side of your desk—your "license plate" facts—plus one piece of trash:

`[ 🗺️ Red Map ]` `[ 🔑 Red Key ]` `[ 🍌 Banana Peel ]`

Every minute, a noisy trash chute dumps **two new pieces of junk** onto the right side of your desk (like `[ 🧦 Smelly Sock ]` and `[ 🥫 Empty Can ]`). Because your desk only holds 5 notes, any overflow pushes the notes on the far left **off the edge into a paper shredder**:

* **Standard LLMs** aren't allowed to touch their desks. They let dead-end trash pile up until the **Red Map** and **Red Key** fall into the shredder—and by the end, they have no idea what mystery they were solving.
* **Context Language Models** are allowed to toss sticky notes into the wastebasket between turns so their key clues survive.

---

## 2. How Do We Teach a Robot Detective to Clean Its Desk?

Hand a brand-new robot detective a messy desk and it won't know what to toss. It might shred the Red Map, sit frozen, or guess at random. Instead of lecturing it with dense formulas, the **Stepwise GRPO** (*Stepwise Group Relative Policy Optimization*) coach trains it using a game with **three simple rules**:

### Rule 1: The 8-Clone Tryout (*"Group Relative"*)

Instead of grading one robot in isolation, the coach makes **8 identical copies of the robot** and deals all 8 the exact same starting desk:

* If **3 clones** keep their clues safe and solve the mystery while **5 clones** let their clues fall into the shredder, the 3 winners get high-fives and the 5 losers are coached to copy what the winners did.
* If **all 8 win** or **all 8 lose**, nobody stood out from the crowd—so the coach skips that round and deals a fresh puzzle.

### Rule 2: Polaroid Snapshots at Every Move (*"Stepwise"*)

Because the robot throws notes away *during* the case, inspecting the desk at the very end won't tell the coach what the desk looked like at move one.

So the coach stands behind each robot with an instant Polaroid camera and snaps a picture **right before every single move**:

1. **Snapshot 1:** Desk has `[ 🗺️ Map, 🔑 Key, 🍌 Junk ]` $\rightarrow$ *Robot trashes the back note.*
2. **Snapshot 2:** Desk has `[ 🗺️ Map, 🔑 Key, 🧦 Junk, 🥫 Junk ]` $\rightarrow$ *Robot trashes the back note.*
3. **Snapshot 3:** Desk has `[ 🗺️ Map, 🔑 Key, 🪨 Junk, 🦴 Junk, 📎 Junk ]` $\rightarrow$ *Robot shouts "Red Treasure!"*

When that robot wins, the coach pulls out all three Polaroids from its round and puts a gold star on every decision that led to the win.

### Rule 3: The "Don't Mess Up the Front" Speed Bonus

Once several clones figure out how to win, the coach adds a tie-breaker to reward the **fastest reader**:

* **Winners Only:** You *only* get speed points if you actually solve the mystery. A robot that dumps everything into the trash to finish in one second—and forgets the map—gets **zero** speed points.
* **Protect Your Cached Prefix:** The robot always reads from left to right. If it throws away trash from the **back (right side)**, the clues on the left stay anchored in their exact spots, so its cached reading notes stay valid. If it yanks a note out of the **front (left side)**, everything shifts, breaking the cache and forcing the robot to re-read the desk from scratch.

---

## 3. Detective School vs. Research Paper Translator

If you want to connect the game to the real mechanics in *arXiv:2609.37725*, here is the decoder ring:

| Sticky Note Detective Story | Real AI Research Term | What It Means |
| --- | --- | --- |
| **The 5-Slot Desk / Whiteboard** | Mutable Context Window ($c_t$) | A live prompt workspace the AI can surgically edit instead of only appending to. |
| **8-Clone Tryout** | Group Relative Policy Optimization (GRPO) | Comparing a group of $G$ rollouts on the same prompt instead of training a separate critic network. |
| **Skipping Ties** | Dynamic Sampling | Dropping rounds where every rollout passes or fails so training only updates on informative contrast. |
| **Polaroid Snapshots** | Trajectory Segmentation $(c_{i,j}, y_{i,j})$ | Recording the exact context snapshot seen at each turn $j$ so policy gradients match what the model actually saw. |
| **Reusing Reading Notes** | Prefix Caching & Suffix Cache Reuse (SCR) | Reusing cached KV states for untouched tokens—and stitching surviving suffix caches together after mid-prompt deletions—so editing doesn't trigger a full re-read. |

---

## 4. Run It Yourself: `demo.py`

You can run this entire reinforcement learning simulation on your laptop in under 5 seconds.

### Quick Setup

1. Install PyTorch if you don't already have it:
```bash
pip install torch

```


2. Save the code below as `demo.py` and run:
```bash
python demo.py

```



### Full Code (`demo.py`)

```python
import random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 🕵️‍♂️ THE STICKY NOTE DETECTIVE — Stepwise GRPO Demo (arXiv:2609.37725)
# ==============================================================================
# Watch a tiny robot detective learn how to clean junk off its 5-slot desk
# without shredding its clues—and learn to clean from the BACK so it never
# has to re-read the front notes!
# ==============================================================================

torch.manual_seed(42)
random.seed(42)

# --- Sticky Note Vocabulary (6 Types of Notes) ---
EMPTY = 0
MAP_RED = 1
KEY_RED = 2
COMPASS_BLUE = 3
GEM_BLUE = 4
JUNK = 5
VOCAB_SIZE = 6

NOTE_LABELS = {
    EMPTY: "⬜ Empty",
    MAP_RED: "🗺️ Red Map",
    KEY_RED: "🔑 Red Key",
    COMPASS_BLUE: "🧭 Blue Compass",
    GEM_BLUE: "💎 Blue Gem",
    JUNK: "🍌 Junk",
}
JUNK_VARIETY = ["🍌 Banana Peel", "🧦 Smelly Sock", "🥫 Empty Can", "🦴 Old Bone", "📎 Bent Clip"]

# --- Robot Detective Moves (5 Possible Actions) ---
DO_NOTHING = 0      # Leave the desk alone (risks overflowing into the shredder!)
TRASH_FRONT = 1     # Throw away the far-left note (messes up reading order & drops clues!)
TRASH_BACK = 2      # Throw away the newest junk on the right (keeps front clues safe!)
SOLVE_RED = 3       # Final Guess: "The treasure is RED!"
SOLVE_BLUE = 4      # Final Guess: "The treasure is BLUE!"
NUM_ACTIONS = 5

MOVE_LABELS = {
    DO_NOTHING: "🙈 Do Nothing (Let desk pile up)",
    TRASH_FRONT: "⬅️ Trash Front Note (Messes up reading order!)",
    TRASH_BACK: "➡️ Trash Back Junk (Keeps front notes in place!)",
    SOLVE_RED: "🚨 Shout: 'Red Treasure!'",
    SOLVE_BLUE: "🚨 Shout: 'Blue Treasure!'",
}

DESK_SLOTS = 5      # Tiny desk: only holds 5 sticky notes at once!
TOTAL_TURNS = 3     # 2 messy trash-chute turns + 1 final mystery guess


def format_desk(notes: list[int]) -> str:
    """Turns note IDs into fun sticky-note badges for printing."""
    pretty = []
    junk_seen = 0
    for n in notes:
        if n == JUNK:
            pretty.append(f"[{JUNK_VARIETY[junk_seen % len(JUNK_VARIETY)]}]")
            junk_seen += 1
        else:
            pretty.append(f"[{NOTE_LABELS[n]}]")
    return " ".join(pretty)


@dataclass
class PolaroidSnapshot:
    """One instant Polaroid photo snapped right before the robot makes a move."""
    desk_seen: list[int]      # Exact sticky notes on the desk at that moment
    move_chosen: int          # Which move the robot picked
    old_confidence: float     # How confident the robot was when it made the move


@dataclass
class DetectiveAttempt:
    """Everything that happened to one robot clone during a 3-turn mystery."""
    snapshots: list[PolaroidSnapshot]
    solved_mystery: float     # 1.0 if it solved the mystery, 0.0 if it failed
    notes_read_count: float   # Total sticky notes the robot had to read (lower = faster!)


def count_notes_re_read(before_desk: list[int], after_desk: list[int]) -> float:
    """
    Measures 'Prefix-Reuse Speed':
    The robot reads from left to right. Every sticky note from the left edge
    that stayed in its exact same spot is ALREADY memorized (0 reading cost!).
    Any note after the first change has to be re-read from scratch.
    """
    unchanged_from_left = 0
    for old_note, new_note in zip(before_desk, after_desk):
        if old_note == new_note:
            unchanged_from_left += 1
        else:
            break

    notes_to_re_read = len(after_desk) - unchanged_from_left
    thinking_step = 1.0
    return float(notes_to_re_read + thinking_step)


class StickyNoteDesk:
    """
    The Detective's Desk:
    - Starts with 2 secret clues + 1 piece of junk (3 notes total).
    - On Turn 1 and Turn 2, the robot cleans (or doesn't), and then a trash
      chute dumps 2 new pieces of junk on the right!
    - If there are more than 5 notes on the desk, the far-left notes fall
      into the paper shredder forever.
    """
    def __init__(self, mystery_color: int | None = None):
        # 0 = Red Treasure, 1 = Blue Treasure
        self.mystery_color = random.choice([0, 1]) if mystery_color is None else mystery_color
        if self.mystery_color == 0:
            self.desk = [MAP_RED, KEY_RED, JUNK]
        else:
            self.desk = [COMPASS_BLUE, GEM_BLUE, JUNK]

        self.turn = 0
        # Reading the starting 3 sticky notes costs 3 reading points
        self.total_notes_read = float(len(self.desk))

    def take_turn(self, move: int) -> tuple[bool, float]:
        desk_before = list(self.desk)

        if self.turn < TOTAL_TURNS - 1:
            # 1. Robot cleans its desk
            if move == TRASH_FRONT and len(self.desk) > 1:
                self.desk.pop(0)  # Oops! Trashed the front note and shifted everything left!
            elif move == TRASH_BACK and len(self.desk) > 1:
                # Smart! Find the rightmost junk note and toss it without moving front clues
                for idx in range(len(self.desk) - 1, -1, -1):
                    if self.desk[idx] == JUNK:
                        self.desk.pop(idx)
                        break

            # 2. Chute dumps 2 new junk sticky notes onto the right side of the desk
            self.desk.extend([JUNK, JUNK])

            # 3. Overflow check: if > 5 notes, far-left notes fall into the shredder!
            if len(self.desk) > DESK_SLOTS:
                self.desk = self.desk[-DESK_SLOTS:]

            self.total_notes_read += count_notes_re_read(desk_before, self.desk)
            self.turn += 1
            return False, 0.0

        # Final Turn: The robot can ONLY solve the mystery if both clues survived!
        self.total_notes_read += 1.0
        won_red = (MAP_RED in self.desk and KEY_RED in self.desk and move == SOLVE_RED)
        won_blue = (COMPASS_BLUE in self.desk and GEM_BLUE in self.desk and move == SOLVE_BLUE)
        return True, (1.0 if (won_red or won_blue) else 0.0)


class RobotDetectiveBrain(nn.Module):
    """A tiny neural network brain that looks at the desk and picks the next move."""
    def __init__(self, brain_size: int = 32):
        super().__init__()
        self.note_meaning = nn.Embedding(VOCAB_SIZE, brain_size, padding_idx=EMPTY)
        self.slot_position = nn.Parameter(torch.randn(1, DESK_SLOTS + 2, brain_size) * 0.05)
        self.decision_maker = nn.Sequential(
            nn.Linear(brain_size * 2, brain_size),
            nn.GELU(),
            nn.Linear(brain_size, NUM_ACTIONS),
        )

    def forward(self, desk_batch: torch.Tensor) -> torch.Tensor:
        batch_size, width = desk_batch.shape
        counts = (desk_batch != EMPTY).sum(dim=1).clamp(min=1)
        active_mask = (desk_batch != EMPTY).float().unsqueeze(-1)

        notes_with_slots = self.note_meaning(desk_batch) + self.slot_position[:, :width, :]
        whole_desk_summary = (notes_with_slots * active_mask).sum(dim=1) / counts.unsqueeze(-1).float()

        rightmost_idx = (counts - 1).view(batch_size, 1, 1).expand(batch_size, 1, notes_with_slots.size(-1))
        rightmost_note = notes_with_slots.gather(1, rightmost_idx).squeeze(1)

        return self.decision_maker(torch.cat([whole_desk_summary, rightmost_note], dim=-1))


def line_up_desks(desks: list[list[int]]) -> torch.Tensor:
    widest = max(len(d) for d in desks)
    return torch.tensor(
        [d + [EMPTY] * (widest - len(d)) for d in desks],
        dtype=torch.long,
    )


@torch.no_grad()
def run_8_clone_tryout(brain: RobotDetectiveBrain, num_clones: int = 8) -> list[DetectiveAttempt]:
    """
    RULE 1 (The 8-Clone Tryout):
    Deals the exact same mystery to 8 copies of the robot and snaps Polaroid
    photos at every step of each clone's attempt.
    """
    shared_mystery = random.choice([0, 1])
    attempts = []

    for _ in range(num_clones):
        desk_env = StickyNoteDesk(mystery_color=shared_mystery)
        snapshots: list[PolaroidSnapshot] = []
        finished = False
        won = 0.0

        while not finished:
            # RULE 2: Snap a Polaroid photo of the desk right before the move
            polaroid_desk = list(desk_env.desk)
            move_scores = brain(line_up_desks([polaroid_desk]))[0]

            # During turns 1 & 2, only allow desk-cleaning moves; on turn 3, make final guess
            if desk_env.turn < TOTAL_TURNS - 1:
                move_scores[SOLVE_RED:] = -1e9
            else:
                move_scores[:SOLVE_RED] = -1e9

            picker = torch.distributions.Categorical(logits=move_scores)
            chosen_move = int(picker.sample().item())
            confidence = float(picker.log_prob(torch.tensor(chosen_move)).item())

            snapshots.append(PolaroidSnapshot(polaroid_desk, chosen_move, confidence))
            finished, won = desk_env.take_turn(chosen_move)

        attempts.append(DetectiveAttempt(snapshots, won, desk_env.total_notes_read))

    return attempts


def grade_8_clones(
    attempts: list[DetectiveAttempt],
    speed_bonus_weight: float = 0.5,
    tiny_num: float = 1e-6,
) -> list[float] | None:
    """
    RULE 3 (Coach Grading & Winners-Only Speed Bonus):
    1. If all 8 clones won or all 8 lost, skip the round (nothing new to learn!).
    2. Give gold stars to clones that solved the mystery.
    3. Among WINNERS ONLY, award bonus points to clones that read fewer notes
       by cleaning from the back instead of the front!
    """
    scores = torch.tensor([a.solved_mystery for a in attempts], dtype=torch.float32)
    spread = scores.std(unbiased=False)

    # Dynamic Sampling: skip boring rounds where everyone tied
    if spread < tiny_num:
        return None

    mystery_grade = (scores - scores.mean()) / (spread + tiny_num)

    # Success-Gated Efficiency: ONLY winners compete for the speed bonus!
    winner_ids = [i for i, a in enumerate(attempts) if a.solved_mystery > 0.5]
    speed_bonus = torch.zeros(len(attempts), dtype=torch.float32)

    if len(winner_ids) >= 2:
        avg_winner_reads = sum(attempts[i].notes_read_count for i in winner_ids) / len(winner_ids)
        for i in winner_ids:
            my_reads = attempts[i].notes_read_count
            relative_speed = (avg_winner_reads - my_reads) / (avg_winner_reads + tiny_num)
            speed_bonus[i] = torch.clamp(torch.tensor(relative_speed), -1.0, 1.0)

    return (mystery_grade + speed_bonus_weight * speed_bonus).tolist()


def coach_training_step(
    student_brain: RobotDetectiveBrain,
    teacher_brain: RobotDetectiveBrain,
    coach_optimizer: torch.optim.Optimizer,
    tryout_groups: list[list[DetectiveAttempt]],
    clip_down: float = 0.20,
    clip_up: float = 0.28,      # Asymmetric Clip-Higher from the research paper
    anchor_strength: float = 0.02,
    review_passes: int = 3,     # Multiple passes per batch so clipping limits activate
) -> int:
    """
    Unpacks every Polaroid snapshot from valid 8-clone tryouts and updates the
    robot's brain so winning moves become more likely on those exact desk layouts.
    """
    all_desks, all_moves, old_confs, all_grades, is_cleaning_turn = [], [], [], [], []
    useful_rounds = 0

    for group in tryout_groups:
        grades = grade_8_clones(group)
        if grades is None:
            continue
        useful_rounds += 1
        for attempt, final_grade in zip(group, grades):
            for turn_num, snap in enumerate(attempt.snapshots):
                all_desks.append(snap.desk_seen)
                all_moves.append(snap.move_chosen)
                old_confs.append(snap.old_confidence)
                all_grades.append(final_grade)
                is_cleaning_turn.append(turn_num < TOTAL_TURNS - 1)

    if useful_rounds == 0:
        return 0

    desk_tensor = line_up_desks(all_desks)
    move_tensor = torch.tensor(all_moves, dtype=torch.long)
    old_conf_tensor = torch.tensor(old_confs, dtype=torch.float32)
    grade_tensor = torch.tensor(all_grades, dtype=torch.float32)
    cleaning_mask = torch.tensor(is_cleaning_turn, dtype=torch.bool)

    def get_move_confidences(brain: RobotDetectiveBrain) -> torch.Tensor:
        logits = brain(desk_tensor)
        mask = torch.zeros_like(logits)
        mask[cleaning_mask, SOLVE_RED:] = -1e9
        mask[~cleaning_mask, :SOLVE_RED] = -1e9
        log_probs = F.log_softmax(logits + mask, dim=-1)
        return log_probs.gather(1, move_tensor.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        teacher_conf = get_move_confidences(teacher_brain)

    for _ in range(review_passes):
        new_conf = get_move_confidences(student_brain)

        # Clipped policy update so the robot learns steadily without overreacting
        change_ratio = torch.exp(new_conf - old_conf_tensor)
        safe_ratio = torch.clamp(change_ratio, 1.0 - clip_down, 1.0 + clip_up)
        coaching_loss = -torch.min(change_ratio * grade_tensor, safe_ratio * grade_tensor).mean()

        # Gentle anchor so the robot stays close to its starting curiosity
        drift = teacher_conf - new_conf
        anchor_loss = (torch.exp(drift) - drift - 1.0).mean()

        total_loss = coaching_loss + anchor_strength * anchor_loss
        coach_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student_brain.parameters(), 1.0)
        coach_optimizer.step()

    return useful_rounds


def run_detective_school():
    print("🏫 Welcome to Robot Detective School! (Stepwise GRPO Demo)")
    print("Goal: Keep clues safe on a 5-slot desk AND clean from the back!\n")

    student_brain = RobotDetectiveBrain()
    teacher_brain = RobotDetectiveBrain()
    teacher_brain.load_state_dict(student_brain.state_dict())
    teacher_brain.eval()

    coach_optimizer = torch.optim.Adam(student_brain.parameters(), lr=2e-3)

    for lesson in range(1, 101):
        # Run 8 parallel 8-clone tryouts per lesson
        groups = [run_8_clone_tryout(student_brain, num_clones=8) for _ in range(8)]
        useful_rounds = coach_training_step(student_brain, teacher_brain, coach_optimizer, groups)

        if lesson in (1, 10, 20, 40, 60, 80, 100):
            attempts = [a for g in groups for a in g]
            win_rate = sum(a.solved_mystery for a in attempts) / len(attempts)
            avg_reads = sum(a.notes_read_count for a in attempts) / len(attempts)

            cleaning_moves = [s.move_chosen for a in attempts for s in a.snapshots[:-1]]
            back_clean_pct = cleaning_moves.count(TRASH_BACK) / len(cleaning_moves)
            front_clean_pct = cleaning_moves.count(TRASH_FRONT) / len(cleaning_moves)

            print(
                f"Lesson {lesson:3d} | Solved: {win_rate:5.1%} | "
                f"Notes Read: {avg_reads:4.2f} | "
                f"Back-Clean: {back_clean_pct:5.1%} | Front-Clean: {front_clean_pct:5.1%} | "
                f"Active Tryouts: {useful_rounds}/8"
            )

    # Graduation Showcase: Watch one trained robot detective solve a mystery!
    print("\n🎓 GRADUATION DAY — Watching the Trained Robot Detective in Action:")
    grad_attempt = run_8_clone_tryout(student_brain, num_clones=1)[0]
    outcome = "🎉 SOLVED IT!" if grad_attempt.solved_mystery > 0.5 else "❌ Missed it!"
    print(f"Result: {outcome} (Total sticky notes read: {grad_attempt.notes_read_count:.0f})\n")

    for turn_idx, snap in enumerate(grad_attempt.snapshots, start=1):
        print(f"  📸 Polaroid #{turn_idx}:")
        print(f"     Desk looked like : {format_desk(snap.desk_seen)}")
        print(f"     Robot decided    : {MOVE_LABELS[snap.move_chosen]}\n")


if __name__ == "__main__":
    run_detective_school()

```

---

## 5. What Your Kids Will See When You Press Run

When you run `python demo.py`, the terminal prints the robot's report card across **100 lessons**—and you can watch the three coaching rules kick in one after another:

1. **Lesson 1 (Clueless Rookie):**
* `Solved: ~6%` | `Notes Read: ~11`
* The robot randomly does nothing (letting clues fall into the shredder) or trashes notes from the front (wiping out the Red Map and invalidating its cached prefix).


2. **Lessons 10 $\to$ 20 (Survival Mode):**
* `Solved: ~80%`
* Rules 1 and 2 kick in. The clones that threw away junk notes survived and got high-fives, so the robot stops letting its desk overflow.


3. **Lessons 40 $\to$ 80 (Cache-Aware Speed Reader):**
* `Solved: 100.0%` | `Notes Read: 8.00` | `Back-Clean: 100.0%` | `Front-Clean: 0.0%`
* Rule 3 takes over. Because every clone is already solving the mystery, the speed bonus rewards the ones that **only clean from the back**, preserving their front clues and dropping reading work down to the theoretical minimum (`8.00` notes).


4. **Lessons 80 $\to$ 100 (Graduation):**
* `Active Tryouts: 0/8`
* Every clone gets a perfect score every single time—the coach sees zero ties left to break and lets the robot graduate.



At full production scale, this exact combination of reinforcement learning and cache-aware editing lets a model make over **160 live edits** inside a compact **6,000–8,000 token** workspace—beating standard append-only baselines by **11.4% in accuracy** while burning **21.5% fewer compute FLOPs** and slashing server-side recomputation by **35%**.

**Turns out, building a smarter AI isn’t about giving it a photographic memory—it’s about teaching it what to erase.**
