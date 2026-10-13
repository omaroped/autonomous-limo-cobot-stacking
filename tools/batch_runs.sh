#!/bin/bash
# Runs the simulation N times without the Gazebo window and keeps every log, for example:
#   TIMEOUT=1500 LAUNCH_ARGS="place_stack_levels:=3 base_pin_enabled:=false exit_when_done:=true" tools/batch_runs.sh 10

export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=42
N="${1:-20}"
WS="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$WS/data/batch_$(date +%Y%m%d_%H%M%S)}"   # OUT=existing folder + START=N resumes a stopped batch
OUT="$(realpath -m "$OUT")"   # ensure absolute path before cd
START="${START:-1}"           # first run number (resume: START=11 OUT=data/batch_... tools/batch_runs.sh 20)
LAUNCH_ARGS="${LAUNCH_ARGS:-}"   # extra launch arguments, e.g. place_stack_levels:=2
TIMEOUT="${TIMEOUT:-600}"     # per-run wall-clock cap; a hung run must not stall the batch.

LOCK=/tmp/thesis_batch_runs.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "another batch is already running (pid $(cat "$LOCK")). Stop it first, or:"
    echo "    rm $LOCK      # only if you are sure nothing else is running"
    exit 1
fi
echo $$ > "$LOCK"
RECPID=""
TRUTHPID=""
stop_pid() {
    [ -n "$1" ] || return 0
    kill -INT "$1" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
    echo "(recorder $1 did not stop in 30 s -- killed)" >&2
    kill -9 "$1" 2>/dev/null; wait "$1" 2>/dev/null
}
cleanup() { rm -f "$LOCK"; [ -n "$RECPID" ] && kill -INT "$RECPID" 2>/dev/null; [ -n "$TRUTHPID" ] && kill -INT "$TRUTHPID" 2>/dev/null; }
trap cleanup EXIT
trap 'cleanup; "$WS/tools/kill_sim.sh" >/dev/null 2>&1; exit 130' INT TERM

mkdir -p "$OUT"

REPO="$WS"
SFX=""; [ "$START" -gt 1 ] && SFX="_from_run$START"   # a resumed part must not overwrite the first part
git -C "$REPO" rev-parse HEAD > "$OUT/git_head$SFX.txt"
git -C "$REPO" status --short > "$OUT/git_status$SFX.txt"
git -C "$REPO" diff           > "$OUT/git_diff$SFX.patch"
echo "TIMEOUT=$TIMEOUT LAUNCH_ARGS=$LAUNCH_ARGS START=$START" > "$OUT/batch_args$SFX.txt"
sha256sum "$WS"/install/limo_car/lib/limo_car/* "$WS"/install/limo_car/share/limo_car/config/*.yaml \
    > "$OUT/installed_sha256$SFX.txt" 2>/dev/null
echo "batch of $N runs -> $OUT"
echo "per-run timeout ${TIMEOUT}s. Ctrl-C to stop; finished runs are kept."
echo

cd "$WS" || exit 1

for i in $(seq "$START" "$N"); do
  for attempt in 1 2 3; do
    LOG="$OUT/run_$(printf '%02d' "$i").log"
    BAG="$OUT/bag_$(printf '%02d' "$i")"
    printf 'run %2d/%d ... ' "$i" "$N"

    "$WS/tools/kill_sim.sh" >/dev/null 2>&1
    sleep 2

    rm -rf "$BAG"
    ( set +u
      source /opt/ros/humble/setup.bash
      exec ros2 bag record -o "$BAG" /odom /cmd_vel /place_tag_pose /plan /amcl_pose
    ) > "$OUT/bag_$(printf '%02d' "$i").rec.log" 2>&1 &
    RECPID=$!

    NN=$(printf '%02d' "$i")
    rm -f "$OUT/events_$NN.jsonl"
    ( set +u
      source /opt/ros/humble/setup.bash
      exec python3 "$WS/tools/truth_recorder.py" "$OUT/truth_$NN.csv"
    ) > "$OUT/truth_$NN.rec.log" 2>&1 &
    TRUTHPID=$!

    (
        set +u
        source /opt/ros/humble/setup.bash
        source "$WS/install/setup.bash"
        export THESIS_EVENT_LOG="$OUT/events_$NN.jsonl"
        timeout "$TIMEOUT" ros2 launch limo_car nav_pick.launch.py \
            use_gzclient:=false use_rviz:=false $LAUNCH_ARGS
    ) > "$LOG" 2>&1

    stop_pid "$RECPID"; RECPID=""
    stop_pid "$TRUTHPID"; TRUTHPID=""

    if ! grep -q 'Nav2 goal accepted' "$LOG" && \
       grep -qE 'Nav2 goal rejected after|NavigateToPose action server not available' "$LOG"; then
        mv "$LOG" "$OUT/run_$(printf '%02d' "$i").startupfail$attempt.log"
        rm -rf "$BAG" "$OUT/truth_$NN.csv"* "$OUT/events_$NN.jsonl"*
        echo "Nav2 did not start (attempt $attempt) -- start-up failure, retrying"
        continue
    fi
    break
  done

    if [ ! -f "$LOG" ]; then
        echo "Nav2 failed to start 3 times -- run $i skipped (logs kept)"
        continue
    fi
    METRICS=$(grep -oE 'cycle [0-9]+ done: .*' "$LOG" | tail -1)
    STACK=$(grep -oE 'STACK STANDING: [0-9]+ level' "$LOG" | tail -1)
    if [ -n "$METRICS" ]; then
        echo "$METRICS  ${STACK:-[no final stack check]}"
    elif grep -q 'NOT A REAL GRIP' "$LOG"; then
        echo "aborted: grasp rejected (not a real grip)"
    elif grep -q 'refusing to place' "$LOG"; then
        echo "aborted: dock never closed, box kept"
    elif grep -q 'LOST in transit' "$LOG"; then
        echo "aborted: box lost in transit"
    else
        echo "no result line (timeout or crash) -- see $(basename "$LOG")"
    fi
done

"$WS/tools/kill_sim.sh" >/dev/null 2>&1
echo
echo "done. logs in $OUT"
echo "summarise with:  python3 tools/batch_summary.py $OUT"
