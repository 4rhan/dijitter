# Real-robot eval with karma

The policy runs on the GPU box as an HTTP server (`scripts/serve_karma.py`). The YAM robot PC runs
karma ([SRA-VJTI/karma](https://github.com/SRA-VJTI/karma)), which sends the cameras and joint state
to the server and plays back the returned action chunks. `karma rollout` asks "successful? [y/n]"
after each episode and saves every attempt as a LeRobot dataset.

The server converts the one thing that differs between karma's wire and our training data: the
gripper (karma sends 1=open, karma's recorded datasets and so the checkpoint use 0=open). Joints are
radians on both sides. Cameras are 848x480 on both sides.

## On the GPU box (in tmux)

1. Optional link test without the model, the arms hold still:
   `MOCK=1 bash scripts/serve_karma.sh`
2. Before the robot, check the real server on recorded frames (second tmux window, server running):
   `python scripts/check_karma_server.py` should end with `PASS`.
3. Start the server (last checkpoint of `pi05_two_cube_lora`; `CKPT=...` picks another):
   `bash scripts/serve_karma.sh`
   It prints this host's addresses and the karma flags. It needs about 8 GB of GPU memory, and jobs
   that wait for a free GPU (Sahil's queue) stay paused while it runs, so Ctrl-C it when done.

## On the robot PC (karma checkout)

Health check: `curl http://GPU_BOX_IP:8202/act` must show `"norm_tag": "pi05_two_cube"`.

```bash
uv run karma rollout --rig yam_bimanual \
  --interface left=can0 --interface right=can1 \
  --camera-serial top=TOP_SERIAL --camera-serial left_wrist=LEFT_WRIST_SERIAL --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --server http://GPU_BOX_IP:8202 --norm-tag pi05_two_cube \
  --repo-id local/two-cube-eval --episodes 10 --episode-seconds 60 --fps 30 --speed 0.5
```

- Same camera serials as in the demonstration recordings, same placement and framing.
- `--speed 0.5` plays each chunk at half the training rate for the first runs; go to 1.0 once it
  behaves. `--max-step-rad` (default 0.10) bounds each joint per tick.
- Every demo starts and ends at the zero pose (all joints within 0.05 rad of 0) with both grippers
  open. Karma parks at the zero pose between episodes; check the grippers are open before each one.
  The server prints the gripper it receives on the first request (`1.00/1.00` = both open).
- Task: the server always conditions on the dataset's single task, "put the two cubes in the box",
  whatever is typed at karma's prompt (`--use-instruction` on the server turns that off).
- Demos last 22-59 s, hence 60 s episodes.

## Afterwards

The rollout dataset (`local/two-cube-eval`) holds the executed states and actions, so smoothness can
be computed offline the same way as in sim (`scripts/metrics.py`).
