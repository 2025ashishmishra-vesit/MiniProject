"""Insert an offline SUMO video comparison into the existing Streamlit dashboard.
Run from MiniProject root: python apply_patch.py
"""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
app = root / 'dashboard' / 'app.py'
if not app.is_file():
    raise SystemExit('Cannot find dashboard/app.py. Copy apply_patch.py to your MiniProject root first.')

text = app.read_text(encoding='utf-8')
start = '# BEGIN SUMO VIDEO COMPARISON'
end = '# END SUMO VIDEO COMPARISON'
if start in text:
    print('Video comparison already installed. No changes made.')
    raise SystemExit(0)

anchor = '# CONTROLLER COMPARISON'
anchor_pos = text.find(anchor)
if anchor_pos < 0:
    raise SystemExit('Could not find the Controller Comparison section. Your dashboard may differ from the GitHub version.')
# Insert directly before the comment header preceding CONTROLLER COMPARISON
comment_header = '# ---------------------------------------------------------\n'
pos = text.rfind(comment_header, 0, anchor_pos)
if pos < 0:
    pos = anchor_pos

block = '''# BEGIN SUMO VIDEO COMPARISON
# ---------------------------------------------------------
# RECORDED SIMULATION PLAYBACK
# ---------------------------------------------------------
st.header("🎬 SUMO Simulation Playback")
st.caption("Recorded 30-second demonstrations. Video controls are independent; playback is not synchronized with the metric charts below.")

VIDEO_DIR = PROJECT_ROOT / "dashboard" / "videos"
VIDEO_FILES = {
    "fixed": VIDEO_DIR / "fixed_time.mp4",
    "rl": VIDEO_DIR / "rl_controlled.mp4",
}

fixed_video_col, rl_video_col = st.columns(2, gap="large")

with fixed_video_col:
    st.subheader("🔴 Fixed-Time Control")
    if VIDEO_FILES["fixed"].is_file():
        st.video(str(VIDEO_FILES["fixed"]), format="video/mp4")
    else:
        st.warning("Fixed-time recording not found. Save it as dashboard/videos/fixed_time.mp4")

with rl_video_col:
    st.subheader("🟢 D3QN RL Control")
    if VIDEO_FILES["rl"].is_file():
        st.video(str(VIDEO_FILES["rl"]), format="video/mp4")
    else:
        st.warning("RL recording not found. Save it as dashboard/videos/rl_controlled.mp4")

st.info("Emergency vehicle signal priority is enabled only for the RL demonstration. Fixed-time traffic lights retain their normal schedule.")
st.divider()
# END SUMO VIDEO COMPARISON

'''

backup = app.with_suffix('.py.bak')
shutil.copy2(app, backup)
app.write_text(text[:pos] + block + text[pos:], encoding='utf-8')
(root / 'dashboard' / 'videos').mkdir(parents=True, exist_ok=True)
print(f'Updated: {app}')
print(f'Original saved: {backup}')
print('Copy your videos to dashboard/videos/fixed_time.mp4 and dashboard/videos/rl_controlled.mp4')
