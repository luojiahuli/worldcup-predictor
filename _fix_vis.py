"""Remove the duplicate finished match review section from visualizer.py"""
with open("visualizer.py", "r") as f:
    content = f.read()

# 1. Find and remove the review generation block (lines 204-278)
#    Replace with simpler counter only
old_block_start = "    # === 已完成比赛复盘 ==="
old_block_end = '            finished_review_rows = "\\n".join(review_rows)'

# Find the exact positions
start = content.find(old_block_start)
if start == -1:
    print("ERROR: Could not find review block start")
    # Debug: show lines around 204
    lines = content.split("\n")
    for i in range(202, min(210, len(lines))):
        print(f"  L{i+1}: {repr(lines[i])}")
    exit(1)

# Find end of the review block (next function or major section)
end_markers = [
    "\n    # === 已完成比赛计数 ===",
    "\n    # === 比赛明细",
    "\n    # === 比赛列表",
    "\n    # === Latest",
]
end = -1
for marker in end_markers:
    pos = content.find(marker, start)
    if pos > 0:
        # Check if there's already a counter section we need to handle
        pass

# Find the next section after the review block
# Look for "count" references after the block end
rest = content[start:]
# Find the end of this block by looking for the next top-level comment
for search_str in ["\n    # === 比赛", "\n    # === 待预测", "\n    # === Latest"]:
    pos = rest.find(search_str, 1)
    if pos > 0:
        end = start + pos
        break

if end == -1:
    print("ERROR: Could not find end of review block")
    exit(1)

print(f"Found review block: {start} to {end}")
print(f"Content to remove ({end-start} chars):")
print(rest[:200])

# Replacement: just the counter
replacement = """    # === 已完成比赛计数 ===
    finished_count = 0
    correct_count = 0
    if predictions is not None and len(predictions) > 0:
        finished = predictions[predictions["is_finished"] == True]
        finished_count = len(finished)
        if finished_count > 0:
            correct_count = int((finished["actual_result"] == finished["pred_result"]).sum())"""

content = content[:start] + replacement + content[end:]

# 2. Remove the finished_review_section construction
old_review_section = """    # 比赛复盘section
    if finished_review_rows:
        finished_review_section = f'''<div class="cd cd-md" style="border-left:3px solid var(--green)">
  <div class="cd-h">
    <span class="cd-h-dot" style="background:var(--green)"></span>
    <h2>已完成比赛复盘</h2>
    <span class="cd-h-b" style="background:var(--green)">{finished_count} MATCHES &middot; {correct_count}/{finished_count} CORRECT</span>
  </div>
  <div style="padding:4px 0">
    {finished_review_rows}
  </div>
</div>'''
    else:
        finished_review_section = \"\"\"

"""
pos = content.find(old_review_section)
if pos >= 0:
    content = content[:pos] + content[pos+len(old_review_section):]
    print("Removed review section construction")
else:
    print("WARNING: Could not find review section construction")
    # Debug
    idx = content.find("比赛复盘section")
    if idx >= 0:
        print(f"Found at {idx}: {content[idx:idx+100]}")

# 3. Remove the HTML insertion point
old_html = "\n<!-- Match Review -->\n{finished_review_section}\n"
pos = content.find(old_html)
if pos >= 0:
    content = content[:pos] + "\n" + content[pos+len(old_html):]
    print("Removed HTML insertion")
else:
    print("WARNING: Could not find HTML insertion")
    idx = content.find("Match Review")
    if idx >= 0:
        print(f"Found at {idx}: {content[idx:idx+100]}")

with open("visualizer.py", "w") as f:
    f.write(content)

print("Done! visualizer.py updated")
