"""
merge_predictions.py - 合并新旧预测结果
保留旧模型对已完赛比赛的预测，使用新模型预测未完赛比赛
"""
import os, pickle, subprocess, tempfile
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load_git_predictions():
    """从git恢复旧模型预测结果"""
    result = subprocess.run(
        ["git", "show", "HEAD:data/wc_predictions.pkl"],
        capture_output=True, cwd=BASE_DIR
    )
    if result.returncode != 0:
        print("无法从git恢复旧预测")
        return None
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(result.stdout)
        tmp = f.name
    with open(tmp, "rb") as f:
        old = pickle.load(f)
    os.unlink(tmp)
    return old


def merge_predictions():
    """合并旧模型(已完赛) + 新模型(未完赛)"""
    # 加载旧模型预测 (git版本)
    old = _load_git_predictions()
    if old is None:
        print("错误: 无法加载旧模型预测")
        return False

    # 加载新模型预测 (当前版本)
    cur_path = os.path.join(DATA_DIR, "wc_predictions.pkl")
    if not os.path.exists(cur_path):
        print(f"错误: 新模型预测文件不存在: {cur_path}")
        return False

    with open(cur_path, "rb") as f:
        cur = pickle.load(f)

    print(f"旧模型: {len(old)} 场比赛 ({old['is_finished'].sum()} 已完赛)")
    print(f"新模型: {len(cur)} 场比赛 ({cur['is_finished'].sum()} 已完赛)")

    # 从旧模型提取已完赛比赛
    old_finished = old[old["is_finished"] == True].copy()
    # 从未完结比赛提取
    cur_unfinished = cur[cur["is_finished"] == False].copy()

    # 创建合并结果: 先用旧模型已完赛，再拼新模型未完赛
    merged = pd.concat([old_finished, cur_unfinished], ignore_index=True)

    # 按原始顺序排序（如果新模型中有排序信息）
    # 尝试用date排序
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.sort_values("date").reset_index(drop=True)
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    # 验证
    n_finished = merged["is_finished"].sum()
    n_unfinished = len(merged) - n_finished
    print(f"\n合并完成: {len(merged)} 场比赛")
    print(f"  已完赛(旧模型): {int(n_finished)} 场")
    print(f"  未完赛(新模型): {int(n_unfinished)} 场")

    # 检查旧模型准确率
    old_correct = old_finished["correct"].sum()
    old_total = len(old_finished)
    print(f"\n旧模型已完赛准确率: {old_correct}/{old_total} = {old_correct/old_total*100:.1f}%")

    if old_correct > 0:
        print("\n旧模型准确预测示例:")
        examples = old_finished[old_finished["correct"] == True].head(3)
        for _, r in examples.iterrows():
            print(f"  ✅ {r['home_team']} vs {r['away_team']}: {r['pred_result']}")
        examples_wrong = old_finished[old_finished["correct"] == False].head(3)
        if len(examples_wrong) > 0:
            print("旧模型错误预测示例:")
            for _, r in examples_wrong.iterrows():
                print(f"  ❌ {r['home_team']} vs {r['away_team']}: pred={r['pred_result']} actual={r.get('actual_result','?')}")

    # 保存合并结果
    backup_path = os.path.join(DATA_DIR, "wc_predictions_merged_backup.pkl")
    with open(backup_path, "wb") as f:
        pickle.dump(merged, f)

    # 替换主文件
    with open(cur_path, "wb") as f:
        pickle.dump(merged, f)

    print(f"\n已保存合并结果到: {cur_path}")
    print(f"备份到: {backup_path}")

    # 输出未完赛的新模型预测
    print(f"\n新模型预测 ({int(n_unfinished)} 场未完赛比赛):")
    for _, r in cur_unfinished.iterrows():
        print(f"  {r['home_team']:25s} vs {r['away_team']:25s} → {r['pred_label']} "
              f"(H:{float(r['pred_H'])*100:.0f}% D:{float(r['pred_D'])*100:.0f}% A:{float(r['pred_A'])*100:.0f}%)")

    return True


if __name__ == "__main__":
    merge_predictions()
