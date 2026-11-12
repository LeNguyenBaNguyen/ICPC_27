"""Export readable, verdict-free Markdown from the existing blinded JSONL packet."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "validation" / "blinded"
    cases = root / "cases"
    cases.mkdir(exist_ok=True)
    for line in (root / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        task = json.loads(line)
        if set(task) != {"task_id", "claim", "anchor_code", "previous_code", "current_code", "diff"}:
            raise ValueError("Unexpected fields in blinded packet")
        blocks = [f"# {task['task_id']}", task["claim"]]
        for key, label in (("current_code", "Code to assess"), ("previous_code", "Previous code"),
                           ("diff", "Change from previous code"), ("anchor_code", "Anchor code")):
            # Longer fence safely contains any triple-backticks inside a source string.
            longest = max((len(part) for part in __import__("re").findall(r"`+", task[key])), default=0)
            fence = "`" * max(3, longest + 1)
            language = "diff" if key == "diff" else "python"
            blocks.append(f"## {label}\n\n{fence}{language}\n{task[key]}\n{fence}")
        (cases / (task["task_id"] + ".md")).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Exported {len(list(cases.glob('V*.md')))} readable cases")


if __name__ == "__main__":
    main()
