from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COURSE_CHAPTERS = [
    "00_ml_foundations.md",
    "01_pytorch_foundations.md",
    "02_rl_math.md",
    "03_tabular_rl.md",
    "04_dqn.md",
    "05_policy_gradient/vpg.md",
    "05_policy_gradient/ppo.md",
    "05_policy_gradient/trpo.md",
    "05_policy_gradient/ddpg.md",
    "05_policy_gradient/td3.md",
    "05_policy_gradient/sac.md",
    "06_spinningup_track.md",
    "07_experiment_engineering.md",
    "08_export_deployment.md",
]

REQUIRED_COURSE_HEADINGS = [
    "## 学习目标",
    "## 为什么重要",
    "## 核心直觉",
    "## 数学与公式",
    "## 算法流程",
    "## 代码双索引",
    "## 实验任务",
    "## Debug Checklist",
    "## Spinning Up 对照",
    "## 学完标准",
]

LAB_DIRS = [
    "00_ml_foundations",
    "01_pytorch_foundations",
    "02_rl_math",
    "03_tabular_rl",
    "04_dqn",
    "05_policy_gradient",
    "05_policy_gradient/vpg",
    "05_policy_gradient/ppo",
    "05_policy_gradient/trpo",
    "05_policy_gradient/ddpg",
    "05_policy_gradient/td3",
    "05_policy_gradient/sac",
    "06_spinningup_track",
    "07_experiment_engineering",
    "08_export_deployment",
]


def test_course_chapters_have_learning_structure() -> None:
    for chapter in COURSE_CHAPTERS:
        text = (ROOT / "course" / chapter).read_text(encoding="utf-8")
        for heading in REQUIRED_COURSE_HEADINGS:
            assert heading in text, f"{chapter} missing {heading}"
        assert "labs/" in text, f"{chapter} must link to lab work"
        assert "src/drl_lab/" in text, f"{chapter} must link to engineering code"
        assert len(text) >= 3_000, f"{chapter} is too short for self-study use"


def test_labs_have_experiment_package_structure() -> None:
    for lab in LAB_DIRS:
        lab_dir = ROOT / "labs" / lab
        assert (lab_dir / "notes.md").is_file(), f"{lab} missing notes.md"
        assert (lab_dir / "exercises.md").is_file(), f"{lab} missing exercises.md"
        assert (lab_dir / "report.md").is_file(), f"{lab} missing report.md"
        assert (lab_dir / "code").is_dir(), f"{lab} missing code/"
        assert any((lab_dir / "code").glob("*.py")), f"{lab} code/ needs a demo script"
        notes = (lab_dir / "notes.md").read_text(encoding="utf-8")
        exercises = (lab_dir / "exercises.md").read_text(encoding="utf-8")
        report = (lab_dir / "report.md").read_text(encoding="utf-8")
        for heading in ["## 前置阅读", "## 实验目标", "## 代码入口", "## 提交产物"]:
            assert heading in notes, f"{lab} notes.md missing {heading}"
        for heading in ["## 纸笔题", "## 代码题", "## 观察题", "## 提交物"]:
            assert heading in exercises, f"{lab} exercises.md missing {heading}"
        for heading in ["## 本章目标", "## 运行命令", "## 我真正理解了什么"]:
            assert heading in report, f"{lab} report.md missing {heading}"


def test_document_templates_exist() -> None:
    for template in [
        "course_chapter_template.md",
        "lab_exercises_template.md",
        "lab_report_template.md",
    ]:
        assert (ROOT / "docs" / "templates" / template).is_file()


def test_learning_status_does_not_claim_unusable_drafts() -> None:
    for relative_path in ["course/index.md", "docs/roadmap.md"]:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "| draft" not in text
        assert "| planned" not in text
        assert "不能独立学习" not in text


def test_repository_learning_entrypoints_and_external_notice_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "中文 Deep Reinforcement Learning 自学课程实验室" in readme
    assert "course/index.md" in readme
    assert agents.startswith("# Repository Guidelines")
    assert 200 <= len(agents.split()) <= 400
    assert (ROOT / "external" / "spinningup" / "SOURCE.md").is_file()
    assert (ROOT / "external" / "spinningup" / "LICENSE").is_file()
