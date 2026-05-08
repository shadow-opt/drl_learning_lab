from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHAPTERS = [
    "00_ml_foundations",
    "01_pytorch_foundations",
    "02_rl_math",
    "03_tabular_rl",
    "04_dqn",
    "05_policy_gradient",
    "06_spinningup_track",
    "07_experiment_engineering",
    "08_export_deployment",
]

BEGINNER_READY = [
    "00_ml_foundations",
    "01_pytorch_foundations",
    "02_rl_math",
]

BEGINNER_READY_DEMOS = {
    "00_ml_foundations": [
        "linear_regression.py",
        "binary_classifier.py",
        "image_classifier.py",
    ],
    "01_pytorch_foundations": ["mlp_export_demo.py"],
    "02_rl_math": ["value_iteration_demo.py"],
}

POLICY_GRADIENT_ALGOS = ["vpg", "ppo", "trpo", "ddpg", "td3", "sac"]

REQUIRED_CHAPTER_FILES = [
    "README.md",
    "lesson.md",
    "walkthrough.md",
    "lab.md",
    "exercises.md",
    "hints.md",
    "solutions.md",
    "report.md",
]


def test_curriculum_entrypoints_exist() -> None:
    assert (ROOT / "curriculum" / "README.md").is_file()
    assert (ROOT / "curriculum" / "status.md").is_file()
    assert (ROOT / "curriculum" / "glossary.md").is_file()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "curriculum/README.md" in readme
    assert "不默认会机器学习、PyTorch 或强化学习" in readme


def test_chapters_have_training_camp_structure() -> None:
    for chapter in CHAPTERS:
        chapter_dir = ROOT / "curriculum" / "chapters" / chapter
        assert chapter_dir.is_dir(), f"missing chapter {chapter}"
        for filename in REQUIRED_CHAPTER_FILES:
            assert (chapter_dir / filename).is_file(), f"{chapter} missing {filename}"
        assert (chapter_dir / "code").is_dir(), f"{chapter} missing code/"
        assert any((chapter_dir / "code").glob("*.py")), f"{chapter} needs a demo script"


def test_beginner_ready_chapters_have_real_support_material() -> None:
    for chapter in BEGINNER_READY:
        chapter_dir = ROOT / "curriculum" / "chapters" / chapter
        support_files = [
            "README.md",
            "lesson.md",
            "walkthrough.md",
            "lab.md",
            "hints.md",
            "solutions.md",
        ]
        for filename in support_files:
            text = (chapter_dir / filename).read_text(encoding="utf-8")
            assert len(text) >= 200, f"{chapter}/{filename} is too thin"
        lab = (chapter_dir / "lab.md").read_text(encoding="utf-8")
        assert "期望" in lab, f"{chapter}/lab.md must explain expected output"
        assert "异常情况" in lab, f"{chapter}/lab.md must explain failure modes"
        solutions = (chapter_dir / "solutions.md").read_text(encoding="utf-8")
        assert "参考答案" in solutions, f"{chapter}/solutions.md needs answers"
        walkthrough = (chapter_dir / "walkthrough.md").read_text(encoding="utf-8")
        assert "逐行走读" in walkthrough, f"{chapter}/walkthrough.md needs line-by-line guidance"


def test_beginner_ready_chapters_keep_tutorial_depth() -> None:
    for chapter in BEGINNER_READY:
        chapter_dir = ROOT / "curriculum" / "chapters" / chapter

        readme = (chapter_dir / "README.md").read_text(encoding="utf-8")
        assert "先修数学" in readme, f"{chapter}/README.md must state math prerequisites"
        assert "预计用时" in readme, f"{chapter}/README.md must state expected workload"
        assert "Problem Set" in readme, f"{chapter}/README.md must describe problem set"
        assert "完成标准" in readme, f"{chapter}/README.md must define completion criteria"

        lesson = (chapter_dir / "lesson.md").read_text(encoding="utf-8")
        assert "形式化定义" in lesson, f"{chapter}/lesson.md must include formal definitions"
        assert "推导" in lesson, f"{chapter}/lesson.md must include derivations"
        assert "例题" in lesson, f"{chapter}/lesson.md must include worked examples"
        assert "常见误解" in lesson, f"{chapter}/lesson.md must teach misconceptions"
        assert "最小例子" in lesson, f"{chapter}/lesson.md must include minimal examples"
        assert "和 DRL 的关系" in lesson, f"{chapter}/lesson.md must connect to DRL"

        walkthrough = (chapter_dir / "walkthrough.md").read_text(encoding="utf-8")
        assert "逐行走读" in walkthrough, f"{chapter}/walkthrough.md must be line-oriented"
        assert "公式对应" in walkthrough, f"{chapter}/walkthrough.md must map code to formulas"
        assert "shape" in walkthrough, f"{chapter}/walkthrough.md must discuss shapes"
        for demo_name in BEGINNER_READY_DEMOS[chapter]:
            assert demo_name in walkthrough, f"{chapter}/walkthrough.md must cover {demo_name}"

        lab = (chapter_dir / "lab.md").read_text(encoding="utf-8")
        assert "期望输出" in lab, f"{chapter}/lab.md must show expected output"
        assert "逐项解释" in lab, f"{chapter}/lab.md must explain output items"
        assert "异常情况" in lab, f"{chapter}/lab.md must discuss abnormal cases"
        assert "artifact" in lab, f"{chapter}/lab.md must explain artifacts"
        assert "Ablation" in lab, f"{chapter}/lab.md must include ablations"

        exercises = (chapter_dir / "exercises.md").read_text(encoding="utf-8")
        assert "概念题" in exercises, f"{chapter}/exercises.md must include concept questions"
        assert "手算题" in exercises, f"{chapter}/exercises.md must include hand calculations"
        assert "推导题" in exercises, f"{chapter}/exercises.md must include derivations"
        assert "代码题" in exercises, f"{chapter}/exercises.md must include code questions"
        assert "实验诊断题" in exercises, f"{chapter}/exercises.md must include diagnostics"

        solutions = (chapter_dir / "solutions.md").read_text(encoding="utf-8")
        assert "参考答案" in solutions, f"{chapter}/solutions.md needs reference answers"
        assert "评分要点" in solutions, f"{chapter}/solutions.md must include grading rubrics"
        assert "为什么" in solutions, f"{chapter}/solutions.md must explain why"
        assert "常见错误" in solutions, f"{chapter}/solutions.md must discuss common mistakes"
        assert "如何验证" in solutions, f"{chapter}/solutions.md must show verification"

        report = (chapter_dir / "report.md").read_text(encoding="utf-8")
        assert "推导检查" in report, f"{chapter}/report.md must prompt derivation checks"
        assert "实验记录" in report, f"{chapter}/report.md must prompt experiment logs"
        assert "证据" in report, f"{chapter}/report.md must prompt evidence"
        assert "卡住点" in report, f"{chapter}/report.md must prompt blockers"
        assert "下一步" in report, f"{chapter}/report.md must prompt next steps"


def test_policy_gradient_subchapters_have_structure() -> None:
    root = ROOT / "curriculum" / "chapters" / "05_policy_gradient"
    for algo in POLICY_GRADIENT_ALGOS:
        algo_dir = root / algo
        assert algo_dir.is_dir(), f"missing {algo} subchapter"
        for filename in REQUIRED_CHAPTER_FILES:
            assert (algo_dir / filename).is_file(), f"{algo} missing {filename}"


def test_old_course_and_labs_directories_removed() -> None:
    assert not (ROOT / "course").exists()
    assert not (ROOT / "labs").exists()


def test_no_stale_course_or_labs_references_in_docs_and_tests() -> None:
    checked_roots = [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "curriculum",
        ROOT / "docs",
        ROOT / "tests",
    ]
    stale = []
    for root in checked_roots:
        files = [root] if root.is_file() else list(root.rglob("*"))
        for path in files:
            if not path.is_file() or path.suffix not in {".md", ".py"}:
                continue
            if path.relative_to(ROOT) == Path("tests/test_learning_structure.py"):
                continue
            text = path.read_text(encoding="utf-8")
            if "course/" in text or "labs/" in text:
                stale.append(str(path.relative_to(ROOT)))
    assert not stale, f"stale course/labs references: {stale}"


def test_external_spinningup_notice_exists() -> None:
    assert (ROOT / "external" / "spinningup" / "SOURCE.md").is_file()
    assert (ROOT / "external" / "spinningup" / "LICENSE").is_file()
