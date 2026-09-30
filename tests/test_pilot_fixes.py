from jeff import cli
from jeff.generators import returns
from jeff.generators.llm.plan import plan_batches, render_writer_prompt


def test_returns_template_avoids_benchmark_phrase():
    for d in returns.generate(300, seed=5):
        assert "asked to send it back" not in d.state


def test_sentiment_prompt_asks_for_tone_not_rules():
    b = next(b for b in plan_batches(200, per_batch=10, seed=0) if b["family"] == "sentiment")
    text = render_writer_prompt(b, "/tmp/x.jsonl")
    assert "tone" in text and "not from a checklist" in text
    assert "Make the policy or criteria in the option descriptions specific" not in text


def test_writer_prompt_discourages_stock_phrasing():
    b = plan_batches(20, per_batch=10, seed=0)[0]
    text = render_writer_prompt(b, "/tmp/x.jsonl")
    assert "do not reuse stock phrasings" in text and "avoid stock policy phrasing" in text


def test_bigger_synth_defaults():
    p = cli.make_parser()
    assert p.parse_args(["synth", "plan", "--dir", "x", "--clusters", "1"]).per_batch == 25
    assert p.parse_args(["synth", "check-prepare", "--dir", "x"]).part_size == 250
    assert p.parse_args(["synth", "attrib-prepare", "--dir", "x"]).part_size == 250


def test_sentiment_prompt_omits_policy_phrasing_bullets():
    batches = plan_batches(200, per_batch=10, seed=0)
    sent = render_writer_prompt(next(b for b in batches if b["family"] == "sentiment"), "/tmp/x.jsonl")
    assert "none of these" not in sent and "stock policy phrasing" not in sent
    other = render_writer_prompt(next(b for b in batches if b["family"] != "sentiment"), "/tmp/x.jsonl")
    assert "none of these" in other and "stock policy phrasing" in other
