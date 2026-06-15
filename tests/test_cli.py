import json

from winnow.cli import main


def test_cli_json_input_and_output(tmp_path, capsys):
    chunks = [
        {"id": "a", "text": "unrelated gardening tips for the spring season"},
        {"id": "b", "text": "to reset your password go to settings and click reset password"},
    ]
    p = tmp_path / "chunks.json"
    p.write_text(json.dumps(chunks))

    rc = main(["compress", "-q", "reset password", "-b", "20", "-i", str(p),
               "--json", "--no-tiktoken", "--format", "json"])
    assert rc == 0

    out = json.loads(capsys.readouterr().out)
    assert out["kept_tokens"] <= 20
    assert "b" in [k["id"] for k in out["kept"]]


def test_cli_report_format(tmp_path, capsys):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(["alpha beta", "gamma delta", "epsilon zeta"]))
    rc = main(["compress", "-q", "alpha", "-b", "6", "-i", str(p), "--json",
               "--no-tiktoken", "--format", "report"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "reduction" in out
    assert "kept chunks" in out


def test_cli_text_input_paragraph_splitting(tmp_path, capsys):
    text = (
        "first paragraph about cats\n\n"
        "second paragraph about backoff and retries\n\n"
        "third about cooking"
    )
    p = tmp_path / "notes.txt"
    p.write_text(text)
    rc = main(["compress", "-q", "backoff retries", "-b", "16", "-i", str(p),
               "--no-tiktoken", "--format", "text"])
    assert rc == 0
    assert "backoff" in capsys.readouterr().out
