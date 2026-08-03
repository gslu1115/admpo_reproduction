import json

from admpo_repro.manifest import _source_revisions


def test_source_revisions_reads_git_free_snapshot_metadata(tmp_path):
    expected = {
        "repository_commit": "a" * 40,
        "vendor_admpo_commit": "b" * 40,
    }
    (tmp_path / ".source-revisions.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )

    assert _source_revisions(tmp_path) == expected


def test_source_revisions_ignores_invalid_json(tmp_path):
    (tmp_path / ".source-revisions.json").write_text("not json", encoding="utf-8")

    assert _source_revisions(tmp_path) == {}
