"""Tests for olmlx.chat.skills."""

from olmlx.chat.skills import SkillManager, _parse_skill_file, load_skills_from_dir


class TestParseSkillFile:
    def test_valid_file(self, tmp_path):
        p = tmp_path / "review.md"
        p.write_text(
            "---\nname: code-review\ndescription: Review code quality\n---\n\nCheck for bugs."
        )
        skill = _parse_skill_file(p)
        assert skill is not None
        assert skill.name == "code-review"
        assert skill.description == "Review code quality"
        assert skill.content == "Check for bugs."
        assert skill.path == p

    def test_missing_frontmatter(self, tmp_path):
        p = tmp_path / "no_front.md"
        p.write_text("Just some text without frontmatter.")
        assert _parse_skill_file(p) is None

    def test_missing_name(self, tmp_path):
        p = tmp_path / "no_name.md"
        p.write_text("---\ndescription: No name field\n---\n\nContent here.")
        assert _parse_skill_file(p) is None

    def test_multiline_content(self, tmp_path):
        p = tmp_path / "multi.md"
        p.write_text("---\nname: multi\n---\n\nLine one.\n\nLine two.\n\nLine three.\n")
        skill = _parse_skill_file(p)
        assert skill is not None
        assert "Line one." in skill.content
        assert "Line three." in skill.content

    def test_empty_body(self, tmp_path):
        p = tmp_path / "empty.md"
        p.write_text("---\nname: empty-skill\ndescription: Has no body\n---\n")
        skill = _parse_skill_file(p)
        assert skill is not None
        assert skill.name == "empty-skill"
        assert skill.content == ""

    def test_description_defaults_to_empty(self, tmp_path):
        p = tmp_path / "nodesc.md"
        p.write_text("---\nname: nodesc\n---\n\nSome content.")
        skill = _parse_skill_file(p)
        assert skill is not None
        assert skill.description == ""


class TestLoadSkillsFromDir:
    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.md").write_text("---\nname: alpha\n---\n\nAlpha content.")
        (tmp_path / "b.md").write_text("---\nname: beta\n---\n\nBeta content.")
        skills = load_skills_from_dir(tmp_path)
        assert "alpha" in skills
        assert "beta" in skills
        assert len(skills) == 2

    def test_ignores_non_md(self, tmp_path):
        (tmp_path / "valid.md").write_text("---\nname: valid\n---\n\nContent.")
        (tmp_path / "ignore.txt").write_text("---\nname: txt\n---\n\nContent.")
        (tmp_path / "ignore.json").write_text("{}")
        skills = load_skills_from_dir(tmp_path)
        assert len(skills) == 1
        assert "valid" in skills

    def test_empty_dir(self, tmp_path):
        skills = load_skills_from_dir(tmp_path)
        assert skills == {}

    def test_missing_dir(self, tmp_path):
        skills = load_skills_from_dir(tmp_path / "nonexistent")
        assert skills == {}

    def test_skips_malformed(self, tmp_path):
        (tmp_path / "good.md").write_text("---\nname: good\n---\n\nGood.")
        (tmp_path / "bad.md").write_text("No frontmatter at all.")
        skills = load_skills_from_dir(tmp_path)
        assert len(skills) == 1
        assert "good" in skills


class TestSkillManager:
    def _make_manager(self, tmp_path, files=None):
        if files:
            for name, content in files.items():
                (tmp_path / name).write_text(content)
        mgr = SkillManager(tmp_path)
        mgr.load()
        return mgr

    def test_load_and_list(self, tmp_path):
        mgr = self._make_manager(
            tmp_path,
            {
                "a.md": "---\nname: alpha\ndescription: Alpha skill\n---\n\nAlpha.",
                "b.md": "---\nname: beta\ndescription: Beta skill\n---\n\nBeta.",
            },
        )
        skills = mgr.list_skills()
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"alpha", "beta"}

    def test_get_by_name(self, tmp_path):
        mgr = self._make_manager(
            tmp_path,
            {
                "a.md": "---\nname: alpha\n---\n\nAlpha content.",
            },
        )
        skill = mgr.get_skill("alpha")
        assert skill is not None
        assert skill.content == "Alpha content."

    def test_get_unknown_returns_none(self, tmp_path):
        mgr = self._make_manager(
            tmp_path,
            {
                "a.md": "---\nname: alpha\n---\n\nAlpha.",
            },
        )
        assert mgr.get_skill("nonexistent") is None

    def test_index_text_contains_all_names(self, tmp_path):
        mgr = self._make_manager(
            tmp_path,
            {
                "a.md": "---\nname: alpha\ndescription: The alpha skill\n---\n\nA.",
                "b.md": "---\nname: beta\ndescription: The beta skill\n---\n\nB.",
            },
        )
        index = mgr.get_skill_index_text()
        assert "alpha" in index
        assert "beta" in index
        assert "The alpha skill" in index
        assert "The beta skill" in index

    def test_index_text_empty_when_no_skills(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.get_skill_index_text() == ""

    def test_tool_definition_has_enum(self, tmp_path):
        mgr = self._make_manager(
            tmp_path,
            {
                "a.md": "---\nname: alpha\n---\n\nA.",
                "b.md": "---\nname: beta\n---\n\nB.",
            },
        )
        tool_def = mgr.get_tool_definition()
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "use_skill"
        params = tool_def["function"]["parameters"]
        enum_values = params["properties"]["name"]["enum"]
        assert set(enum_values) == {"alpha", "beta"}

    def test_tool_definition_none_when_no_skills(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.get_tool_definition() is None

    def test_handle_use_skill_returns_content(self, tmp_path):
        mgr = self._make_manager(
            tmp_path,
            {
                "a.md": "---\nname: alpha\n---\n\nAlpha instructions here.",
            },
        )
        result = mgr.handle_use_skill({"name": "alpha"})
        assert "Alpha instructions here." in result

    def test_handle_use_skill_not_found(self, tmp_path):
        mgr = self._make_manager(
            tmp_path,
            {
                "a.md": "---\nname: alpha\n---\n\nA.",
            },
        )
        result = mgr.handle_use_skill({"name": "nonexistent"})
        assert "not found" in result.lower()
