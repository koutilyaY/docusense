# Contributing to DocuSense

Thank you for considering contributing!

---

## Code of Conduct

- Be respectful and inclusive
- Critique ideas, not people
- Welcome diverse perspectives

---

## Ways to Contribute

### Report Bugs

Open an issue with:
- Clear title
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version)

### Suggest Features

Open an issue with:
- Feature summary
- Use case
- Proposed solution

### Submit Code

```bash
git clone https://github.com/koutilyaY/docusense.git
cd docusense
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Code Style**:
```bash
black src/ tests/
ruff check src/ tests/
pytest tests/ -v
```

**Git Workflow**:
1. `git checkout -b feature/your-feature`
2. Make commits
3. `git push origin feature/your-feature`
4. Open pull request

---

## License

By contributing, you agree contributions are licensed under **MIT License**.

Thank you! 🙏
