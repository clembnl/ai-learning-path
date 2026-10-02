# Publication notes

## Medium draft

- Title: From Linear Regression to a Neural Network: One Dataset, Six Steps
- Subtitle: Build the maths in NumPy, verify it with scikit-learn, and discover what a hidden layer actually changes.
- Suggested tags: Machine Learning, Artificial Intelligence, Python, Deep Learning, Education.
- Paste `medium-article.md` into a Medium draft and upload the two PNG figures from `assets/` at the indicated positions. Use the decision-boundary figure as the preview image if it crops well.
- Medium may not preserve Markdown tables or mathematical notation reliably when pasted. This draft uses code blocks for equations; verify formatting and image captions in the editor.
- Replace the provisional repository paragraph with a verified public link before publishing. Check both cloning and Code → Download ZIP while signed out.
- The article is a draft for the author's review; no Medium publication has been performed.

## Planned GitHub repository

Owner: `clembnl`; name: `ai-learning-path`; visibility: public.

Description: Learn machine learning from scratch through seven notebooks: NumPy, scikit-learn and PyTorch on one synthetic dataset.

The prepared distribution excludes virtual environments, caches, Windows download metadata, generated outputs and the internal memory bank. Include the article and its assets.

After authenticating with the GitHub CLI, the prepared repository can be published with:

```bash
gh repo create clembnl/ai-learning-path --public --source . --remote origin --push
```

This command requires a local Git commit first. If the remote already exists, inspect it before choosing how to publish; do not overwrite an existing repository.
