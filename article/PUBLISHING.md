# Publication notes

## Medium draft

- Title: From Linear Regression to a Neural Network: One Dataset, Six Steps
- Subtitle: Understand prediction, loss and learning—and see how the same ideas grow into a neural network.
- Suggested tags: Machine Learning, Artificial Intelligence, Python, Deep Learning, Education.
- `medium-article.md` is the editable source with TeX mathematical notation, intended for GitHub's Markdown rendering.
- Open `medium-article.html` for an offline preview with composed equations. `medium-ready.md` contains the same article with display equations replaced by PNG images and short inline symbols written in Unicode.
- For Medium, use `medium-ready.md` as the publishing copy. Insert the images from `assets/equations/` in numerical order at the indicated positions, plus the two plot figures from `assets/`. These equation images avoid relying on a TeX extension in the publishing editor. `equations.md` maps each image to its TeX source.
- Check headings, the results table, image captions and equation sizes in the Medium editor before publishing. Use the decision-boundary figure as the preview image if it crops well.
- After editing the article source, regenerate the preview and equation assets with `.venv/bin/python article/render_article.py`.
- The repository link is set to `https://github.com/clembnl/ai-learning-path`. Check both cloning and Code → Download ZIP while signed out.
- The article is a draft for the author's review; no Medium publication has been performed.

## GitHub repository

Owner: `clembnl`; name: `ai-learning-path`; visibility: public.

Description: Learn machine learning from scratch through seven notebooks: NumPy, scikit-learn and PyTorch on one synthetic dataset.

The prepared distribution excludes virtual environments, caches, Windows download metadata, generated outputs and the internal memory bank. Include the article and its assets.

The repository uses the MIT licence selected by the author on GitHub.
