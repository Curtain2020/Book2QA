# Book2QA: Enhancing Question-Answering with LLMs

The Book2QA project integrates multiple medium-scale language models to generate high-quality question-answering data from textbooks.
![Figure 1](./image/fingure1.png)

## Features

- **Book Data Preprocessing**: Converts textbook content into a structured format for further processing.
- **Question Generation and Filtering**: Generates diverse questions using LLMs, followed by filtering to enhance quality.
- **Answer Generation and Filtering**: Produces and refines answers using a fine-tuned model, ensuring relevance and clarity.

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Curtain2020/Book2QA
cd Book2QA
pip install -r requirements.txt
```

## Usage

Follow these steps to run the project:

1. **Data Preparation**: Ensure your data is formatted correctly as per the guidelines in `data/README.md`.
2. **Run Question Generation**:
    ```python
    python generate_and_merge_questions.py
    ```
3. **Run Answer Generation**:
    ```python
    python generate_best_answers.py
    ```

## Contributing

Contributions to Book2QA are welcome! Please refer to `CONTRIBUTING.md` for more details on how to submit pull requests, report issues, or make feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{book2qa2024,
  title={Book2QA: A Framework for Integrating LLMs to Generate High-quality QA Data from Textbooks},
  author={Anonymous},
  booktitle={EMNLP},
  year={2024}
}
```

## Acknowledgments

- Thanks to the contributors who have invested their time in improving the Book2QA framework.
- Special thanks to [List any funders, institutions, or other acknowledgments].
