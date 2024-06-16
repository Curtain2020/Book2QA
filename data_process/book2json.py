import os
import json
import re
import timeit
from tqdm import tqdm
from book_structures import Book, Section, Subsection, Line, Generator

RESOURCE_PATH: str = r"./resources"  # 资源路径


def get_book_paths() -> list[str]:
    return [os.path.join(RESOURCE_PATH, path) for path in os.listdir(RESOURCE_PATH)]


def book2json(
    save_path: str = "struct_books.json", skip_table: bool = False
) -> list[Book]:
    book_paths: list[str] = get_book_paths()
    books: list[Book] = []
    for path in book_paths:
        with Book(path, skip_table=skip_table) as book:
            # 初始化
            book.skip_page = True  # 跳过单元开头和单元结尾
            section: Section = Section()
            subsection: Subsection = Subsection()
            for page in tqdm(book, desc=book.name):
                if re.search(r"单元小结|单 元 小 结", page.get_text(skip_table=False)):
                    book.skip_page = True

                lines: Generator[Line, None, None] = page.lines

                try:
                    line: Line = next(lines)
                    while True:
                        next_line: Line = next(lines)  # 有可能出现两行的标题
                        if line.font == book.section_font:
                            book.skip_page = False
                            book.sections.append(section := Section(name=line.text))
                            if next_line.font == book.section_font:
                                section.name += next_line.text
                                next_line = next(lines)

                        elif line.font == book.subsection_font and section:
                            section.subsections.append(
                                subsection := Subsection(name=line.text)
                            )
                            if next_line.font == book.subsection_font:
                                subsection.name += next_line.text
                                next_line = next(lines)

                        elif not book.skip_page:
                            if not section.subsections:
                                section.subsections.append(
                                    subsection := Subsection(name=section.name)
                                )
                            subsection.paragraphs += line.text
                            if line.is_start_of_paragraph(next_line):
                                subsection.paragraphs += "\n"
                        line = next_line
                except StopIteration:
                    try:
                        # 还有最后一行的文本
                        if not book.skip_page:
                            subsection.paragraphs += line.text
                    except NameError:
                        # line未定义
                        pass
                # print(book, "\n")
            books.append(book)

    if save_path:
        # 保存为json
        books_json: dict[str, dict] = {}
        for book in books:
            books_json[book.name] = book.to_dict()
        file_name = "struct_books.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(books_json, f, ensure_ascii=False)

    return books


if __name__ == "__main__":
    print(
        timeit.timeit(
            "book2json(skip_table = True)",
            setup="from __main__ import book2json",
            number=1,
        ),
        "秒",
    )
